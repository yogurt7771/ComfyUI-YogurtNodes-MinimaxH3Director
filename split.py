"""自动分段: split a long source stretch into H3 generation windows that join cleanly.

Model, in 24 fps frames: one take per segment. H3 rewrites a single take in step with its source, but a cut inside a
window is mostly not followed: dropped, moved to the window's start, or replaced by invented motion. So every sure cut
is a boundary where the next segment starts fresh, and only takes shorter than BRIEF are grouped (see ``takes``). A
group up to ``longest`` frames is one segment however short; a longer one is cut into ``shortest``..``longest``
segments at points inside a take, where the next segment carries the previous one's last ``overlap`` frames, or at a
sure cut inside a group of brief takes. A carried boundary needs those frames and ``MARGIN`` after it in one take and
clear of fades and flashes, and the segment before it exactly on the 17k+5 grid, so what is carried is its true end.
Cost: motion over each carried tail and length away from the target. A shortest path over boundaries is exact: it
returns the cheapest plan, or None when none exists.
"""

import statistics

import torch

from .project import FPS

MARGIN = 12
THUMB = (64, 36)
# Cuts are read from each frame's HSV difference to the previous one (hue on the colour wheel, so a cut between two
# similarly lit scenes still shows), above CUT_MIN. A sure cut stands SURE_RATIO above the mean of the SURE_WINDOW
# frames on each side: motion rises over several frames, a cut is one spike. A possible cut only stands POSSIBLE_RATIO
# above the median of the POSSIBLE_WINDOW frames on each side; it catches quiet cuts along with some sudden motion.
# Only sure cuts may become hard-cut boundaries; carried tails stay clear of both.
CUT_MIN = 0.06
SURE_RATIO, SURE_WINDOW = 3.0, 3
POSSIBLE_RATIO, POSSIBLE_WINDOW = 2.1, 12
CUT_GAP = 6
# Hue and saturation are unstable near black (a fade's last frame read as a cut); they count fully above this value.
DARK = 0.15
# Mean brightness stepping this much per frame for FADE_RUN frames in a row is a fade or flash.
FADE_STEP = 0.012
FADE_RUN = 3
# Takes shorter than this are too brief for a window of their own: a run of them is one group, like a fast-cut montage.
BRIEF = FPS
# A cover is a still the uploader put in front of the video for its thumbnail, cut from a frame further in: a first take
# shorter than COVER_TAKE whose frames all repeat one frame at least COVER_GAP later within COVER_MATCH (mean difference
# of the thumbnails; a Douyin cover measured 0.015, ordinary frames never came closer than 0.035 to a frame 1 s away).
COVER_TAKE = FPS // 2
COVER_GAP = FPS
COVER_MATCH = 0.025


def _hsv(frames):
    high, low = frames.max(-1).values, frames.min(-1).values
    spread = high - low
    r, g, b = ((high[..., None] - frames) / spread.clamp_min(1e-6)[..., None]).unbind(-1)
    hue = torch.where(frames[..., 0] == high, b - g, torch.where(frames[..., 1] == high, 2 + r - b, 4 + g - r))
    hue = torch.where(spread > 1e-6, hue / 6 % 1.0, 0.0)
    return hue, torch.where(high > 1e-6, spread / high.clamp_min(1e-6), 0.0), high


def analyze(frames):
    """(change, cuts, possible, fades) of [N, H, W, 3] frames in 0..1: each frame's pixel change from the previous one,
    the frames that surely start a new take, the frames that may, and the frames inside a fade or flash."""
    count = frames.shape[0]
    change = [0.0] + (frames[1:] - frames[:-1]).abs().mean((1, 2, 3)).tolist()
    hue, saturation, value = _hsv(frames)
    turn = (hue[1:] - hue[:-1]).abs()
    lit = (torch.minimum(value[1:], value[:-1]) / DARK).clamp(max=1.0)
    shift = (torch.minimum(turn, 1 - turn) * 2 + (saturation[1:] - saturation[:-1]).abs()) * lit + (value[1:] - value[:-1]).abs()
    content = [0.0] + (shift / 3).mean((1, 2)).tolist()

    def around(i, window):
        return content[max(1, i - window):i] + content[i + 1:i + 1 + window] or [0.0]

    def keep_strongest(found, i):
        if found and i - found[-1] < CUT_GAP:
            if content[i] <= content[found[-1]]:
                return
            found.pop()
        found.append(i)

    cuts, possible = [], []
    for i in range(1, count):
        if content[i] <= CUT_MIN:
            continue
        if content[i] > SURE_RATIO * statistics.mean(around(i, SURE_WINDOW)):
            keep_strongest(cuts, i)
        elif content[i] > POSSIBLE_RATIO * statistics.median(around(i, POSSIBLE_WINDOW)):
            keep_strongest(possible, i)
    possible = [i for i in possible if all(abs(i - c) >= CUT_GAP for c in cuts)]
    luma = frames.mean((1, 2, 3)).tolist()
    fades, run = set(), []
    for i in range(1, count):
        if abs(luma[i] - luma[i - 1]) > FADE_STEP and i not in cuts:
            run = run + [i] if run and run[-1] == i - 1 else [i]
            if len(run) >= FADE_RUN:
                fades.update(range(run[0] - 1, i + 1))
    return change, cuts, possible, sorted(fades)


def takes(count, cuts):
    """[(start, end)] covering frames 0..count, one per take between ``cuts``. Consecutive takes shorter than BRIEF are
    one group; a group still shorter than BRIEF opens the next one, where H3 keeps it in step (the last one closes the
    group before it)."""
    bounds = [0, *cuts, count]
    runs = []
    for a, b in zip(bounds, bounds[1:]):
        if b - a < BRIEF and runs and runs[-1][2]:
            runs[-1][1] = b
        else:
            runs.append([a, b, b - a < BRIEF])
    groups = []
    for a, b, _ in runs:
        if groups and groups[-1][1] - groups[-1][0] < BRIEF:
            groups[-1][1] = b
        else:
            groups.append([a, b])
    if len(groups) > 1 and groups[-1][1] - groups[-1][0] < BRIEF:
        end = groups.pop()[1]
        groups[-1][1] = end
    return [(a, b) for a, b in groups]


def plan(count, change, cuts, possible, fades, shortest, longest, overlap, segments=0):
    """[(start, frames, carried)] covering frames 0..count, or None when the constraints cannot all hold. Carried tails
    stay clear of ``cuts``, ``possible`` cuts and ``fades``. ``segments`` fixes the number of segments; 0 lets the cost
    choose. With ``overlap`` 0 nothing carries and boundaries inside a take are plain joins."""
    cut_set, marks, fade_set = set(cuts), set(cuts) | set(possible), set(fades)
    group = [None] * count
    for a, b in takes(count, cuts):
        group[a:b] = [(a, b)] * (b - a)
    scale = statistics.median(change[1:]) or 1e-6
    cuts_before, fades_before, change_before = [0], [0], [0.0]
    for f in range(count):
        cuts_before.append(cuts_before[-1] + (f in marks))
        fades_before.append(fades_before[-1] + (f in fade_set))
        change_before.append(change_before[-1] + change[f])

    def carry_cost(q):
        """Tail motion of an in-take boundary at q; None when the carried tail and the frames after q are not one take
        clear of fades."""
        a, b = q - overlap, q + MARGIN
        if a < 0 or b > count or cuts_before[b] - cuts_before[a + 1] or fades_before[b] - fades_before[a]:
            return None
        return (change_before[b] - change_before[a]) / ((b - a) * scale)

    target = count / segments if segments else (shortest + longest) / 2
    carry = [carry_cost(q) for q in range(count)]
    # states[boundary][(carried head, segments so far)] = (cost, previous state). Boundaries only move forward, so
    # visiting them in order settles each one before it is extended. Without a fixed count, k stays 0.
    states = [{} for _ in range(count)]
    states[0][(0, 0)] = (0.0, None)
    finals = []
    for p in range(count):
        a, b = group[p]
        for (head, k), (cost, _) in states[p].items():
            here, nk = (p, head, k), k + 1 if segments else 0
            # The rest of the group in one segment: a whole group of any length, or the last piece of a long one.
            moves = [(b, 0, 0.0)] if b - p <= longest and (p == a or b - p >= shortest) else []
            for q in range(p + shortest, min(p + longest, b - shortest) + 1):
                if q in cut_set:
                    moves.append((q, 0, 0.0))
                if carry[q] is not None and (not overlap or (head + q - p) % 17 == 5):
                    moves.append((q, overlap, carry[q]))
            for q, carried, extra in moves:
                total = cost + ((q - p - target) / target) ** 2 + extra
                if q == count:
                    if not segments or nk == segments:
                        finals.append((total, here))
                elif segments and nk >= segments:
                    continue
                elif (carried, nk) not in states[q] or total < states[q][(carried, nk)][0]:
                    states[q][(carried, nk)] = (total, here)
    if not finals:
        return None
    _, state = min(finals)
    bounds = []
    while state is not None:
        p, head, k = state
        bounds.append((p, head))
        state = states[p][(head, k)][1]
    bounds.reverse()
    ends = [p for p, _ in bounds[1:]] + [count]
    return [(p, end - p, bool(head)) for (p, head), end in zip(bounds, ends)]


def find_cover(frames, cuts):
    """(frames, repeated frame) of a cover the stretch opens with, or None."""
    if not cuts or cuts[0] >= COVER_TAKE or frames.shape[0] <= cuts[0] + COVER_GAP:
        return None
    count, later = cuts[0], frames[cuts[0] + COVER_GAP:]
    worst = torch.stack([(later - frames[i]).abs().mean((1, 2, 3)) for i in range(count)]).amax(0)
    k = int(worst.argmin())
    return (count, count + COVER_GAP + k) if float(worst[k]) < COVER_MATCH else None


def split_source(media, item, start, count, shortest, longest, overlap, segments=0, detect_cover=True):
    """Auto-segment ``count`` frames of source ``item`` from ``start``: {"cover": {frames, copy} or None, "segments":
    [{start, frames, carry, cuts}], "note": text}. With ``detect_cover``, a cover at the start of the video is left out
    of the segments, which begin after it; ``copy`` is the frame it repeats. ``cuts`` are a segment's own frames where
    the source cuts."""
    frames = media.video(item, start, count, *THUMB)
    change, cuts, possible, fades = analyze(frames)
    cover = find_cover(frames, cuts) if detect_cover and start == 0 else None
    if cover:
        change, cuts, possible, fades = analyze(frames[cover[0]:])
        start, count = start + cover[0], count - cover[0]
    note = ""
    result = plan(count, change, cuts, possible, fades, shortest, longest, overlap, segments)
    if result is None and not segments:
        # A stretch only slightly longer than ``longest`` can have no valid split; it renders as one segment instead.
        relaxed = longest + overlap + MARGIN
        result = plan(count, change, cuts, possible, fades, shortest, relaxed, overlap)
        if result is not None:
            note = f"按 {shortest / FPS:g}–{longest / FPS:g} 秒找不到合格的切点，最长放宽到 {relaxed / FPS:g} 秒。"
    if result is None:
        wanted = f"分成 {segments} 段" if segments else "切分"
        raise ValueError(f"{count / FPS:.2f} 秒的源视频按每段 {shortest / FPS:g}–{longest / FPS:g} 秒无法{wanted}：调大最长时长、调小最短时长或改段数。")
    return {"cover": cover and {"frames": cover[0], "copy": cover[1]},
            "segments": [{"start": start + a, "frames": n, "carry": carried, "cuts": [c - a for c in cuts if a < c < a + n]}
                         for a, n, carried in result], "note": note}
