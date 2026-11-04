// Shared project model for the Yogurt H3 director UI. Frame math mirrors project.py.
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

export const FPS = 24;
export const NODE = "YogurtH3Director";
export const STAGES = {
    encoding: "编码条件", sampling: "生成中", repainting: "重绘中", upscaling: "放大中", refining: "二采中", rerendering: "低噪重绘中",
    segmenting: "识别遮罩", decoding: "解码中", detailing: "放大贴回中", superres: "超分中", rewriting: "改写提示词", rewritten: "已改写",
    complete: "完成",
};
// Phases of a run in the order the backend works through them (Render.announce), with a rough share of the time one shot
// spends in each, for the overall progress bar.
export const PHASES = {
    encoding: ["编码", 1], sampling: ["生成", 6], two_pass: ["两段式", 3], rerender: ["低噪重绘", 3],
    decoding: ["解码", 2], detailing: ["放大贴回", 3], assembling: ["成片", 1],
};

export const align = n => { n = Math.max(5, Math.round(n)); while (n % 17 !== 5) n++; return n; };
export const uid = () => Math.random().toString(36).slice(2, 9);
export const clamp = (v, lo, hi) => Math.min(Math.max(v, lo), hi);
export const secs = frames => `${(frames / FPS).toFixed(2)}s`;
export const clone = value => JSON.parse(JSON.stringify(value));

/** Every image or video the rewrite AI could see: shown to it by default, or kept private with optional words instead. */
export const SHARED = { share: true, description: "" };

const newSource = () => ({ media: null, start: 0, regen: false, strength: 1, ...SHARED });

export function newShot(kind = "generate") {
    return {
        id: uid(), kind, frames: 121, prompt: "", continue: kind === "generate", seed: 0, keyframes: [],
        refs: { images: [], videos: [], audios: [] },
        source: kind === "source" ? newSource() : null,
        audio: kind === "source" ? "source" : "generate", repaints: [], refine: true, rewrite: true,
        inherit_refs: true, inherit_prompt: true, ref_strength: 1,
    };
}

export function newProject() {
    return {
        version: 1, soundtrack: null, refs: { images: [], videos: [], audios: [] }, story: "", rewrite: { enabled: false, notes: "", seed: 0 },
        swap: { enabled: false, what: "", into: "", track: false, objects: 1 },
        shots: [newShot()],
    };
}

// Reference images carry how they meet the canvas (older projects stored bare media), videos the frame they start at.
const refsOf = refs => ({ audios: [], ...refs,
    images: (refs?.images || []).map(m => (m.media ? { fit: "auto", ...SHARED, ...m } : { media: m, fit: "auto", ...SHARED })),
    videos: (refs?.videos || []).map(v => ({ audio: false, start: 0, ...SHARED, ...v })) });

/**
 * A 局部重绘 layer keeps every mask source's settings, so switching source loses nothing: the painted mask, the masks
 * socket, the SAM 3 text target and the box (fixed, or framing a face). ``crop`` pastes it back after 二采.
 */
const LAYER = { enabled: true, source: "paint", mask: null, input: "", target: "", threshold: 0.5, objects: 4, box: [0.3, 0.15, 0.4, 0.5],
    detector: "face_yolov8m.pt", crop: false, megapixels: 0.5, start: 0, end: -1, strength: 1, prompt: "", grow: 8, feather: 8, seed: 0 };
export const newLayer = () => ({ ...clone(LAYER), id: uid() });

function layerOf(layer) {
    // A connected mask used to be stored as {input, file} in place of the painted one.
    const socket = layer.mask?.input ? { source: layer.source || "input", input: layer.mask.input, mask: layer.mask.file || null } : {};
    return { ...clone(LAYER), ...layer, ...socket };
}

/** A former 局部精修 region: a 放大贴回 layer on its fixed or face-following box. */
function detailLayer({ follow, denoise, ...item }) {
    return layerOf({ ...item, source: follow === "face" ? "face" : "box", crop: true, strength: denoise ?? 0.4 });
}

export function normalize(project) {
    const base = newProject();
    const out = { ...base, ...project, rewrite: { ...base.rewrite, ...(project?.rewrite || {}) }, swap: { ...base.swap, ...(project?.swap || {}) }, refs: refsOf(project?.refs) };
    out.shots = (project?.shots || []).map(({ details = [], ...shot }) => {
        const fresh = newShot(shot.kind === "source" ? "source" : "generate");
        return { ...fresh, ...shot, refs: refsOf(shot.refs), keyframes: (shot.keyframes || []).map(k => ({ fit: "auto", ...SHARED, ...k })),
            source: shot.kind === "source" ? { ...fresh.source, ...(shot.source || {}) } : shot.source || null,
            repaints: [...(shot.repaints || []).map(layerOf), ...details.map(detailLayer)] };
    });
    if (!out.shots.length) out.shots = [newShot()];
    return out;
}

/** Same windows as project.plan_windows: head (carried frames), gen (17k+5 window), keep (new frames), start. */
/** A source shot that starts where the previous shot left the same video, as 自动分段 cuts one stretch. */
export const follows = (previous, shot) => shot.kind === "source" && previous?.kind === "source"
    && JSON.stringify(shot.source.media) === JSON.stringify(previous.source.media) && shot.source.start === previous.source.start + previous.frames;

export function planWindows(project) {
    const shots = project.shots, overlap = project.overlap;
    const carries = shots.map((s, i) => i > 0 && overlap > 0 && s.continue && (s.kind === "generate" || follows(shots[i - 1], s)));
    const gens = [];
    shots.forEach((s, i) => {
        gens.push(align((carries[i] ? overlap : 0) + s.frames));
        // A continued source shot needs the previous window to end on its own last frame: no grid padding after it.
        const padded = carries[i] && s.kind === "source" && gens[i - 1] !== (carries[i - 1] ? overlap : 0) + shots[i - 1].frames;
        if (carries[i] && (gens[i - 1] < overlap || padded)) { carries[i] = false; gens[i] = align(s.frames); }
    });
    let start = 0;
    return shots.map((s, i) => {
        const head = carries[i] ? overlap : 0;
        const fixed = (i + 1 < shots.length && carries[i + 1]) || s.keyframes.some(k => k.at === -1);
        const keep = fixed ? gens[i] - head : s.frames;
        const w = { index: i, shot: s, head, gen: gens[i], keep, start, carry: carries[i] };
        start += keep;
        return w;
    });
}

export const totalFrames = project => planWindows(project).reduce((sum, w) => sum + w.keep, 0);

export const widget = (node, name) => node.widgets?.find(w => w.name === name);

// Canvas, overlap, refine, super resolution and the detail switch are node widgets; the editor reads them next to the shots
// but never stores them.
const WIDGET_KEYS = ["width", "height", "overlap", "refine", "superres", "detail"];

export function readProject(node) {
    let project;
    try { project = normalize(JSON.parse(widget(node, "project")?.value || "{}")); } catch { project = newProject(); }
    const value = name => widget(node, name)?.value;
    return { ...project, width: value("width"), height: value("height"), overlap: Number(value("overlap")),
        refine: { enabled: Boolean(value("two_pass") || value("rerender")), twoPass: value("two_pass") && value("two_pass_scale"),
            rerender: value("rerender") && value("rerender_scale") },
        superres: value("rtx_vsr") && value("rtx_vsr_scale"),
        detail: { enabled: value("detail") } };
}

// Python's round (halves to even) on the 32 px grid, as project.py lifts the canvas.
const round32 = v => { const q = v / 32, f = Math.floor(q); return Math.max(64, (q - f === 0.5 ? f + (f % 2) : Math.round(q)) * 32); };

/** [[name, width, height]] from the draft (width × height) through each enabled pass to the film. */
export function canvasChain(project) {
    const chain = [["首采", project.width, project.height]];
    for (const [name, scale] of [["两段式", project.refine.twoPass], ["低噪重绘", project.refine.rerender], ["RTX 超分", project.superres]]) {
        const [, w, h] = chain[chain.length - 1];
        if (scale) chain.push([name, round32(w * scale), round32(h * scale)]);
    }
    return chain;
}

/** [width, height] of the film. */
export const filmSize = project => canvasChain(project).pop().slice(1);

/** The project as stored in the node's project widget and in exported files. */
export const projectData = project => Object.fromEntries(Object.entries(project).filter(([key]) => !WIDGET_KEYS.includes(key)));

let pending = null;
/** Let the workflow tracker see edits made in the panel or modal (persistence and undo). */
export function changed() {
    clearTimeout(pending);
    pending = setTimeout(() => app.extensionManager?.workflow?.activeWorkflow?.changeTracker?.checkState?.(), 300);
}

export function writeProject(node, project) {
    const w = widget(node, "project");
    if (!w) return;
    w.value = JSON.stringify(projectData(project));
    w.callback?.(w.value);
    node.setDirtyCanvas?.(true, true);
    changed();
}

/** Connected Autogrow sockets of the given kinds the project can reference, e.g. {input: "video_0"}. */
export function graphInputs(node, ...kinds) {
    return kinds.flatMap(kind => {
        const pattern = new RegExp(`(^|\\.)${kind}_(\\d+)$`);
        return (node.inputs || []).filter(i => i.link != null && pattern.test(i.name)).map(i => {
            const input = `${kind}_${i.name.match(pattern)[2]}`;
            return { input, label: input };
        });
    });
}

/** Shots whose prompt the rewrite model writes (a kept-pixel source shot only uses its own repaint prompts). */
export const rewritable = (project, shot) => project.rewrite.enabled && shot.rewrite !== false
    && (project.swap.enabled ? shotMode(shot) === "rewrite" : !(shot.kind === "source" && !shot.source?.regen));

/** How a shot gets its picture. Every view shows the same three names and colors. */
export const MODES = { generate: "生成", keep: "保留原画", rewrite: "整体改写" };
export const modeNames = project => (project.swap.enabled ? { ...MODES, rewrite: "替换" } : MODES);
export const shotMode = shot => (shot.kind === "generate" ? "generate" : shot.source?.regen ? "rewrite" : "keep");

/** A shot switched to 生成 keeps its source video, so a wrong click on the switch costs nothing. */
export function applyMode(shot, mode) {
    if (mode === "generate") {
        if (shot.kind !== "generate") Object.assign(shot, { kind: "generate", audio: "generate", continue: true });
        return;
    }
    if (shot.kind !== "source") Object.assign(shot, { kind: "source", source: shot.source || newSource(), audio: "source", continue: false });
    shot.source.regen = mode === "rewrite";
    if (mode === "keep" && shot.audio === "generate") shot.audio = "source";
}

export function modeSwitch(project, shot, onpick) {
    const current = shotMode(shot);
    return h("div.seg.modes", {}, ...Object.entries(modeNames(project)).map(([mode, label]) =>
        h(`button.${mode}${mode === current ? ".on" : ""}`, { onclick: () => onpick(mode) }, label)));
}

/** A shot H3 samples: everything but a source kept as it is (its repaint layers still sample). */
export const samples = shot => shotMode(shot) !== "keep" || shot.repaints.some(layer => layer.enabled !== false);

/** 替换模式's prompt as project.swap_prompt writes it: "" until it says what to replace with what. */
export function swapPrompt(project) {
    const typed = project.swap.into.trim();
    const picture = typed.match(/<Picture \d+>/)?.[0] || (project.refs.images.length ? "<Picture 1>" : "");
    const into = typed.replace(/<Picture \d+>\s*(里的)?\s*/, "").trim() || (picture ? "person" : "");
    if (!project.swap.what.trim() || !into) return "";
    const what = project.swap.track ? "figure" : project.swap.what.trim();
    const source = picture ? ` in ${picture}` : "", look = picture ? `whose appearance comes from ${picture} and ` : "";
    const traits = picture ? `the face, hair and clothing of ${picture} are` : "this look is";
    return `subject_definitions:
<Subject 1> is the ${into}${source}, ${look}whose position, pose, movement and timing come from the ${what} in <Video 1>.
<Video 1> is the source video for the target video edit.

summary:
[video editing] The target video is an edited version of <Video 1>. <Subject 1> takes the place of the ${what}; the camera, the background, every other person and the timing stay as they are in <Video 1>.

retention_analysis:
<Subject 1> (appears in [Shot 1]): attribute_transfer - ${traits} transferred onto the ${what} of <Video 1>, who no longer looks like the original person.
<Video 1> (edit source): partially_preserved - the composition, camera movement, background, other people and timing are retained; only the ${what} changes.

detailed_description:
The target video keeps the visual style, lighting and color of <Video 1>.
[Shot 1] <Subject 1>, the ${into}, is exactly where the ${what} is in <Video 1>, at the same size in the frame, and performs the same movements with the same timing. The camera movement, the background and every other person match <Video 1> frame by frame.

overall_soundscape:
The ambience and physical sounds of <Video 1> continue.

non_diegetic_music:
N/A`;
}

/** What a 替换 prompt replaces with what, for the one line a shot shows of it; other text stays as it is. */
export function swapSummary(prompt) {
    const into = prompt.match(/<Subject 1> is the (.+?)(?: in <Picture \d+>)?, whose /)?.[1], what = prompt.match(/takes the place of the (.+?);/)?.[1];
    return !into ? prompt : what && what !== "figure" ? `${what} → ${into}` : into;
}

/** The prompt of a shot the rewrite model does not write: 替换模式's, its own, else the global one (project.load_project). */
export function ownPrompt(project, shot) {
    if (project.swap.enabled) return shotMode(shot) === "rewrite" ? swapPrompt(project) : shot.prompt.trim();
    return shot.prompt.trim() || (shot.inherit_prompt !== false ? project.story.trim() : "");
}

/** What a run will do, next to the run buttons: a film that generates nothing should never be a surprise. */
export function runPlan(project) {
    const sampled = project.shots.filter(samples), kept = project.shots.length - sampled.length;
    if (!sampled.length) return { warn: true, text: "全片保留原画，运行不会改变画面" };
    const rewritten = sampled.filter(shot => rewritable(project, shot)).length;
    const empty = sampled.filter(shot => !rewritable(project, shot) && !ownPrompt(project, shot)).length;
    const replaced = project.swap.enabled ? sampled.filter(shot => shotMode(shot) === "rewrite").length : 0;
    if (!swapPrompt(project) && sampled.some(shot => project.swap.enabled && shotMode(shot) === "rewrite" && !rewritable(project, shot))) {
        return { warn: true, text: "替换模式还没写把什么换成什么" };
    }
    const parts = [replaced && `将替换 ${replaced} 镜`, sampled.length > replaced && `将生成 ${sampled.length - replaced} 镜`, kept && `保留原画 ${kept} 镜`, rewritten && `自动改写 ${rewritten} 镜`, empty && `${empty} 镜没有提示词`];
    return { warn: empty > 0, text: parts.filter(Boolean).join("，") };
}

const AGENT_NODES = { YogurtH3Codex: "codex", YogurtH3Claude: "claude", YogurtH3Grok: "grok" };

/** Widget values of the rewriter node feeding the director's rewriter input, or null. */
export function rewriterConfig(node) {
    const slot = node.inputs?.find(i => i.name === "rewriter");
    if (slot?.link == null) return null;
    const links = app.graph.links;
    const link = links.get ? links.get(slot.link) : links[slot.link];
    const source = link ? app.graph.getNodeById(link.origin_id) : null;
    const value = name => source?.widgets?.find(w => w.name === name)?.value;
    const agent = AGENT_NODES[source?.type];
    if (agent) return { provider: agent, path: value("path"), model: value("model"), effort: value("effort"), vision: value("vision") };
    if (source?.type !== "YogurtH3RewriteModel") return null;
    return { provider: value("provider"), model: value("model"), base_url: value("base_url"), api_key: value("api_key"),
        temperature: value("temperature"), max_tokens: value("max_tokens"), vision: value("vision") };
}

export async function previewRewrite(node, project) {
    const rewriter = rewriterConfig(node);
    if (!rewriter) throw new Error("先把“Yogurt H3 改写模型”或改写 Codex / Claude / Grok 节点连接到导演台的 rewriter。");
    const response = await api.fetchApi("/yogurt_h3/director/rewrite", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ project: JSON.stringify(projectData(project)), settings: { width: project.width, height: project.height, overlap: project.overlap }, rewriter }),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || `改写失败：${response.status}`);
    return data;
}

/** Remember rewritten prompts on the node so they survive reloads and show in the editor. */
export function storePrompts(node, prompts, report) {
    node.properties = node.properties || {};
    node.properties.yh3Prompts = { ...(node.properties.yh3Prompts || {}), ...prompts };
    if (report) node.properties.yh3RewriteReport = report;
    changed();
}

/** A connected socket is named exactly as on the node (image_0, video_1, ...). */
export const mediaLabel = m => (!m ? "未选择" : m.input || m.filename);

/** "sub/name.png [output]" as LoadImage / LoadVideo widgets store it -> a media item. */
function annotated(value) {
    const [, path, type] = /^(.*?)(?: \[(input|output|temp)\])?$/.exec(String(value));
    const slash = path.lastIndexOf("/");
    return { filename: path.slice(slash + 1), subfolder: slash < 0 ? "" : path.slice(0, slash), type: type || "input" };
}

/**
 * What the editor can show for a connected socket: the file of a loader wired straight into it, or the thumbnail the
 * last run saved (for anything processed upstream). {url, video} / {audio, name}, or null when nothing is known yet.
 */
/** The node feeding one of the director's media sockets, e.g. "video_0"; legacy Reroute nodes are passed through. */
function inputOrigin(node, input) {
    const slot = node.inputs?.find(i => i.link != null && (i.name === input || i.name.endsWith(`.${input}`)));
    const links = app.graph.links, get = id => (links.get ? links.get(id) : links[id]);
    let link = slot && get(slot.link), origin = link ? app.graph.getNodeById(link.origin_id) : null;
    while (origin?.type === "Reroute" && origin.inputs?.[0]?.link != null) {
        link = get(origin.inputs[0].link);
        origin = link ? app.graph.getNodeById(link.origin_id) : null;
    }
    return origin;
}

/** The file behind a source video: an upload, or what a connected LoadVideo loads; null for any other socket. */
export function sourceFile(node, media) {
    if (!media?.input) return media;
    const origin = inputOrigin(node, media.input);
    const file = origin?.type === "LoadVideo" && origin.widgets?.find(w => w.name === "file")?.value;
    return file ? annotated(file) : null;
}

export function inputPreview(node, input) {
    const origin = inputOrigin(node, input);
    const value = name => origin?.widgets?.find(w => w.name === name)?.value;
    if (origin?.type === "LoadVideo" && value("file")) return { url: mediaUrl(annotated(value("file"))), video: true };
    if (origin?.type === "LoadAudio" && value("audio")) return { audio: true, name: annotated(value("audio")).filename };
    // A mask socket fed by LoadImage carries its alpha channel, not the picture, so masks wait for the run thumbnail.
    if (origin?.type === "LoadImage" && value("image") && input.startsWith("image_")) return { url: mediaUrl(annotated(value("image"))) };
    // Thumbnails used to live in temp, which ComfyUI empties at every start: those records point at nothing now.
    const saved = node.properties?.yh3Inputs?.[input];
    if (saved && saved.type !== "temp") return { url: mediaUrl(saved) };
    // Any other image node that already shows its picture on the canvas (other loaders, previews of an earlier run).
    const shown = input.startsWith("image_") && origin?.imgs?.[0]?.src;
    return shown ? { url: shown } : null;
}

/** What a media item looks like: an upload's own file, or what is known about a connected socket. */
export function mediaPreview(node, media) {
    if (!media) return null;
    if (media.input) return inputPreview(node, media.input);
    return { url: mediaUrl(media), video: isVideo(media), audio: isAudio(media), name: media.filename };
}

/** Queue the director alone (with its upstream) with ``extra`` merged into the project: a preview that loads no H3 model. */
export async function queueAlone(node, extra) {
    const { output, workflow } = await app.graphToPrompt();
    const id = String(node.id), inputs = output[id]?.inputs;
    if (!inputs) throw new Error("找不到导演台节点。");
    inputs.project = JSON.stringify({ ...JSON.parse(inputs.project), ...extra });
    const response = await api.fetchApi("/prompt", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ client_id: api.clientId, prompt: output, extra_data: { extra_pnginfo: { workflow } }, partial_execution_targets: [id] }),
    });
    if (!response.ok) {
        const body = await response.json().catch(() => ({}));
        throw new Error(body.error?.message || `HTTP ${response.status}`);
    }
}

/** 读取连线素材: pictures of every connected image / video / mask socket, without running H3. */
export async function readInputs(node, panel) {
    try {
        await queueAlone(node, { inputs_preview: true });
        panel?.say("正在读取连线素材…");
    } catch (error) {
        alert(`读取连线素材没能开始：${error.message}`);
    }
}

export function mediaUrl(m) {
    if (!m || m.input) return null;
    const params = new URLSearchParams({ filename: m.filename, subfolder: m.subfolder || "", type: m.type || "input" });
    return api.apiURL(`/view?${params}`);
}

export const isVideo = m => !!m?.filename && /\.(mp4|webm|mov|mkv|m4v|avi)$/i.test(m.filename);
export const isAudio = m => !!m?.filename && /\.(wav|mp3|flac|ogg|m4a|aac|opus)$/i.test(m.filename);

export async function upload(file, subfolder = "yogurt_h3") {
    const body = new FormData();
    body.append("image", file, file.name);
    body.append("subfolder", subfolder);
    body.append("type", "input");
    const response = await api.fetchApi("/upload/image", { method: "POST", body });
    if (response.status !== 200) throw new Error(`上传失败：${response.status} ${response.statusText}`);
    const data = await response.json();
    return { filename: data.name, subfolder: data.subfolder ?? subfolder, type: data.type || "input" };
}

export function pickFile(accept) {
    return new Promise(resolve => {
        const input = Object.assign(document.createElement("input"), { type: "file", accept });
        input.onchange = () => resolve(input.files?.[0] || null);
        input.click();
    });
}

/** Tiny element builder: h("div.cls", {onclick}, child, "text"). */
export function h(spec, attrs = {}, ...children) {
    const [tag, ...classes] = spec.split(".");
    const el = document.createElement(tag || "div");
    if (classes.length) el.className = classes.join(" ");
    for (const [key, value] of Object.entries(attrs || {})) {
        if (value == null || value === false) continue;
        if (key.startsWith("on")) el.addEventListener(key.slice(2), value);
        else if (key === "style" && typeof value === "object") Object.assign(el.style, value);
        else if (key in el) el[key] = value;
        else el.setAttribute(key, value === true ? "" : value);
    }
    for (const child of children.flat()) if (child != null && child !== false) el.append(child instanceof Node ? child : String(child));
    return el;
}

/** References a shot is generated with: the global ones first (unless switched off), then its own; same as project.py. */
export function effectiveRefs(project, shot) {
    const shared = shot.inherit_refs === false ? { images: [], videos: [], audios: [] } : project.refs;
    return {
        images: [...shared.images, ...shot.refs.images], videos: [...shared.videos, ...shot.refs.videos],
        audios: [...shared.audios, ...shot.refs.audios],
        shared: { images: shared.images.length, videos: shared.videos.length, audios: shared.audios.length },
    };
}

export const refCount = refs => refs.images.length + refs.videos.length + refs.audios.length;

export function features(shot, project) {
    const marks = [];
    if (shot.keyframes.length) marks.push(shot.keyframes.some(k => k.at === -1) ? "首尾" : "关键帧");
    const refs = refCount(effectiveRefs(project, shot));
    if (refs && samples(shot)) marks.push(`参考${refs}`);
    if (shot.repaints.length) marks.push(`重绘${shot.repaints.length}`);
    return marks;
}

export const blockLabel = (shot, project) => [shot.still ? "封面" : modeNames(project)[shotMode(shot)], ...features(shot, project)].join(" ");

const css = `
.yh3 { --yh3-bg:#16191d; --yh3-panel:#1f2328; --yh3-raise:#2a2f36; --yh3-line:#3a414b; --yh3-ink:#e8ecef; --yh3-dim:#9aa5b1;
  --yh3-gen:#3fb6a8; --yh3-src:#e0a14a; --yh3-keep:#6d7683; --yh3-accent:#7aa7ff; --yh3-warn:#ff8a7a; --yh3-ok:#7fd37f;
  color:var(--yh3-ink); font:12px/1.45 "Segoe UI","Microsoft YaHei UI",sans-serif; box-sizing:border-box; }
.yh3 *, .yh3 *::before, .yh3 *::after { box-sizing:border-box; }
.yh3 button { font:inherit; color:var(--yh3-ink); background:var(--yh3-raise); border:1px solid var(--yh3-line); border-radius:6px; padding:5px 10px; cursor:pointer; white-space:nowrap; }
.yh3 button:hover { border-color:var(--yh3-accent); }
.yh3 button.primary { background:#2f5fd0; border-color:#5b86f0; font-weight:600; }
.yh3 button.ghost { background:transparent; }
.yh3 button.danger:hover { border-color:var(--yh3-warn); color:var(--yh3-warn); }
.yh3 button.on { background:#29457a; border-color:var(--yh3-accent); }
.yh3 button:disabled { opacity:.45; cursor:default; }
.yh3 input, .yh3 textarea, .yh3 select { font:inherit; color:var(--yh3-ink); background:#12151a; border:1px solid var(--yh3-line); border-radius:6px; padding:5px 7px; }
.yh3 input[type=range] { padding:0; accent-color:var(--yh3-accent); }
.yh3 input[type=checkbox] { accent-color:var(--yh3-accent); }
.yh3 textarea { width:100%; resize:vertical; min-height:72px; }
.yh3 textarea:disabled { opacity:.45; }
.yh3 .seg { display:inline-flex; border:1px solid var(--yh3-line); border-radius:6px; overflow:hidden; }
.yh3 .seg button { border:0; border-radius:0; background:transparent; }
.yh3 .seg button.on { background:#29457a; }
.yh3 .seg.modes button { padding:3px 9px; color:var(--yh3-dim); }
.yh3 .seg.modes button.on { color:#10141a; font-weight:600; }
.yh3 .seg.modes button.generate.on { background:var(--yh3-gen); }
.yh3 .seg.modes button.keep.on { background:#a4adb9; }
.yh3 .seg.modes button.rewrite.on { background:var(--yh3-src); }
.yh3 .plan { color:var(--yh3-dim); }
.yh3 .plan.warn { color:var(--yh3-warn); }
.yh3 :focus-visible { outline:2px solid var(--yh3-accent); outline-offset:1px; }
.yh3 .dim { color:var(--yh3-dim); }
.yh3 .row { display:flex; gap:8px; align-items:center; flex-wrap:wrap; }
.yh3 label.row:has(> input[type=checkbox]) { flex-wrap:nowrap; }
.yh3 .grow { flex:1; min-width:0; }
.yh3 .strip { position:relative; display:flex; gap:3px; height:54px; overflow-x:auto; overflow-y:hidden; padding:2px 0; scrollbar-width:thin; }
.yh3 .block { position:relative; flex:none; min-width:44px; height:46px; border-radius:6px; border:1px solid transparent; padding:4px 6px; overflow:hidden; cursor:pointer;
  background:linear-gradient(180deg,#2b6f68,#205651); text-align:left; }
.yh3 .block.rewrite { background:linear-gradient(180deg,#8a6430,#6c4c22); }
.yh3 .block.keep { background:repeating-linear-gradient(135deg,#343a43 0 6px,#2c3139 6px 12px); color:var(--yh3-dim); }
.yh3 .block.selected { border-color:#fff; box-shadow:0 0 0 1px #fff inset; }
.yh3 .block .carry { position:absolute; right:0; top:0; bottom:0; background:repeating-linear-gradient(135deg,#ffffff38 0 3px,transparent 3px 7px); pointer-events:none; }
.yh3 .block b { display:block; font-size:11px; white-space:nowrap; position:relative; }
.yh3 .block small { display:block; font-size:10px; opacity:.85; white-space:nowrap; position:relative; }
.yh3 .block .state { position:absolute; right:4px; top:4px; width:7px; height:7px; border-radius:50%; background:#0000; }
.yh3 .block[data-state=complete] .state { background:var(--yh3-ok); }
.yh3 .block[data-state=running] .state { background:#ffd257; box-shadow:0 0 6px #ffd257; animation:yh3pulse 1s infinite; }
.yh3 .block .edge { position:absolute; right:0; top:0; bottom:0; width:8px; cursor:ew-resize; background:#ffffff14; }
.yh3 .block .edge:hover { background:#ffffff40; }
@keyframes yh3pulse { 50% { opacity:.35; } }
.yh3-panel { padding:8px; background:var(--yh3-bg); border-radius:8px; height:100%; display:flex; flex-direction:column; gap:7px; overflow:hidden; }
.yh3-panel header { display:flex; align-items:center; gap:8px; }
.yh3-panel header strong { font:600 14px "Bahnschrift","Segoe UI",sans-serif; letter-spacing:.03em; }
.yh3-panel .status { min-height:16px; color:var(--yh3-dim); white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.yh3-panel .strip { flex:none; }
.yh3-panel textarea { min-height:54px; flex:1; }
.yh3-panel .result { flex:1.3; min-height:54px; overflow:auto; white-space:pre-wrap; font:11px/1.5 Consolas,monospace; color:var(--yh3-dim);
  background:#12151a; border:1px solid var(--yh3-line); border-radius:6px; padding:5px 7px; }
.yh3-panel .result b { color:var(--yh3-ink); font-family:"Segoe UI",sans-serif; }
.yh3-panel label.field { flex:1; display:flex; flex-direction:column; gap:2px; }
.yh3-panel label.field.storyf { flex:.6; }
.yh3-panel label.field > span { color:var(--yh3-dim); font-size:11px; }
.yh3-panel label.field.storyf textarea { min-height:40px; }
.yh3-panel .runbox { flex:none; display:flex; gap:8px; padding:6px; border-radius:6px; background:var(--yh3-panel); border:1px solid var(--yh3-line); }
.yh3-panel .runbox[hidden], .yh3-panel .runbox img[hidden], .yh3-panel .refs[hidden] { display:none; }
.yh3-panel .runbox img { height:62px; aspect-ratio:16/9; object-fit:cover; border-radius:4px; background:#000; flex:none; }
.yh3-panel .runbox .info { flex:1; min-width:0; display:flex; flex-direction:column; justify-content:center; gap:4px; }
.yh3-panel .runbox .line { display:flex; gap:8px; align-items:baseline; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.yh3 .bar { position:relative; height:6px; border-radius:3px; background:#0f1216; overflow:hidden; }
.yh3 .bar i { position:absolute; left:0; top:0; bottom:0; background:linear-gradient(90deg,var(--yh3-gen),var(--yh3-accent)); transition:width .3s; }
.yh3 .runbox.done .bar i { background:var(--yh3-ok); }
.yh3 .runbox.error .bar i { background:var(--yh3-warn); }
.yh3 .runbox.error .line b { color:var(--yh3-warn); }
.yh3 .block .fill { position:absolute; left:0; bottom:0; height:3px; background:#ffd257; pointer-events:none; transition:width .3s; }
.yh3 .block[data-state=waiting] { opacity:.75; }
.yh3 .block[data-state=cached] { opacity:.55; }
.yh3-panel .refs { flex:none; display:flex; gap:5px; align-items:center; overflow-x:auto; scrollbar-width:thin; }
.yh3-panel .thumb { position:relative; flex:none; height:52px; min-width:52px; border-radius:5px; border:1px solid var(--yh3-line); background:#12151a center/cover no-repeat;
  overflow:hidden; cursor:pointer; display:flex; align-items:center; justify-content:center; padding:2px 6px; font-size:10px; color:var(--yh3-dim); text-align:center; }
.yh3-panel .thumb.picture { aspect-ratio:1; padding:2px; }
.yh3-panel .thumb video { position:absolute; inset:0; width:100%; height:100%; object-fit:cover; }
.yh3-panel .thumb .tag { position:absolute; left:2px; top:2px; z-index:1; background:#000b; border-radius:3px; padding:0 3px; font-size:9px; color:#fff; white-space:nowrap; }
.yh3-modal { position:fixed; inset:0; z-index:10000; background:#000a; display:flex; }
.yh3-modal .frame { margin:auto; width:min(1680px,97vw); height:min(1000px,95vh); background:var(--yh3-bg); border:1px solid var(--yh3-line); border-radius:12px;
  display:grid; grid-template-rows:auto 1fr auto; overflow:hidden; box-shadow:0 20px 70px #000c; }
.yh3-modal header { display:flex; align-items:center; gap:10px; padding:10px 14px; border-bottom:1px solid var(--yh3-line); background:var(--yh3-panel); }
.yh3-modal header strong { font:600 17px "Bahnschrift","Segoe UI",sans-serif; letter-spacing:.04em; }
.yh3-modal header .status { flex:1; min-width:0; color:var(--yh3-accent); display:-webkit-box; -webkit-box-orient:vertical; -webkit-line-clamp:2; overflow:hidden; }
.yh3-modal main { display:grid; grid-template-columns:minmax(0,1fr) 480px; min-height:0; }
.yh3-modal header .plan { white-space:nowrap; }
.yh3-modal .stage { display:flex; flex-direction:column; min-width:0; min-height:0; padding:12px; gap:8px; }
.yh3-modal .viewport { position:relative; flex:1; min-height:0; display:flex; align-items:center; justify-content:center; background:#0b0d10; border-radius:8px; overflow:hidden; }
.yh3-modal .screen { position:relative; max-width:100%; max-height:100%; }
.yh3-modal .screen video, .yh3-modal .screen img { position:absolute; inset:0; width:100%; height:100%; object-fit:fill; background:#000; }
.yh3-modal .screen canvas { position:absolute; inset:0; width:100%; height:100%; touch-action:none; }
.yh3-modal .placeholder { position:absolute; inset:0; display:flex; align-items:center; justify-content:center; text-align:center; color:var(--yh3-dim); padding:20px; background:#101318; }
.yh3-modal .transport { display:flex; gap:8px; align-items:center; }
.yh3-modal .trimbar { display:flex; flex-direction:column; gap:5px; padding:6px 8px; border-radius:8px; background:var(--yh3-panel); }
.yh3-modal .trimbar[hidden] { display:none; }
.yh3-modal .trimbar .lane { position:relative; height:24px; border-radius:5px; background:repeating-linear-gradient(90deg,#1b2027 0 1px,#0f1216 1px 40px); cursor:pointer; overflow:hidden; }
.yh3-modal .trimbar .window { position:absolute; top:0; bottom:0; min-width:4px; box-sizing:border-box; border:2px solid var(--yh3-src); border-radius:4px; background:#e0a14a40; cursor:grab; }
.yh3-modal .trimbar .window .out { position:absolute; right:-2px; top:-2px; bottom:-2px; width:7px; background:var(--yh3-src); cursor:ew-resize; }
.yh3-modal .trimbar .head { position:absolute; top:0; bottom:0; width:2px; margin-left:-1px; background:#ff5a5a; pointer-events:none; }
.yh3-modal .transport input[type=range] { flex:1; }
.yh3-modal .toolbar { display:flex; gap:6px; align-items:center; flex-wrap:wrap; min-height:32px; padding:4px 8px; border-radius:8px; background:var(--yh3-panel); }
.yh3-modal .toolbar.painting { background:#3a2b14; border:1px solid var(--yh3-src); }
.yh3-modal aside { border-left:1px solid var(--yh3-line); display:flex; flex-direction:column; min-height:0; background:var(--yh3-panel); }
.yh3-modal .tabs { display:flex; border-bottom:1px solid var(--yh3-line); }
.yh3-modal .tabs button { flex:1; border:0; border-radius:0; background:transparent; padding:9px 4px; color:var(--yh3-dim); border-bottom:2px solid transparent; }
.yh3-modal .tabs button.on { color:var(--yh3-ink); border-bottom-color:var(--yh3-accent); }
.yh3-modal .pane { flex:1; overflow:auto; padding:12px; display:flex; flex-direction:column; gap:10px; }
.yh3-modal .field { display:flex; flex-direction:column; gap:4px; }
.yh3-modal .field > span { color:var(--yh3-dim); font-size:11px; }
.yh3-modal .card { border:1px solid var(--yh3-line); border-radius:8px; padding:8px; display:flex; flex-direction:column; gap:7px; background:#191d22; }
.yh3-modal .card.active { border-color:var(--yh3-src); }
.yh3-modal .card header { padding:0; border:0; background:none; gap:6px; }
.yh3-modal .slots { display:grid; grid-template-columns:repeat(3,1fr); gap:6px; }
.yh3-modal .slot { position:relative; aspect-ratio:1; border:1px dashed var(--yh3-line); border-radius:6px; overflow:hidden; display:flex; align-items:center; justify-content:center;
  color:var(--yh3-dim); cursor:pointer; background:#12151a center/cover no-repeat; font-size:11px; text-align:center; padding:4px; }
.yh3-modal .slot.filled { border-style:solid; }
.yh3-modal .slot.drop { border-color:var(--yh3-accent); }
.yh3-modal .slot .tag { position:absolute; left:3px; top:3px; background:#000b; border-radius:4px; padding:0 4px; font-size:10px; color:#fff; }
.yh3-modal .slot .src { position:absolute; left:3px; bottom:3px; background:#000b; border-radius:4px; padding:0 4px; font-size:10px; color:#9fd0ff; }
.yh3-modal .slot .x { position:absolute; right:3px; top:3px; padding:0 5px; line-height:16px; border-radius:4px; background:#000b; border:0; }
.yh3-modal .slot video { width:100%; height:100%; object-fit:cover; }
.yh3-modal .chips { display:flex; flex-wrap:wrap; gap:4px; }
.yh3-modal .chips button { padding:2px 7px; border-radius:10px; font-size:11px; }
.yh3-modal .globals { display:flex; flex-direction:column; gap:8px; padding-bottom:12px; border-bottom:1px solid var(--yh3-line); }
.yh3-modal .slots.thumbs { grid-template-columns:repeat(5,1fr); }
.yh3-modal .pane textarea { field-sizing:content; min-height:52px; max-height:320px; }
.yh3-modal .shots { display:flex; flex-direction:column; gap:2px; margin:0 -12px; }
.yh3-modal .shot { display:flex; flex-direction:column; gap:6px; padding:8px 12px 10px 9px; border-left:3px solid var(--mode); }
.yh3-modal .shot.generate { --mode:var(--yh3-gen); }
.yh3-modal .shot.rewrite { --mode:var(--yh3-src); }
.yh3-modal .shot.keep { --mode:var(--yh3-keep); }
.yh3-modal .shot.selected { background:#2a2f3699; }
.yh3-modal .shot:not(.selected) .more { display:none; }
.yh3-modal .shot .more { display:flex; flex-direction:column; gap:6px; }
.yh3-modal .shot .slot, .yh3-modal .pane > .field > .field > .slot { aspect-ratio:16/5; }
.yh3-modal .shot summary { cursor:pointer; color:var(--yh3-dim); white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.yh3-modal footer { border-top:1px solid var(--yh3-line); background:var(--yh3-panel); padding:8px 14px 10px; display:flex; flex-direction:column; gap:6px; }
.yh3-modal .ruler { position:relative; height:16px; color:var(--yh3-dim); font-size:10px; }
.yh3-modal .ruler span { position:absolute; top:0; border-left:1px solid var(--yh3-line); padding-left:3px; height:16px; }
.yh3-modal .track { position:relative; height:62px; }
.yh3-modal .track .block { position:absolute; top:0; height:58px; }
.yh3-modal .track .marks { position:absolute; left:0; right:0; bottom:3px; height:6px; }
.yh3-modal .track .marks i { position:absolute; bottom:0; height:6px; border-radius:2px; }
.yh3-modal .track .playhead { position:absolute; top:-18px; bottom:0; width:2px; background:#ff5a5a; pointer-events:none; }
.yh3-modal .note { padding:8px; border-radius:6px; background:#22303f; color:#cfe0ff; }
.yh3-modal img.maskpreview { width:100%; border-radius:6px; background:#000; }
.yh3-modal .warn { padding:8px; border-radius:6px; background:#3d2522; color:#ffd0c8; }
.yh3-modal .report { white-space:pre-wrap; font:11px/1.5 Consolas,monospace; color:var(--yh3-dim); max-height:240px; overflow:auto; }
`;
if (!document.getElementById("yh3-style")) document.head.append(Object.assign(document.createElement("style"), { id: "yh3-style", textContent: css }));
