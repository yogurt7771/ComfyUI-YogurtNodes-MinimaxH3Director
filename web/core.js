// Shared project model for the Yogurt H3 director UI. Frame math mirrors project.py.
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

export const FPS = 24;
export const NODE = "YogurtH3Director";
export const STAGES = {
    encoding: "编码条件", sampling: "生成中", repainting: "重绘中", upscaling: "放大中", refining: "二采中", rerendering: "低噪重绘中",
    decoding: "解码中", detailing: "精修中", rewriting: "改写提示词", rewritten: "已改写", complete: "完成",
};

export const align = n => { n = Math.max(5, Math.round(n)); while (n % 17 !== 5) n++; return n; };
export const uid = () => Math.random().toString(36).slice(2, 9);
export const clamp = (v, lo, hi) => Math.min(Math.max(v, lo), hi);
export const secs = frames => `${(frames / FPS).toFixed(2)}s`;
export const clone = value => JSON.parse(JSON.stringify(value));

/** Every image or video the rewrite AI could see: shown to it by default, or kept private with optional words instead. */
export const SHARED = { share: true, description: "" };

export function newShot(kind = "generate") {
    return {
        id: uid(), kind, frames: 121, prompt: "", continue: kind === "generate", seed: 0, keyframes: [],
        refs: { images: [], videos: [], audios: [] },
        source: kind === "source" ? { media: null, start: 0, regen: false, strength: 1, ...SHARED } : null,
        audio: kind === "source" ? "source" : "generate", repaints: [], details: [], refine: true, rewrite: true,
        inherit_refs: true, inherit_prompt: true, ref_strength: 1,
    };
}

export function newProject() {
    return {
        version: 1, soundtrack: null, refs: { images: [], videos: [], audios: [] }, story: "", rewrite: { enabled: false, notes: "", seed: 0 },
        shots: [newShot()],
    };
}

// Reference images carry how they meet the canvas (older projects stored bare media), videos the frame they start at.
const refsOf = refs => ({ audios: [], ...refs,
    images: (refs?.images || []).map(m => (m.media ? { fit: "auto", ...SHARED, ...m } : { media: m, fit: "auto", ...SHARED })),
    videos: (refs?.videos || []).map(v => ({ audio: false, start: 0, ...SHARED, ...v })) });

export function normalize(project) {
    const base = newProject();
    const out = { ...base, ...project, rewrite: { ...base.rewrite, ...(project?.rewrite || {}) }, refs: refsOf(project?.refs) };
    out.shots = (project?.shots || []).map(shot => {
        const fresh = newShot(shot.kind === "source" ? "source" : "generate");
        return { ...fresh, ...shot, refs: refsOf(shot.refs), keyframes: (shot.keyframes || []).map(k => ({ fit: "auto", ...SHARED, ...k })),
            source: shot.kind === "source" ? { ...fresh.source, ...(shot.source || {}) } : null };
    });
    if (!out.shots.length) out.shots = [newShot()];
    return out;
}

/** Same windows as project.plan_windows: head (carried frames), gen (17k+5 window), keep (new frames), start. */
export function planWindows(project) {
    const shots = project.shots, overlap = project.overlap;
    const carries = shots.map((s, i) => i > 0 && overlap > 0 && s.continue && s.kind === "generate");
    const gens = [];
    shots.forEach((s, i) => {
        gens.push(align((carries[i] ? overlap : 0) + s.frames));
        if (carries[i] && gens[i - 1] < overlap) { carries[i] = false; gens[i] = align(s.frames); }
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

// Canvas, overlap, refine and the detail switch are node widgets; the editor reads them next to the shots but never stores them.
const WIDGET_KEYS = ["width", "height", "overlap", "refine", "detail"];

export function readProject(node) {
    let project;
    try { project = normalize(JSON.parse(widget(node, "project")?.value || "{}")); } catch { project = newProject(); }
    const value = name => widget(node, name)?.value;
    return { ...project, width: value("width"), height: value("height"), overlap: Number(value("overlap")),
        refine: { enabled: Boolean(value("two_pass") || value("rerender")), twoPass: value("two_pass") && value("two_pass_scale"),
            rerender: value("rerender") && value("rerender_scale") },
        detail: { enabled: value("detail") } };
}

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
export const rewritable = (project, shot) => project.rewrite.enabled && shot.rewrite !== false && !(shot.kind === "source" && !shot.source?.regen);

/** Widget values of the 改写模型 node feeding the director's rewriter input, or null. */
export function rewriterConfig(node) {
    const slot = node.inputs?.find(i => i.name === "rewriter");
    if (slot?.link == null) return null;
    const links = app.graph.links;
    const link = links.get ? links.get(slot.link) : links[slot.link];
    const source = link ? app.graph.getNodeById(link.origin_id) : null;
    if (source?.type !== "YogurtH3RewriteModel") return null;
    const value = name => source.widgets?.find(w => w.name === name)?.value;
    return { provider: value("provider"), model: value("model"), base_url: value("base_url"), api_key: value("api_key"),
        temperature: value("temperature"), max_tokens: value("max_tokens"), vision: value("vision") };
}

export async function previewRewrite(node, project) {
    const rewriter = rewriterConfig(node);
    if (!rewriter) throw new Error("先把“Yogurt H3 改写模型”节点连接到导演台的 rewriter。");
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
export function inputPreview(node, input) {
    const slot = node.inputs?.find(i => i.link != null && (i.name === input || i.name.endsWith(`.${input}`)));
    const links = app.graph.links;
    const link = slot && (links.get ? links.get(slot.link) : links[slot.link]);
    const origin = link ? app.graph.getNodeById(link.origin_id) : null;
    const value = name => origin?.widgets?.find(w => w.name === name)?.value;
    if (origin?.type === "LoadVideo" && value("file")) return { url: mediaUrl(annotated(value("file"))), video: true };
    if (origin?.type === "LoadAudio" && value("audio")) return { audio: true, name: annotated(value("audio")).filename };
    // A mask socket fed by LoadImage carries its alpha channel, not the picture, so masks wait for the run thumbnail.
    if (origin?.type === "LoadImage" && value("image") && input.startsWith("image_")) return { url: mediaUrl(annotated(value("image"))) };
    const saved = node.properties?.yh3Inputs?.[input];
    return saved ? { url: mediaUrl(saved) } : null;
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
    if (refs) marks.push(`参考${refs}`);
    if (shot.repaints.length) marks.push(`重绘${shot.repaints.length}`);
    if (shot.details.length) marks.push(`精修${shot.details.length}`);
    return marks;
}

const css = `
.yh3 { --yh3-bg:#16191d; --yh3-panel:#1f2328; --yh3-raise:#2a2f36; --yh3-line:#3a414b; --yh3-ink:#e8ecef; --yh3-dim:#9aa5b1;
  --yh3-gen:#3fb6a8; --yh3-src:#e0a14a; --yh3-accent:#7aa7ff; --yh3-warn:#ff8a7a; --yh3-ok:#7fd37f;
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
.yh3 :focus-visible { outline:2px solid var(--yh3-accent); outline-offset:1px; }
.yh3 .dim { color:var(--yh3-dim); }
.yh3 .row { display:flex; gap:8px; align-items:center; flex-wrap:wrap; }
.yh3 .grow { flex:1; min-width:0; }
.yh3 .strip { position:relative; display:flex; gap:3px; height:54px; overflow-x:auto; overflow-y:hidden; padding:2px 0; scrollbar-width:thin; }
.yh3 .block { position:relative; flex:none; min-width:44px; height:46px; border-radius:6px; border:1px solid transparent; padding:4px 6px; overflow:hidden; cursor:pointer;
  background:linear-gradient(180deg,#2b6f68,#205651); text-align:left; }
.yh3 .block.source { background:linear-gradient(180deg,#8a6430,#6c4c22); }
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
.yh3-modal { position:fixed; inset:0; z-index:10000; background:#000a; display:flex; }
.yh3-modal .frame { margin:auto; width:min(1680px,97vw); height:min(1000px,95vh); background:var(--yh3-bg); border:1px solid var(--yh3-line); border-radius:12px;
  display:grid; grid-template-rows:auto 1fr auto; overflow:hidden; box-shadow:0 20px 70px #000c; }
.yh3-modal header { display:flex; align-items:center; gap:10px; padding:10px 14px; border-bottom:1px solid var(--yh3-line); background:var(--yh3-panel); }
.yh3-modal header strong { font:600 17px "Bahnschrift","Segoe UI",sans-serif; letter-spacing:.04em; }
.yh3-modal main { display:grid; grid-template-columns:minmax(0,1fr) 400px; min-height:0; }
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
.yh3-modal .seg { display:inline-flex; border:1px solid var(--yh3-line); border-radius:6px; overflow:hidden; }
.yh3-modal .seg button { border:0; border-radius:0; background:transparent; }
.yh3-modal .seg button.on { background:#29457a; }
.yh3-modal footer { border-top:1px solid var(--yh3-line); background:var(--yh3-panel); padding:8px 14px 10px; display:flex; flex-direction:column; gap:6px; }
.yh3-modal .ruler { position:relative; height:16px; color:var(--yh3-dim); font-size:10px; }
.yh3-modal .ruler span { position:absolute; top:0; border-left:1px solid var(--yh3-line); padding-left:3px; height:16px; }
.yh3-modal .track { position:relative; height:62px; }
.yh3-modal .track .block { position:absolute; top:0; height:58px; }
.yh3-modal .track .marks { position:absolute; left:0; right:0; bottom:3px; height:6px; }
.yh3-modal .track .marks i { position:absolute; bottom:0; height:6px; border-radius:2px; }
.yh3-modal .track .playhead { position:absolute; top:-18px; bottom:0; width:2px; background:#ff5a5a; pointer-events:none; }
.yh3-modal .note { padding:8px; border-radius:6px; background:#22303f; color:#cfe0ff; }
.yh3-modal .warn { padding:8px; border-radius:6px; background:#3d2522; color:#ffd0c8; }
.yh3-modal .report { white-space:pre-wrap; font:11px/1.5 Consolas,monospace; color:var(--yh3-dim); max-height:240px; overflow:auto; }
`;
if (!document.getElementById("yh3-style")) document.head.append(Object.assign(document.createElement("style"), { id: "yh3-style", textContent: css }));
