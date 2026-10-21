// Full-screen director: stage (result / source / keyframe preview with paint overlay), inspector, timeline.
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import {
    FPS, OVERLAPS, STAGES, canvasFor, clamp, clone, effectiveRefs, features, graphInputs, h, isAudio, isVideo, mediaLabel, mediaUrl, refCount,
    newShot, pickFile, planWindows, previewRewrite, readProject, rewritable, rewriterConfig, secs, storePrompts, totalFrames, uid,
    upload, writeProject,
} from "./core.js";
import { Painter } from "./painter.js";

const ASPECTS = [["16:9", 16 / 9], ["9:16", 9 / 16], ["1:1", 1], ["4:3", 4 / 3], ["3:4", 3 / 4], ["21:9", 21 / 9]];
const AUDIO_LABELS = { generate: "模型生成", soundtrack: "配乐音轨（锁定）", source: "源视频原声（锁定）", silent: "静音" };
const TABS = [["story", "剧本"], ["shot", "镜头"], ["keys", "首尾帧"], ["refs", "参考"], ["repaint", "局部重绘"], ["detail", "局部精修"], ["project", "工程"]];

let detectorList = null;
async function detectors() {
    if (!detectorList) {
        try { detectorList = await (await api.fetchApi("/yogurt_h3/director/detectors")).json(); } catch { detectorList = []; }
    }
    return detectorList;
}

export function openEditor(node, panel, index) {
    const editor = new Editor(node, panel, index);
    editor.open();
    return editor;
}

class Editor {
    constructor(node, panel, index) {
        this.node = node;
        this.panel = panel;
        this.project = readProject(node);
        this.selected = clamp(index ?? 0, 0, this.project.shots.length - 1);
        this.tab = "shot";
        this.frame = 0;
        this.view = "auto";
        this.paint = -1;
        this.boxing = -1;
    }

    get shot() { return this.project.shots[this.selected]; }
    get windows() { return planWindows(this.project); }
    get win() { return this.windows[this.selected]; }
    get result() { return this.node.properties?.yh3Result || null; }

    open() {
        this.root = h("div.yh3.yh3-modal", { tabindex: "-1" });
        this.root.addEventListener("keydown", e => { this.key(e); e.stopPropagation(); });
        for (const type of ["keyup", "keypress", "paste", "copy", "cut"]) this.root.addEventListener(type, e => e.stopPropagation());
        for (const type of ["dragover", "drop"]) this.root.addEventListener(type, e => { e.preventDefault(); e.stopPropagation(); });
        this.video = h("video", { muted: true, playsInline: true, preload: "auto", crossOrigin: "anonymous" });
        this.image = h("img", { alt: "" });
        this.overlay = h("canvas");
        this.placeholder = h("div.placeholder");
        this.screen = h("div.screen", {}, this.video, this.image, this.overlay, this.placeholder);
        this.viewport = h("div.viewport", {}, this.screen);
        this.painter = new Painter(this.overlay);
        this.scrubber = h("input", { type: "range", min: 0, value: 0, oninput: e => this.seek(Number(e.target.value)) });
        this.clock = h("span.dim", { style: { minWidth: "150px", textAlign: "right" } });
        this.playButton = h("button", { onclick: () => this.togglePlay(), title: "空格" }, "▶");
        this.toolbar = h("div.toolbar");
        this.pane = h("div.pane");
        this.tabs = h("div.tabs");
        this.meta = h("span.dim");
        this.ruler = h("div.ruler");
        this.track = h("div.track");
        this.footerTools = h("div.row");
        const frame = h("div.frame", {},
            h("header", {},
                h("strong", {}, "H3 导演台"), this.meta, h("span.grow"),
                h("button.primary", { onclick: () => this.run() }, "▶ 运行全片"),
                h("button", { onclick: () => this.close(), title: "Esc" }, "关闭")),
            h("main", {},
                h("section.stage", {}, this.viewport,
                    h("div.transport", {}, this.playButton, this.scrubber, this.clock), this.toolbar),
                h("aside", {}, this.tabs, this.pane)),
            h("footer", {}, this.footerTools, this.ruler, this.track));
        this.root.append(frame);
        this.root.addEventListener("pointerdown", e => { if (e.target === this.root) this.close(); });
        document.body.append(this.root);
        this.root.focus();
        this.resizer = new ResizeObserver(() => { this.fitScreen(); this.renderTimeline(); });
        this.resizer.observe(this.viewport);
        this.resizer.observe(this.track);
        this.video.addEventListener("timeupdate", () => this.played());
        this.video.addEventListener("loadeddata", () => this.showFrame());
        this.unlisten = this.panel.listen(() => this.renderTimeline());
        this.refresh();
    }

    close() {
        this.video.pause();
        this.resizer.disconnect();
        this.unlisten();
        this.root.remove();
        this.panel.select(this.selected);
    }

    key(e) {
        if (["INPUT", "TEXTAREA", "SELECT"].includes(e.target.tagName)) return;
        if (e.key === "Escape") { this.paint >= 0 || this.boxing >= 0 ? this.stopTools() : this.close(); }
        else if (e.key === " ") { e.preventDefault(); this.togglePlay(); }
        else if (e.key === "ArrowLeft") this.seek(this.frame - (e.shiftKey ? FPS : 1));
        else if (e.key === "ArrowRight") this.seek(this.frame + (e.shiftKey ? FPS : 1));
        else if (e.key === "[" && this.paint >= 0) this.setBrush(this.painter.size * 0.8);
        else if (e.key === "]" && this.paint >= 0) this.setBrush(this.painter.size * 1.25);
    }

    commit(structural = false) {
        writeProject(this.node, this.project);
        this.panel.refresh();
        this.renderMeta();
        this.renderTimeline();
        if (structural) { this.renderInspector(); this.renderStage(); }
    }

    refresh() {
        this.renderMeta();
        this.renderTabs();
        this.renderInspector();
        this.renderStage();
        this.renderTimeline();
        this.renderFooterTools();
    }

    select(index, frame = 0) {
        this.stopTools();
        this.selected = clamp(index, 0, this.project.shots.length - 1);
        this.frame = frame;
        this.view = "auto";
        this.refresh();
    }

    run() {
        writeProject(this.node, this.project);
        app.queuePrompt(0, 1);
        this.panel.say("已加入队列。");
    }

    // ------------------------------------------------------------------ stage

    stageSource() {
        const shot = this.shot, entry = this.result?.timeline?.find(e => e.id === shot.id);
        const sourceUrl = shot.kind === "source" ? mediaUrl(shot.source.media) : null;
        if (entry && this.view !== "source") return { kind: "video", url: mediaUrl(this.result.video), base: entry.start, label: "上次成片", stale: entry.frames !== this.win.keep };
        if (sourceUrl) return { kind: "video", url: sourceUrl, base: shot.source.start, label: "源视频" };
        const first = [...shot.keyframes].sort((a, b) => (a.at === -1) - (b.at === -1) || a.at - b.at)[0];
        if (first?.media && !first.media.input) return { kind: "image", url: mediaUrl(first.media), label: "首帧图" };
        return { kind: "none" };
    }

    fitScreen() {
        const { width, height } = this.project;
        const box = this.viewport.getBoundingClientRect();
        const scale = Math.min(box.width / width, box.height / height);
        Object.assign(this.screen.style, { width: `${Math.floor(width * scale)}px`, height: `${Math.floor(height * scale)}px` });
        this.painter.resize(width, height);
    }

    renderStage() {
        this.fitScreen();
        const source = this.stage = this.stageSource();
        this.video.style.display = source.kind === "video" ? "" : "none";
        this.image.style.display = source.kind === "image" ? "" : "none";
        this.placeholder.style.display = source.kind === "none" ? "" : "none";
        this.placeholder.textContent = this.shot.kind === "source" ? "选择源视频后在这里预览。" : "运行后在这里预览本镜成片；在“首尾帧”里放一张首帧图可先看构图。";
        if (source.kind === "video" && this.video.dataset.url !== source.url) {
            this.video.dataset.url = source.url;
            this.video.src = source.url;
        } else if (source.kind === "image" && this.image.src !== source.url) {
            this.image.src = source.url;
        }
        this.scrubber.max = Math.max(0, this.win.keep - 1);
        this.seek(this.frame);
        this.renderToolbar();
        this.painter.boxes = this.tab === "detail" ? this.shot.details.map(d => d.box) : [];
        this.painter.active = this.boxing;
        this.painter.render();
    }

    seek(frame) {
        this.frame = clamp(Math.round(frame), 0, Math.max(0, this.win.keep - 1));
        this.scrubber.value = this.frame;
        this.showFrame();
        this.renderClock();
        this.renderPlayhead();
    }

    showFrame() {
        if (this.stage?.kind !== "video" || !this.video.paused || !this.video.readyState) return;
        const time = (this.stage.base + this.frame + 0.5) / FPS;
        if (Math.abs(this.video.currentTime - time) > 0.5 / FPS) this.video.currentTime = time;
    }

    togglePlay() {
        if (this.stage?.kind !== "video") return;
        if (this.video.paused) {
            if (this.frame >= this.win.keep - 1) this.seek(0);
            this.video.currentTime = (this.stage.base + this.frame + 0.5) / FPS;
            this.video.muted = false;
            this.video.play();
            this.playButton.textContent = "❚❚";
        } else {
            this.video.pause();
            this.playButton.textContent = "▶";
        }
    }

    played() {
        if (this.video.paused || this.stage?.kind !== "video") return;
        const frame = Math.floor(this.video.currentTime * FPS) - this.stage.base;
        if (frame >= this.win.keep) {
            this.video.pause();
            this.playButton.textContent = "▶";
            this.seek(this.win.keep - 1);
            return;
        }
        this.frame = Math.max(0, frame);
        this.scrubber.value = this.frame;
        this.renderClock();
        this.renderPlayhead();
    }

    renderClock() {
        const w = this.win;
        this.clock.textContent = `第 ${this.frame + 1}/${w.keep} 帧 · ${secs(this.frame)} · 全片 ${secs(w.start + this.frame)}`;
    }

    async captureFrame() {
        const canvas = h("canvas", { width: this.project.width, height: this.project.height });
        const source = this.stage?.kind === "video" ? this.video : this.stage?.kind === "image" ? this.image : null;
        if (!source) throw new Error("当前没有可截取的画面。");
        canvas.getContext("2d").drawImage(source, 0, 0, canvas.width, canvas.height);
        const blob = await new Promise(resolve => canvas.toBlob(resolve, "image/png"));
        return upload(new File([blob], `frame_${this.shot.id}_${uid()}.png`, { type: "image/png" }), "yogurt_h3/frames");
    }

    renderToolbar() {
        const bar = this.toolbar;
        bar.replaceChildren();
        bar.classList.toggle("painting", this.paint >= 0 || this.boxing >= 0);
        if (this.paint >= 0) {
            const tool = (name, label) => h(`button${this.painter.tool === name ? ".on" : ""}`, { onclick: () => { this.painter.tool = name; this.renderToolbar(); } }, label);
            const size = h("input", { type: "range", min: 4, max: 400, value: this.painter.size, oninput: e => this.setBrush(Number(e.target.value)) });
            bar.append(h("b", {}, `绘制重绘 ${this.paint + 1} 的遮罩`), tool("brush", "画笔"), tool("eraser", "橡皮"), tool("rect", "矩形"),
                h("span.dim", {}, "笔刷"), size,
                h("button", { onclick: () => this.painter.invertMask() }, "反选"), h("button", { onclick: () => this.painter.clearMask() }, "清空"),
                h("span.grow"), h("span.dim", {}, "[ ] 调笔刷 · 白色区域重新生成"),
                h("button.primary", { onclick: () => this.saveMask() }, "完成"), h("button", { onclick: () => this.stopTools() }, "取消"));
            return;
        }
        if (this.boxing >= 0) {
            bar.append(h("b", {}, `在画面上拖出精修 ${this.boxing + 1} 的区域`), h("span.grow"), h("button", { onclick: () => this.stopTools() }, "取消"));
            return;
        }
        const source = this.stage || {};
        bar.append(h("span.dim", {}, source.label ? `显示：${source.label}` : "无画面"));
        if (source.stale) bar.append(h("span.warn", { style: { padding: "2px 8px" } }, "时长已改，成片与当前设置不一致，重新运行后更新"));
        if (this.shot.kind === "source" && this.result?.timeline?.some(e => e.id === this.shot.id)) {
            bar.append(h("div.seg", {},
                h(`button${this.view !== "source" ? ".on" : ""}`, { onclick: () => { this.view = "auto"; this.renderStage(); } }, "成片"),
                h(`button${this.view === "source" ? ".on" : ""}`, { onclick: () => { this.view = "source"; this.renderStage(); } }, "源视频")));
        }
        bar.append(h("span.grow"), h("span.dim", {}, "空格播放 · ←/→ 逐帧 · Shift 加速"));
    }

    setBrush(size) {
        this.painter.size = clamp(Math.round(size), 4, 400);
        this.painter.render();
        this.renderToolbar();
    }

    async startPaint(index) {
        this.stopTools();
        this.tab = "repaint";
        this.paint = index;
        const layer = this.shot.repaints[index];
        this.fitScreen();
        await this.painter.loadMask(layer.mask && !layer.mask.input ? mediaUrl(layer.mask) : null);
        this.painter.setMode("mask");
        this.renderTabs();
        this.renderInspector();
        this.renderToolbar();
    }

    async saveMask() {
        const layer = this.shot.repaints[this.paint];
        if (this.painter.dirty) {
            if (!this.painter.hasMask()) {
                layer.mask = null;
            } else {
                const blob = await this.painter.exportMask();
                layer.mask = await upload(new File([blob], `mask_${this.shot.id}_${uid()}.png`, { type: "image/png" }), "yogurt_h3/masks");
            }
        }
        this.stopTools();
        this.commit(true);
    }

    startBox(index) {
        this.stopTools();
        this.tab = "detail";
        this.boxing = index;
        this.painter.setMode("box");
        this.painter.onbox = box => {
            this.shot.details[index].box = box.map(v => Number(v.toFixed(4)));
            this.stopTools();
            this.commit(true);
        };
        this.renderTabs();
        this.renderStage();
    }

    stopTools() {
        if (this.paint < 0 && this.boxing < 0) return;
        this.paint = this.boxing = -1;
        this.painter.onbox = null;
        this.painter.clearMask();
        this.painter.dirty = false;
        this.painter.setMode("view");
        this.renderInspector();
        this.renderStage();
    }

    // ------------------------------------------------------------------ timeline

    renderMeta() {
        const p = this.project, total = totalFrames(p);
        const out = p.refine.enabled ? ` → 二采 ${Math.round(p.width * p.refine.scale / 32) * 32}×${Math.round(p.height * p.refine.scale / 32) * 32}` : "";
        this.meta.textContent = `${p.width}×${p.height}${out} · ${p.shots.length} 镜 · ${secs(total)} · 24fps`;
    }

    renderFooterTools() {
        const move = delta => {
            const i = this.selected, j = i + delta;
            if (j < 0 || j >= this.project.shots.length) return;
            const shots = this.project.shots;
            [shots[i], shots[j]] = [shots[j], shots[i]];
            this.selected = j;
            this.commit(true);
        };
        this.footerTools.replaceChildren(
            h("button", { onclick: () => this.addShot("generate") }, "+ 生成镜头"),
            h("button", { onclick: () => this.addShot("source") }, "+ 源视频镜头"),
            h("button", { onclick: () => this.duplicate() }, "复制本镜"),
            h("button", { onclick: () => move(-1), title: "前移" }, "←"),
            h("button", { onclick: () => move(1), title: "后移" }, "→"),
            h("button.danger", { onclick: () => this.removeShot() }, "删除本镜"),
            h("span.grow"),
            h("span.dim", {}, "拖动镜头右边缘调整时长 · 斜纹是下一镜续接的尾部帧"));
    }

    addShot(kind) {
        const shot = newShot(kind);
        if (kind === "generate") shot.prompt = this.shot?.prompt || "";
        this.project.shots.splice(this.selected + 1, 0, shot);
        this.selected += 1;
        this.frame = 0;
        this.tab = "shot";
        this.commit(true);
        this.renderTabs();
    }

    duplicate() {
        const shot = clone(this.shot);
        shot.id = uid();
        shot.repaints.forEach(layer => { layer.id = uid(); });
        shot.details.forEach(item => { item.id = uid(); });
        this.project.shots.splice(this.selected + 1, 0, shot);
        this.selected += 1;
        this.commit(true);
    }

    removeShot() {
        if (this.project.shots.length === 1) return;
        if (!confirm(`删除第 ${this.selected + 1} 镜？`)) return;
        this.project.shots.splice(this.selected, 1);
        this.selected = clamp(this.selected, 0, this.project.shots.length - 1);
        this.frame = 0;
        this.commit(true);
    }

    renderTimeline() {
        const windows = this.windows, total = Math.max(1, windows.reduce((s, w) => s + w.keep, 0));
        const width = this.track.clientWidth || 1;
        const scale = width / total;
        this.scale = scale;
        const step = [1, 2, 5, 10, 15, 30, 60, 120].find(s => s * FPS * scale >= 56) || 300;
        this.ruler.replaceChildren(...Array.from({ length: Math.floor(total / FPS / step) + 1 }, (_, i) =>
            h("span", { style: { left: `${i * step * FPS * scale}px` } }, `${i * step}s`)));
        const states = this.panel.states;
        const carried = w => windows[w.index + 1]?.head || 0;
        const blocks = windows.map(w => {
            const shot = w.shot, state = states[shot.id];
            const block = h(`button.block${shot.kind === "source" ? ".source" : ""}${w.index === this.selected ? ".selected" : ""}`, {
                style: { left: `${w.start * scale}px`, width: `${Math.max(4, w.keep * scale - 2)}px` },
                "data-state": state === "complete" ? "complete" : state ? "running" : "",
                title: `${shot.prompt || "（无提示词）"}\n${secs(w.keep)}${state ? ` · ${STAGES[state] || state}` : ""}`,
                onclick: e => {
                    const local = Math.floor((e.clientX - block.getBoundingClientRect().left) / scale);
                    w.index === this.selected ? this.seek(local) : this.select(w.index, clamp(local, 0, w.keep - 1));
                },
            },
                carried(w) ? h("i.carry", { style: { width: `${Math.min(carried(w), w.keep) * scale}px` }, title: "下一镜从这里续接" }) : null,
                h("i.state"),
                h("b", {}, `${w.index + 1} · ${secs(w.keep)}${state && state !== "complete" ? ` · ${STAGES[state] || ""}` : ""}`),
                h("small", {}, features(shot, this.project).join(" ") || (shot.kind === "source" ? "源视频" : shot.prompt.slice(0, 40))),
                this.marks(w, scale));
            const edge = h("span.edge", { title: "拖动调整时长" });
            edge.addEventListener("pointerdown", e => this.resizeShot(e, w));
            block.append(edge);
            return block;
        });
        this.playheadEl = h("div.playhead");
        this.track.replaceChildren(...blocks, this.playheadEl);
        this.renderPlayhead();
    }

    marks(w, scale) {
        const marks = h("div.marks");
        const shot = w.shot;
        const span = (start, end, color) => {
            const a = clamp(start, 0, w.keep - 1), b = end < 0 ? w.keep : clamp(end + 1, a + 1, w.keep);
            marks.append(h("i", { style: { left: `${a * scale}px`, width: `${Math.max(3, (b - a) * scale)}px`, background: color } }));
        };
        shot.repaints.forEach(layer => span(layer.start, layer.end, "#ff8a2a"));
        shot.details.forEach(item => span(item.start, item.end, "#7aa7ff"));
        shot.keyframes.forEach(k => {
            const at = k.at === -1 ? w.keep - 1 : clamp(k.at, 0, w.keep - 1);
            marks.append(h("i", { style: { left: `${at * scale - 2}px`, width: "5px", height: "9px", background: "#fff", bottom: "-1px" } }));
        });
        return marks;
    }

    renderPlayhead() {
        if (!this.playheadEl || !this.scale) return;
        this.playheadEl.style.left = `${(this.win.start + this.frame) * this.scale}px`;
    }

    resizeShot(event, w) {
        event.preventDefault();
        event.stopPropagation();
        const shot = this.project.shots[w.index];
        const startX = event.clientX, startFrames = shot.frames;
        const move = e => {
            shot.frames = clamp(Math.round(startFrames + (e.clientX - startX) / this.scale), 5, FPS * 600);
            this.renderTimeline();
            this.renderMeta();
        };
        const up = () => {
            window.removeEventListener("pointermove", move);
            window.removeEventListener("pointerup", up);
            this.commit();
            if (w.index === this.selected) this.renderInspector();
            this.renderStage();
        };
        window.addEventListener("pointermove", move);
        window.addEventListener("pointerup", up);
    }

    // ------------------------------------------------------------------ inspector

    renderTabs() {
        const counts = { keys: this.shot.keyframes.length, refs: refCount(effectiveRefs(this.project, this.shot)),
            repaint: this.shot.repaints.length, detail: this.shot.details.length };
        this.tabs.replaceChildren(...TABS.map(([id, label]) => h(`button${this.tab === id ? ".on" : ""}`, {
            onclick: () => { this.stopTools(); this.tab = id; this.renderTabs(); this.renderInspector(); this.renderStage(); },
        }, counts[id] ? `${label} ${counts[id]}` : label)));
    }

    /** Rewritten prompts arrived: redraw the panes that show them, unless the user is editing in the side pane. */
    showPrompts() {
        if ((this.tab === "story" || this.tab === "shot") && !this.pane.contains(document.activeElement)) this.renderInspector();
    }

    renderInspector() {
        const builders ={ story: this.storyPane, shot: this.shotPane, keys: this.keysPane, refs: this.refsPane, repaint: this.repaintPane, detail: this.detailPane, project: this.projectPane };
        this.pane.replaceChildren(...[builders[this.tab].call(this)].flat(3).filter(child => child != null && child !== false));
        this.renderTabs();
    }

    field(label, ...control) {
        return h("label.field", {}, h("span", {}, label), ...control);
    }

    /** min/max/multiple are in stored units; the input shows obj[key] / scale (e.g. frames shown as seconds). */
    number(obj, key, { min = -Infinity, max = Infinity, step = 1, scale = 1, multiple = 0, after } = {}) {
        const shown = v => Number((v / scale).toFixed(4));
        return h("input", {
            type: "number", step, min: Number.isFinite(min) ? min / scale : null, max: Number.isFinite(max) ? max / scale : null, value: shown(obj[key]),
            onchange: e => {
                let value = Number(e.target.value) * scale;
                if (!Number.isFinite(value)) return;
                value = multiple ? Math.round(value / multiple) * multiple : scale !== 1 || Number.isInteger(step) ? Math.round(value) : Math.round(value * 1e4) / 1e4;
                obj[key] = clamp(value, min, max);
                e.target.value = shown(obj[key]);
                this.commit();
                after?.();
            },
        });
    }

    slider(obj, key, min, max, step, format = v => v.toFixed(2)) {
        const label = h("span.dim", { style: { minWidth: "40px" } }, format(obj[key]));
        return h("div.row", {}, h("input.grow", {
            type: "range", min, max, step, value: obj[key],
            oninput: e => { obj[key] = Number(e.target.value); label.textContent = format(obj[key]); },
            onchange: () => this.commit(),
        }), label);
    }

    check(obj, key, label, after) {
        return h("label.row", {}, h("input", { type: "checkbox", checked: !!obj[key], onchange: e => { obj[key] = e.target.checked; this.commit(); after?.(); } }), label);
    }

    seg(options, value, onpick) {
        return h("div.seg", {}, ...options.map(([v, label]) => h(`button${v === value ? ".on" : ""}`, { onclick: () => onpick(v) }, label)));
    }

    range(item) {
        const keep = this.win.keep;
        const pos = key => this.number(item, key, { min: key === "end" ? -1 : 0, max: keep - 1 });
        const here = key => h("button.ghost", { onclick: () => { item[key] = this.frame; this.commit(true); }, title: "设为当前帧" }, "⌖");
        return this.field("帧范围（本镜第几帧，止 -1 = 到结尾）",
            h("div.row", {}, "起", pos("start"), here("start"), "止", pos("end"), here("end"),
                h("button.ghost", { onclick: () => { item.start = 0; item.end = -1; this.commit(true); } }, "整镜")));
    }

    reseed(item) {
        return h("button", { onclick: () => { item.seed = (item.seed || 0) + 1; this.commit(); this.renderInspector(); }, title: "换一个随机结果" }, `🎲 换种子（${item.seed || 0}）`);
    }

    /** Media slot: click to upload, drop a file, pick a graph input, or capture the stage frame. */
    slot(media, { accept = "image/*", tag, graph = "image", capture = false, onpick }) {
        const url = mediaUrl(media);
        const el = h(`div.slot${media ? ".filled" : ""}`, { title: media ? mediaLabel(media) : "点击上传，或把文件拖到这里" });
        if (media && url && isVideo(media)) el.append(h("video", { src: url, muted: true, preload: "metadata" }));
        else if (media && url && isAudio(media)) el.append(h("span", {}, `♪ ${media.filename}`));
        else if (url) el.style.backgroundImage = `url("${url}")`;
        else el.append(h("span", {}, media ? mediaLabel(media) : "+ 上传"));
        if (tag) el.append(h("span.tag", {}, tag));
        const choose = async file => {
            if (!file) return;
            try { onpick(await upload(file)); } catch (error) { alert(error.message); }
        };
        el.addEventListener("click", async e => { if (e.target === el || e.target.parentElement === el) choose(await pickFile(accept)); });
        el.addEventListener("dragover", e => { e.preventDefault(); el.classList.add("drop"); });
        el.addEventListener("dragleave", () => el.classList.remove("drop"));
        el.addEventListener("drop", e => { e.preventDefault(); el.classList.remove("drop"); choose(e.dataTransfer.files?.[0]); });
        if (media) el.append(h("button.x", { onclick: e => { e.stopPropagation(); onpick(null); }, title: "移除" }, "✕"));
        const extras = [];
        const inputs = graph ? graphInputs(this.node, graph) : [];
        if (inputs.length) {
            extras.push(h("select", {
                onchange: e => { if (e.target.value) onpick({ input: e.target.value }); },
            }, h("option", { value: "" }, "图输入…"), ...inputs.map(i => h("option", { value: i.input, selected: media?.input === i.input }, i.label))));
        }
        if (capture) extras.push(h("button", { onclick: async () => { try { onpick(await this.captureFrame()); } catch (error) { alert(error.message); } } }, "截取当前画面"));
        return h("div.field", {}, el, extras.length ? h("div.row", {}, ...extras) : null);
    }

    insertTag(tag) {
        const area = this.promptArea;
        if (!area) return;
        const at = area.selectionStart ?? area.value.length;
        area.value = area.value.slice(0, at) + tag + area.value.slice(area.selectionEnd ?? at);
        area.focus();
        area.selectionStart = area.selectionEnd = at + tag.length;
        this.shot.prompt = area.value;
        this.commit();
    }

    tags() {
        const shot = this.shot, refs = effectiveRefs(this.project, shot), out = [];
        const scope = (kind, i, label) => i < refs.shared[kind] ? `全局${label} ${i + 1}` : `${label} ${i + 1 - refs.shared[kind]}`;
        refs.images.forEach((_, i) => out.push([`<Picture ${i + 1}>`, scope("images", i, "参考图")]));
        const source = shot.kind === "source" && shot.source.regen ? 1 : 0;
        const videos = (source ? [{ audio: false, source: true }] : []).concat(refs.videos);
        let audio = 0;
        videos.forEach((v, i) => {
            if (v.audio) out.push([`<Audio ${++audio}>`, `<Video ${i + 1}> 原声`]);
            out.push([`<Video ${i + 1}>`, v.source ? "源视频" : scope("videos", i - source, "参考视频")]);
        });
        refs.audios.forEach((_, i) => out.push([`<Audio ${++audio}>`, scope("audios", i, "参考音频")]));
        if (!out.length && shot.keyframes.length) {
            const ends = shot.keyframes.filter(k => k.at === -1 || (k.at === 0 && !this.win.head));
            const first = ends.find(k => k.at === 0), last = ends.find(k => k.at === -1);
            if (first) out.push(["<Picture 1>", "首帧"]);
            if (last) out.push([`<Picture ${first ? 2 : 1}>`, "尾帧"]);
        }
        return out;
    }

    shotPane() {
        const shot = this.shot, w = this.win, first = this.selected === 0;
        const parts = [];
        parts.push(h("div.row", {}, h("b", {}, `第 ${this.selected + 1} 镜`),
            this.seg([["generate", "生成"], ["source", "源视频"]], shot.kind, kind => this.setKind(kind))));
        if (shot.kind === "source") {
            parts.push(this.field("源视频（24fps 时间轴，可为任意帧率）",
                this.slot(shot.source.media, { accept: "video/*", graph: "image", onpick: m => { shot.source.media = m; this.commit(true); } })));
            parts.push(this.field("起点（秒）", this.number(shot.source, "start", { min: 0, step: 1 / FPS, scale: FPS })));
            parts.push(this.field("改写方式", this.seg([[false, "保留原画 · 只改遮罩"], [true, "整体改写（参考 <Video 1>）"]], shot.source.regen,
                v => { shot.source.regen = v; this.commit(true); })));
            if (shot.source.regen) {
                parts.push(this.field("改写强度（1 = 只参考源视频，越低越贴近原画）", this.slider(shot.source, "strength", 0.05, 1, 0.05)));
            } else {
                parts.push(h("div.note", {}, "保留原画：遮罩外像素原样输出。在“局部重绘”画遮罩、在“局部精修”框区域。"));
            }
        }
        parts.push(this.field("时长（秒）", h("div.row", {}, this.number(shot, "frames", { min: 5, max: FPS * 600, step: 0.125, scale: FPS, after: () => this.commit(true) }),
            h("span.dim", {}, `生成 ${w.gen} 帧${w.head ? `（含续接 ${w.head} 帧）` : ""} · 输出 ${w.keep} 帧`))));
        if (shot.kind === "generate" && !first) {
            parts.push(h("div.field", {}, this.check(shot, "continue", "接着上一镜继续拍（latent 续接，画面与声音连贯）", () => this.commit(true)),
                !w.carry && shot.continue ? h("span.warn", {}, "上一镜过短或工程关闭了续接，本镜将硬切开始。") : null));
        }
        const keepsPixels = shot.kind === "source" && !shot.source.regen;
        if (this.project.rewrite.enabled && !keepsPixels) {
            parts.push(this.check(shot, "rewrite", "本镜自动改写（取消则直接使用下面的提示词）", () => this.commit(true)));
        }
        const auto = rewritable(this.project, shot);
        const inherits = shot.inherit_prompt !== false;
        if (auto) parts.push(this.check(shot, "inherit_prompt", "继承全局提示词（关掉则这一镜只按下面的片段提示词写）", () => this.commit(true)));
        this.promptArea = h("textarea", {
            rows: auto ? 4 : 9, value: shot.prompt,
            placeholder: !auto ? "描写这一镜：人物、动作、镜头、环境、台词与声音……"
                : inherits ? "可留空：改写模型按全局提示词写这一镜。也可以写这一镜的要求，例如：她停下脚步，回头对镜头微笑。"
                    : "这一镜要拍什么：人物、动作、场景、声音……",
            oninput: e => { shot.prompt = e.target.value; this.commit(); },
        });
        const label = !auto ? "提示词" : inherits ? "片段提示词（这一镜的要求，和全局提示词一起交给改写模型）" : "片段提示词（只按这一镜写，不看全局提示词）";
        parts.push(this.field(label, this.promptArea,
            h("div.chips", {}, ...this.tags().map(([tag, label]) => h("button", { onclick: () => this.insertTag(tag), title: label }, `${tag} ${label}`)))));
        if (auto) {
            const text = this.node.properties?.yh3Prompts?.[shot.id];
            parts.push(this.field("改写结果", h("div.report", {}, text || "（运行或在“剧本”页生成后显示）"),
                text ? h("button", { onclick: () => { shot.prompt = text; shot.rewrite = false; this.commit(true); } }, "采用为手写提示词") : null));
        }
        const audioModes = shot.kind === "source" && !shot.source.regen ? ["source", "soundtrack", "silent"] : ["generate", "soundtrack", "silent"].concat(shot.kind === "source" ? ["source"] : []);
        parts.push(this.field("声音", h("select", {
            onchange: e => { shot.audio = e.target.value; this.commit(true); },
        }, ...audioModes.map(m => h("option", { value: m, selected: shot.audio === m }, AUDIO_LABELS[m])))));
        if (shot.audio === "soundtrack" && !this.project.soundtrack) parts.push(h("div.warn", {}, "在“工程”里先设置配乐音轨。"));
        parts.push(h("div.row", {}, this.reseed(shot), this.project.refine.enabled ? this.check(shot, "refine", "参与二采") : null));
        return parts;
    }

    storyPane() {
        const p = this.project, config = rewriterConfig(this.node);
        const prompts = this.node.properties?.yh3Prompts || {};
        const status = h("div.dim", {}, (this.node.properties?.yh3RewriteReport || []).join("；"));
        const run = async all => {
            p.rewrite.enabled = true;
            if (all) p.rewrite.seed = (p.rewrite.seed || 0) + 1;
            this.commit();
            status.className = "dim";
            status.textContent = "改写中…改写模型正在读 H3 写作指南并逐镜编写，通常需要几十秒。";
            try {
                const data = await previewRewrite(this.node, p);
                storePrompts(this.node, data.prompts, data.report);
                this.renderInspector();
            } catch (error) {
                status.className = "warn";
                status.textContent = error.message;
            }
        };
        return [
            h("div.note", {}, "全局提示词写整部片的情况，打开自动改写后交给改写模型（deepagents + MiniMax h3-prompt-writing 技能）作为背景；它按时间线上每一镜的时长、续接、首尾帧、参考素材和声音，判断这一镜该拍什么，写成完整的 H3 提示词。每镜的片段提示词是这一镜自己的要求，可写可不写；某一镜关掉“继承全局提示词”就只按它自己的写。不开自动改写时只用片段提示词。"),
            config ? h("div.row", {}, h("span.dim", {}, "改写模型"), h("b", {}, `${config.provider}:${config.model}`), config.vision ? h("span.dim", {}, "· 看图") : null)
                : h("div.warn", {}, "还没有把“Yogurt H3 改写模型”节点连接到导演台的 rewriter。"),
            this.check(p.rewrite, "enabled", "自动改写提示词（运行时按剧情写每一镜）", () => this.commit(true)),
            this.field("全局提示词（整部片的剧情、人物、风格、场景）", h("textarea", { rows: 8, value: p.story, placeholder: "例如：雨夜，穿红雨衣的女孩独自走在霓虹街头，停下来回头一笑，然后走进一家咖啡馆，在窗边坐下。",
                oninput: e => { p.story = e.target.value; this.commit(); } })),
            this.field("全局要求（风格、语言、节奏，可留空）", h("textarea", { rows: 3, value: p.rewrite.notes, placeholder: "例如：写实电影感，镜头克制；台词用中文。",
                oninput: e => { p.rewrite.notes = e.target.value; this.commit(); } })),
            h("div.row", {}, h("button.primary", { onclick: () => run(false), disabled: !config }, "生成剧本（只改写，不渲染）"),
                h("button", { onclick: () => run(true), disabled: !config, title: "换一个改写结果；所有自动改写的镜头都会重写" }, "全部重新改写")),
            status,
            h("b", {}, "各镜提示词"),
            ...this.windows.map(w => {
                const shot = w.shot, auto = rewritable(p, shot);
                const text = auto ? prompts[shot.id] : shot.prompt;
                return h("div.card", {},
                    h("header.row", {}, h("b", {}, `第 ${w.index + 1} 镜 · ${secs(w.keep)}`),
                        h("span.dim", {}, auto ? (text ? "自动改写" : "待改写") + (shot.inherit_prompt === false ? " · 不继承全局" : "") : "手写"), h("span.grow"),
                        h("button.ghost", { onclick: () => { this.select(w.index); this.tab = "shot"; this.renderTabs(); this.renderInspector(); } }, "编辑本镜"),
                        auto && text ? h("button", { onclick: () => { shot.prompt = text; shot.rewrite = false; this.commit(true); } }, "采用为手写") : null),
                    auto && shot.prompt ? h("div.dim", {}, `片段提示词：${shot.prompt}`) : null,
                    h("div.report", {}, text || (auto ? "（运行或点“生成剧本”后显示）" : "（空）")));
            }),
        ];
    }

    setKind(kind) {
        const shot = this.shot;
        if (shot.kind === kind) return;
        shot.kind = kind;
        shot.source = kind === "source" ? { media: null, start: 0, regen: false, strength: 1 } : null;
        shot.audio = kind === "source" ? "source" : "generate";
        shot.continue = kind === "generate";
        this.commit(true);
    }

    keysPane() {
        const shot = this.shot, w = this.win;
        const find = at => shot.keyframes.find(k => k.at === at);
        const put = (at, media) => {
            shot.keyframes = shot.keyframes.filter(k => k.at !== at);
            if (media) shot.keyframes.push({ at, media });
            this.commit(true);
        };
        const middles = shot.keyframes.filter(k => k.at !== 0 && k.at !== -1).sort((a, b) => a.at - b.at);
        const referenced = refCount(effectiveRefs(this.project, shot)) > 0 || (shot.kind === "source" && shot.source.regen);
        return [
            referenced ? h("div.warn", {}, "这一镜带参考素材（含全局参考），按参考生成：首尾帧和中间帧都作为固定画面锚点，不走 FL2V 首尾帧模式。要用首尾帧模式，在“参考”页关掉继承全局参考并清空本镜参考。") : null,
            h("div.note", {}, w.head
                ? `本镜续接上一镜：开头 ${w.head} 帧来自上一镜，“首帧”会锚在续接后的第一帧。`
                : "首帧 / 尾帧固定镜头的起止画面（官方 FL2V）。只放尾帧也可以。中间帧把画面锚在指定帧。"),
            h("div.slots", {},
                this.slot(find(0)?.media, { tag: "首帧", capture: true, onpick: m => put(0, m) }),
                this.slot(find(-1)?.media, { tag: "尾帧", capture: true, onpick: m => put(-1, m) })),
            h("b", {}, "中间关键帧"),
            ...middles.map(k => h("div.card", {},
                h("div.row", {}, h("span", {}, "第"), this.number(k, "at", { min: 1, max: w.keep - 2, after: () => this.commit(true) }), h("span", {}, `帧（${secs(k.at)}）`),
                    h("span.grow"), h("button.danger", { onclick: () => { shot.keyframes.splice(shot.keyframes.indexOf(k), 1); this.commit(true); } }, "删除")),
                this.slot(k.media, { capture: true, onpick: m => { if (m) { k.media = m; } else { shot.keyframes.splice(shot.keyframes.indexOf(k), 1); } this.commit(true); } }))),
            h("button", {
                onclick: async () => {
                    const at = clamp(this.frame, 1, w.keep - 2);
                    let media = null;
                    try { media = await this.captureFrame(); } catch { media = null; }
                    if (!media) { const file = await pickFile("image/*"); if (!file) return; media = await upload(file); }
                    shot.keyframes.push({ at, media });
                    this.commit(true);
                },
            }, `+ 在当前帧（第 ${this.frame + 1} 帧）加关键帧`),
            h("div.dim", {}, "提示：“截取当前画面”可以把上一镜成片的最后一帧拿来做下一镜首帧，用于硬切后的人物衔接。"),
        ];
    }

    /** Upload slots for one reference set; ``offset`` is where its <Picture N> / <Video N> numbering starts. */
    refSlots(refs, offset) {
        const list = (items, options, build) => h("div.slots", {}, ...items.map(build),
            this.slot(null, { ...options, onpick: m => { if (m) { items.push(options.wrap ? options.wrap(m) : m); this.commit(true); } } }));
        return [
            h("span.dim", {}, `图片 ${refs.images.length}`),
            list(refs.images, { accept: "image/*" }, (m, i) => this.slot(m, { tag: `<Picture ${offset.images + i + 1}>`,
                onpick: v => { v ? (refs.images[i] = v) : refs.images.splice(i, 1); this.commit(true); } })),
            h("span.dim", {}, `视频 ${refs.videos.length}`),
            list(refs.videos, { accept: "video/*", wrap: m => ({ media: m, audio: false }) }, (v, i) => h("div.field", {},
                this.slot(v.media, { accept: "video/*", tag: `<Video ${offset.videos + i + 1}>`, onpick: m => { m ? (v.media = m) : refs.videos.splice(i, 1); this.commit(true); } }),
                this.check(v, "audio", "带原声", () => this.commit(true)))),
            h("span.dim", {}, `音频 ${refs.audios.length}`),
            list(refs.audios, { accept: "audio/*,video/*", graph: null }, (m, i) => this.slot(m, { accept: "audio/*,video/*", graph: null, tag: "音频",
                onpick: v => { v ? (refs.audios[i] = v) : refs.audios.splice(i, 1); this.commit(true); } })),
        ];
    }

    refsPane() {
        const p = this.project, shot = this.shot;
        const source = shot.kind === "source" && shot.source.regen ? 1 : 0;
        const shared = effectiveRefs(p, shot).shared;
        return [
            h("div.note", {}, "参考素材让模型保持人物、服装、场景或声音（官方 R2V）。提示词里用 <Picture N> / <Video N> / <Audio N> 指代，编号见“镜头”页的标签；带参考的镜头使用 ref_model。H3 每段建议合计不超过 9 张图、3 段视频、3 段音频。"),
            h("b", {}, "全局参考（每一段都带上）"),
            ...this.refSlots(p.refs, { images: 0, videos: source }),
            h("b", {}, `第 ${this.selected + 1} 镜参考`),
            this.check(shot, "inherit_refs", "继承全局参考（关掉则这一镜只用下面自己的参考）", () => this.commit(true)),
            source ? h("div.dim", {}, "整体改写时源视频是 <Video 1>，参考视频接在它后面编号。") : null,
            shared.images || shared.videos ? h("div.dim", {}, "本镜参考的编号接在全局参考之后。") : null,
            ...this.refSlots(shot.refs, { images: shared.images, videos: shared.videos + source }),
        ];
    }

    repaintPane() {
        const shot = this.shot;
        const cards = shot.repaints.map((layer, i) => {
            const masks = graphInputs(this.node, "mask");
            return h(`div.card${this.paint === i ? ".active" : ""}`, {},
                h("header.row", {}, h("b", {}, `重绘 ${i + 1}`), this.check(layer, "enabled", "启用"), h("span.grow"),
                    h("button.danger", { onclick: () => { shot.repaints.splice(i, 1); this.commit(true); } }, "删除")),
                h("div.row", {},
                    h("div.slot.filled", { style: { width: "84px", aspectRatio: `${this.project.width}/${this.project.height}`, backgroundImage: layer.mask && !layer.mask.input ? `url("${mediaUrl(layer.mask)}")` : "none" } },
                        layer.mask ? (layer.mask.input ? mediaLabel(layer.mask) : "") : "无遮罩"),
                    h("div.field.grow", {},
                        h("button.primary", { onclick: () => this.startPaint(i) }, layer.mask && !layer.mask.input ? "编辑遮罩" : "在画面上绘制遮罩"),
                        masks.length ? h("select", { onchange: e => { layer.mask = e.target.value ? { input: e.target.value } : null; this.commit(true); } },
                            h("option", { value: "" }, "使用遮罩输入…"), ...masks.map(m => h("option", { value: m.input, selected: layer.mask?.input === m.input }, m.label))) : null)),
                this.field("重绘强度（1 = 遮罩内完全重新生成）", this.slider(layer, "strength", 0.05, 1, 0.05)),
                this.range(layer),
                this.field("本层提示词（留空沿用镜头提示词）", h("textarea", { rows: 3, value: layer.prompt, oninput: e => { layer.prompt = e.target.value; this.commit(); } })),
                h("div.row", {}, this.field("遮罩扩展 px", this.number(layer, "grow", { min: 0, max: 256 })),
                    this.field("边缘羽化 px", this.number(layer, "feather", { min: 0, max: 256 })), this.reseed(layer)));
        });
        return [
            h("div.note", {}, this.shot.kind === "source" && !this.shot.source.regen
                ? "白色遮罩区域按提示词重新生成，遮罩外保持源视频原样。遮罩画在 24fps 帧上，可设帧范围；多帧遮罩可从“遮罩输入”接入（如 SAM 跟踪结果）。"
                : "在本镜成片上画遮罩，只重新生成遮罩区域（先运行一次得到成片）。接了 inpaint_patch（Fun ControlNet）时，遮罩区更贴合周围画面；要在遮罩里加入新物体，不接效果更明显。"),
            ...cards,
            h("button", { onclick: () => { shot.repaints.push({ id: uid(), enabled: true, mask: null, start: 0, end: -1, strength: 1, prompt: "", grow: 8, feather: 8, seed: 0 }); this.commit(true); this.startPaint(shot.repaints.length - 1); } }, "+ 新建重绘层"),
        ];
    }

    detailPane() {
        const shot = this.shot;
        const cards = shot.details.map((item, i) => {
            const picker = h("select");
            detectors().then(list => {
                picker.replaceChildren(...(list.length ? list : ["face_yolov8m.pt"]).map(name => h("option", { value: name, selected: item.detector === name }, name)));
            });
            picker.onchange = e => { item.detector = e.target.value; this.commit(); };
            const [x, y, bw, bh] = item.box;
            return h(`div.card${this.boxing === i ? ".active" : ""}`, {},
                h("header.row", {}, h("b", {}, `精修 ${i + 1}`), this.check(item, "enabled", "启用"), h("span.grow"),
                    h("button.danger", { onclick: () => { shot.details.splice(i, 1); this.commit(true); } }, "删除")),
                h("div.row", {}, h("button.primary", { onclick: () => this.startBox(i) }, "在画面上框选区域"),
                    h("span.dim", {}, `${Math.round(x * 100)}%,${Math.round(y * 100)}% · ${Math.round(bw * 100)}%×${Math.round(bh * 100)}%`)),
                this.field("区域", this.seg([["none", "固定区域"], ["face", "跟随人脸"]], item.follow, v => { item.follow = v; this.commit(true); })),
                item.follow === "face" ? this.field("人脸检测器（models/ultralytics/bbox）", picker) : null,
                this.field("精修强度（越高细节越多，也越可能改变长相）", this.slider(item, "denoise", 0.1, 0.9, 0.05)),
                this.field("精修清晰度（裁剪后放大到的像素量，MP）", this.slider(item, "megapixels", 0.2, 1, 0.05)),
                this.range(item),
                this.field("精修提示词（留空沿用镜头提示词）", h("textarea", { rows: 3, value: item.prompt, placeholder: "例如：清晰的面部特写，皮肤细节自然", oninput: e => { item.prompt = e.target.value; this.commit(); } })),
                this.reseed(item));
        });
        return [
            h("div.note", {}, "局部精修在成片上裁出区域、放大后低强度重采样，再羽化贴回（脸、手、小物体）。“跟随人脸”从框内的人脸开始跟踪，适合多人画面。"),
            ...cards,
            h("button", { onclick: () => {
                shot.details.push({ id: uid(), enabled: true, box: [0.3, 0.15, 0.4, 0.5], follow: "none", detector: "face_yolov8m.pt", start: 0, end: -1, prompt: "", denoise: 0.4, megapixels: 0.5, seed: 0 });
                this.commit(true);
                this.startBox(shot.details.length - 1);
            } }, "+ 新建精修区域"),
        ];
    }

    projectPane() {
        const p = this.project;
        const aspect = p.width / p.height;
        const current = ASPECTS.find(([, a]) => Math.abs(a - aspect) < 0.02)?.[0] || "custom";
        const mp = Number((p.width * p.height / 1e6).toFixed(2));
        const setCanvas = (a, m) => { [p.width, p.height] = canvasFor(a, m); this.commit(true); };
        const refineOut = `${Math.round(p.width * p.refine.scale / 32) * 32}×${Math.round(p.height * p.refine.scale / 32) * 32}`;
        const report = this.result?.report;
        return [
            h("b", {}, "画面"),
            this.field("画幅", this.seg(ASPECTS.map(([name]) => [name, name]), current, name => setCanvas(ASPECTS.find(a => a[0] === name)[1], mp))),
            this.field(`首采像素量：${p.width}×${p.height}（${mp} MP）`, h("input", {
                type: "range", min: 0.2, max: 1.05, step: 0.01, value: mp, onchange: e => setCanvas(aspect, Number(e.target.value)),
            })),
            h("div.row", {}, this.field("宽", this.number(p, "width", { min: 64, max: 4096, step: 32, multiple: 32, after: () => this.commit(true) })),
                this.field("高", this.number(p, "height", { min: 64, max: 4096, step: 32, multiple: 32, after: () => this.commit(true) }))),
            this.field("续接重叠帧", h("select", { onchange: e => { p.overlap = Number(e.target.value); this.commit(true); } },
                ...OVERLAPS.map(v => h("option", { value: v, selected: p.overlap === v }, v ? `${v} 帧（${secs(v)}）` : "关闭续接")))),
            this.field("参考图尺寸", this.seg([["match", "匹配画面"], ["max", "最高清（慢）"]], p.ref_image_size, v => { p.ref_image_size = v; this.commit(true); })),
            this.field("配乐音轨（镜头声音选“配乐音轨”时按时间线切分并锁定）",
                this.slot(p.soundtrack, { accept: "audio/*,video/*", graph: null, onpick: m => { p.soundtrack = m; this.commit(true); } })),
            this.field("负面提示词（cfg ≠ 1 时生效）", h("textarea", { rows: 2, value: p.negative_prompt, oninput: e => { p.negative_prompt = e.target.value; this.commit(); } })),
            h("b", {}, "二采放大"),
            this.check(p.refine, "enabled", "开启二采（全片放大后低强度再采样）", () => this.commit(true)),
            p.refine.enabled ? [
                this.field(`放大倍数 → ${refineOut}`, this.slider(p.refine, "scale", 1, 2.5, 0.125, v => `${v.toFixed(3)}×`)),
                this.field("放大方式", this.seg([["lanczos", "Lanczos"], ["model", "放大模型（接 upscale_model）"]], p.refine.upscaler, v => { p.refine.upscaler = v; this.commit(true); })),
                h("div.row", {}, this.field("二采步数", this.number(p.refine, "steps", { min: 1, max: 200 })),
                    this.field("二采强度", this.number(p.refine, "denoise", { min: 0.05, max: 1, step: 0.05 }))),
            ] : null,
            h("b", {}, "局部精修"),
            this.field("精修步数", this.number(p.detail, "steps", { min: 1, max: 200 })),
            h("b", {}, "工程"),
            h("div.row", {},
                h("button", { onclick: () => this.exportProject() }, "导出工程 JSON"),
                h("button", { onclick: () => this.importProject() }, "导入工程 JSON")),
            report?.length ? [h("b", {}, "上次运行"), h("div.report", {}, report.join("\n"))] : null,
        ].flat();
    }

    exportProject() {
        const blob = new Blob([JSON.stringify(this.project, null, 2)], { type: "application/json" });
        const link = h("a", { href: URL.createObjectURL(blob), download: "yogurt_h3_project.json" });
        link.click();
        URL.revokeObjectURL(link.href);
    }

    async importProject() {
        const file = await pickFile(".json,application/json");
        if (!file) return;
        try {
            const data = JSON.parse(await file.text());
            if (!Array.isArray(data.shots)) throw new Error("不是导演台工程文件。");
            if (!confirm("导入会替换当前全部镜头，继续？")) return;
            writeProject(this.node, data);
            this.project = readProject(this.node);
            this.selected = 0;
            this.commit(true);
        } catch (error) {
            alert(`导入失败：${error.message}`);
        }
    }
}
