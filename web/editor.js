// Full-screen director: stage (result / source / keyframe preview with paint overlay), inspector, timeline.
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import {
    FPS, SHARED, STAGES, changed, clamp, clone, effectiveRefs, features, follows, graphInputs, h, inputPreview, isAudio, isVideo, mediaLabel, mediaUrl,
    refCount, newLayer, newShot, pickFile, planWindows, previewRewrite, projectData, readProject, rewritable, rewriterConfig, secs, sourceFile,
    storePrompts, totalFrames, uid, upload, widget, writeProject,
} from "./core.js";
import { Painter } from "./painter.js";

/** Material position as m:ss.ss, or s.ss for clips under a minute. */
const clipTime = frames => {
    const seconds = frames / FPS, minutes = Math.floor(seconds / 60);
    return minutes ? `${minutes}:${(seconds - minutes * 60).toFixed(2).padStart(5, "0")}` : `${seconds.toFixed(2)}s`;
};
const AUDIO_LABELS = { generate: "模型生成", soundtrack: "配乐音轨（锁定）", source: "源视频原声", silent: "静音" };
const TABS = [["story", "剧本"], ["shot", "镜头"], ["keys", "首尾帧"], ["refs", "参考"], ["repaint", "局部重绘"]];
const SOURCES = [["paint", "手绘"], ["input", "连线遮罩"], ["text", "文字识别"], ["face", "跟随人脸"], ["box", "固定框"]];
const boxed = layer => layer.source === "face" || layer.source === "box";

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
        this.view = null;
        this.trim = null;
        this.playing = false;
        this.paint = -1;
        this.boxing = -1;
        // 自动分段 settings, in frames; segments 0 lets the planner choose the count.
        this.splitting = { shortest: 5 * FPS, longest: 10 * FPS, segments: 0, cover: true };
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
        this.scrubber = h("input", { type: "range", min: 0, value: 0, oninput: e => { this.stopPlay(); this.seek(Number(e.target.value)); } });
        this.trimBar = h("div.trimbar", { hidden: true });
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
                h("button", { onclick: () => this.importProject(), title: "从 JSON 文件导入镜头、参考和剧本（画幅、二采等在节点参数里）" }, "导入工程"),
                h("button", { onclick: () => this.exportProject() }, "导出工程"),
                h("button.primary", { onclick: () => this.run() }, "▶ 运行全片"),
                h("button", { onclick: () => this.close(), title: "Esc" }, "关闭")),
            h("main", {},
                h("section.stage", {}, this.viewport,
                    h("div.transport", {}, this.playButton, this.scrubber, this.clock), this.trimBar, this.toolbar),
                h("aside", {}, this.tabs, this.pane)),
            h("footer", {}, this.footerTools, this.ruler, this.track));
        this.root.append(frame);
        this.root.addEventListener("pointerdown", e => { if (e.target === this.root) this.close(); });
        document.body.append(this.root);
        this.root.focus();
        this.resizer = new ResizeObserver(() => { this.fitScreen(); this.renderTimeline(); });
        this.resizer.observe(this.viewport);
        this.resizer.observe(this.track);
        this.video.addEventListener("loadedmetadata", () => this.renderTrim());
        this.video.addEventListener("loadeddata", () => this.showFrame());
        this.unlisten = this.panel.listen(() => this.renderTimeline());
        this.refresh();
    }

    close() {
        this.stopPlay();
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
        else if (e.key === "ArrowLeft") { this.stopPlay(); this.seek(this.frame - (e.shiftKey ? FPS : 1)); }
        else if (e.key === "ArrowRight") { this.stopPlay(); this.seek(this.frame + (e.shiftKey ? FPS : 1)); }
        else if (e.key === "i" || e.key === "I") this.markIn();
        else if (e.key === "o" || e.key === "O") this.markOut();
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

    select(index, frame = 0, playing = false) {
        if (!playing) this.stopPlay();
        this.stopTools();
        this.selected = clamp(index, 0, this.project.shots.length - 1);
        this.frame = frame;
        this.view = null;
        this.trim = null;
        this.refresh();
    }

    run() {
        writeProject(this.node, this.project);
        app.queuePrompt(0, 1);
        this.panel.say("已加入队列。");
    }

    // ------------------------------------------------------------------ stage

    /** The last render is on the stage only when chosen, or by default where repaint layers are drawn on it. */
    showsResult() {
        if (!this.result?.timeline?.some(e => e.id === this.shot.id)) return false;
        if (this.view) return this.view === "result";
        return this.shot.kind === "generate" && this.tab === "repaint";
    }

    /** Storyboard frame of a generated shot: the last keyframe at or before kept frame ``frame``, else the first one. */
    keyAt(frame) {
        const keys = this.shot.keyframes.filter(k => this.preview(k.media)?.url)
            .map(k => ({ ...k, pos: k.at === -1 ? this.win.keep - 1 : clamp(k.at, 0, this.win.keep - 1) })).sort((a, b) => a.pos - b.pos);
        return keys.filter(k => k.pos <= frame).pop() || keys[0] || null;
    }

    /** What the stage shows at the playhead: the shot's material on the timeline, or the last render when chosen. */
    stageSource() {
        const shot = this.shot, entry = this.result?.timeline?.find(e => e.id === shot.id);
        if (this.trim) {
            const clip = this.trim.clip;
            return { kind: "video", url: this.preview(clip.media).url, base: clip.start, label: `素材：参考视频 ${this.trim.tag}`, trim: clip, length: this.win.gen };
        }
        if (this.showsResult()) return { kind: "video", url: mediaUrl(this.result.video), base: entry.start, label: "上次成片", stale: entry.frames !== this.win.keep };
        if (shot.kind === "source") {
            const shown = this.preview(shot.source.media);
            return shown?.video ? { kind: "video", url: shown.url, base: shot.source.start, label: "素材：源视频", trim: shot.source, length: this.win.keep }
                : { kind: "none" };
        }
        return { kind: "storyboard", label: "素材：关键帧" };
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
        this.image.style.display = source.kind === "storyboard" ? "" : "none";
        this.placeholder.style.display = source.kind === "none" ? "" : "none";
        this.placeholder.textContent = this.shot.source?.media?.input ? `${this.shot.source.media.input} 不是直接接的 LoadVideo，导演台里放不了：起点在“镜头”页按秒设置，运行后切到“成片”查看。`
            : "选择源视频后，这里按时间线显示这一镜用到的素材。";
        if (source.kind === "video" && this.video.dataset.url !== source.url) {
            this.video.dataset.url = source.url;
            this.video.src = source.url;
        }
        this.scrubber.max = Math.max(0, this.win.keep - 1);
        this.renderTrim();
        this.seek(this.frame);
        this.renderToolbar();
        this.painter.boxes = this.tab === "repaint" ? this.shot.repaints.map(layer => (boxed(layer) ? layer.box : null)) : [];
        this.painter.active = this.boxing;
        this.painter.render();
    }

    seek(frame) {
        this.frame = clamp(Math.round(frame), 0, Math.max(0, this.win.keep - 1));
        this.scrubber.value = this.frame;
        this.showFrame();
        this.renderClock();
        this.renderPlayhead();
        this.renderTrimHead();
    }

    showFrame() {
        const stage = this.stage;
        if (stage?.kind === "storyboard") {
            const key = this.keyAt(this.frame), url = this.preview(key?.media)?.url;
            // Preview the keyframe the way it meets the canvas; "auto" stretches the generation's first frame, crops the rest.
            const mode = key && (key.fit !== "auto" ? key.fit : key.pos + this.win.head === 0 ? "stretch" : "crop");
            this.image.style.objectFit = { crop: "cover", pad: "contain", stretch: "fill" }[mode] || "fill";
            this.image.style.display = url ? "" : "none";
            this.placeholder.style.display = url ? "none" : "";
            if (!url) this.placeholder.textContent = "生成镜头还没有画面：在“首尾帧”放关键帧，就能按时间线预览构图；运行后可切到“成片”。";
            else if (this.image.dataset.url !== url) { this.image.dataset.url = url; this.image.src = url; }
            return;
        }
        if (stage?.kind !== "video" || !this.video.paused || !this.video.readyState) return;
        const time = (stage.base + this.frame + 0.5) / FPS;
        if (Math.abs(this.video.currentTime - time) > 0.5 / FPS) this.video.currentTime = time;
    }

    /** Plays the timeline from the playhead to the end of the film, shot after shot. */
    togglePlay() {
        if (this.playing) return this.stopPlay();
        if (this.selected === this.project.shots.length - 1 && this.frame >= this.win.keep - 1) this.select(0);
        this.playing = true;
        this.playButton.textContent = "❚❚";
        this.startClock();
        requestAnimationFrame(now => this.tick(now));
    }

    stopPlay() {
        if (!this.playing) return;
        this.playing = false;
        this.video.pause();
        this.playButton.textContent = "▶";
    }

    /** Video material plays itself (with its sound); keyframes and empty shots follow a wall clock. */
    startClock() {
        this.clock0 = performance.now();
        this.frame0 = this.frame;
        if (this.stage.kind === "video") {
            this.video.currentTime = (this.stage.base + this.frame + 0.5) / FPS;
            this.video.muted = false;
            this.video.play().catch(() => this.stopPlay());
        }
    }

    tick(now) {
        if (!this.playing) return;
        const video = this.stage.kind === "video";
        const frame = !video ? this.frame0 + Math.floor((now - this.clock0) * FPS / 1000)
            : this.video.ended ? this.win.keep : Math.floor(this.video.currentTime * FPS) - this.stage.base;
        if (frame >= this.win.keep) {
            if (this.selected + 1 >= this.project.shots.length) {
                this.stopPlay();
                this.seek(this.win.keep - 1);
                return;
            }
            this.select(this.selected + 1, 0, true);
            this.startClock();
        } else if (frame !== this.frame) {
            this.frame = Math.max(0, frame);
            this.scrubber.value = this.frame;
            if (!video) this.showFrame();
            this.renderClock();
            this.renderPlayhead();
            this.renderTrimHead();
        }
        requestAnimationFrame(t => this.tick(t));
    }

    renderClock() {
        const w = this.win;
        const material = this.stage?.trim ? ` · 素材 ${clipTime(this.stage.trim.start + this.frame)}` : "";
        this.clock.textContent = `第 ${this.frame + 1}/${w.keep} 帧 · ${secs(this.frame)} · 全片 ${secs(w.start + this.frame)}${material}`;
    }

    // ------------------------------------------------------------------ material trim

    /** The whole clip under the stage: the highlighted part is what this shot (or reference) uses. */
    renderTrim() {
        const stage = this.stage, bar = this.trimBar;
        bar.hidden = !stage?.trim;
        this.trimTotal = 0;
        if (!stage?.trim) return;
        const duration = this.video.dataset.url === stage.url ? this.video.duration : NaN;
        if (!Number.isFinite(duration) || duration <= 0) {
            bar.replaceChildren(h("span.dim", {}, "读取素材时长…"));
            return;
        }
        const clip = stage.trim, length = stage.length, total = this.trimTotal = Math.max(1, Math.floor(duration * FPS));
        const source = clip === this.shot.source;
        const pct = frames => `${clamp(frames / total, 0, 1) * 100}%`;
        const win = h("div.window", { style: { left: pct(clip.start), width: pct(length) }, title: "拖动：选择用素材的哪一段" });
        const lane = h("div.lane", { title: "点击：把选区移到这里" }, win);
        this.trimHead = h("i.head");
        lane.append(this.trimHead);
        const info = h("span.dim", {}, `素材 ${clipTime(total)} · 使用 ${clipTime(clip.start)} – ${clipTime(clip.start + length)}（${secs(length)}）`);
        const drag = (event, apply) => {
            event.preventDefault();
            event.stopPropagation();
            this.stopPlay();
            const x0 = event.clientX, perPixel = total / lane.clientWidth;
            const move = e => {
                apply(Math.round((e.clientX - x0) * perPixel));
                win.style.left = pct(clip.start);
                win.style.width = pct(stage.length);
                stage.base = clip.start;
                info.textContent = `素材 ${clipTime(total)} · 使用 ${clipTime(clip.start)} – ${clipTime(clip.start + stage.length)}（${secs(stage.length)}）`;
                this.showFrame();
                this.renderClock();
                this.renderTrimHead();
            };
            const up = () => {
                window.removeEventListener("pointermove", move);
                window.removeEventListener("pointerup", up);
                this.commit(true);
            };
            window.addEventListener("pointermove", move);
            window.addEventListener("pointerup", up);
            move(event);
        };
        const slide = event => {
            const start0 = clip.start;
            drag(event, delta => { clip.start = clamp(start0 + delta, 0, Math.max(0, total - length)); });
        };
        win.addEventListener("pointerdown", slide);
        lane.addEventListener("pointerdown", e => {
            if (e.target !== lane) return;
            const x = e.clientX - lane.getBoundingClientRect().left;
            clip.start = clamp(Math.round(x / lane.clientWidth * total - length / 2), 0, Math.max(0, total - length));
            slide(e);
        });
        if (source) {
            const out = h("i.out", { title: "拖动：出点（本镜时长）" });
            out.addEventListener("pointerdown", e => {
                const frames0 = this.shot.frames;
                drag(e, delta => {
                    this.shot.frames = clamp(frames0 + delta, 5, Math.max(5, total - clip.start));
                    stage.length = this.win.keep;
                });
            });
            win.append(out);
        }
        bar.replaceChildren(lane, h("div.row", {}, info, h("span.grow"),
            source ? [h("button", { onclick: () => this.markIn(), title: "I" }, "当前帧设为入点"), h("button", { onclick: () => this.markOut(), title: "O" }, "当前帧设为出点")] : null));
        this.renderTrimHead();
    }

    renderTrimHead() {
        if (this.trimTotal && this.trimHead) this.trimHead.style.left = `${clamp((this.stage.trim.start + this.frame) / this.trimTotal, 0, 1) * 100}%`;
    }

    /** In point at the playhead; the out point stays where it was. */
    markIn() {
        const shot = this.shot;
        if (shot.kind !== "source" || this.stage?.trim !== shot.source) return;
        const out = shot.source.start + shot.frames;
        shot.source.start += this.frame;
        shot.frames = Math.max(5, out - shot.source.start);
        this.frame = 0;
        this.commit(true);
    }

    markOut() {
        const shot = this.shot;
        if (shot.kind !== "source" || this.stage?.trim !== shot.source) return;
        shot.frames = Math.max(5, this.frame + 1);
        this.commit(true);
    }

    async captureFrame() {
        const canvas = h("canvas", { width: this.project.width, height: this.project.height });
        const source = this.stage?.kind === "video" ? this.video : this.stage?.kind === "storyboard" && this.image.style.display !== "none" ? this.image : null;
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
            bar.append(h("b", {}, `在画面上拖出重绘 ${this.boxing + 1} 的区域`), h("span.grow"), h("button", { onclick: () => this.stopTools() }, "取消"));
            return;
        }
        if (this.trim) {
            bar.append(h("b", {}, `截取参考视频 ${this.trim.tag}`), h("span.dim", {}, `拖动素材条上的选区，选出 H3 参考的那 ${secs(this.win.gen)}`), h("span.grow"),
                h("button.primary", { onclick: () => { this.trim = null; this.renderStage(); this.renderInspector(); } }, "完成"));
            return;
        }
        const source = this.stage || {}, result = this.showsResult();
        bar.append(h("span.dim", {}, source.label ? `显示：${source.label}` : "无画面"));
        if (source.stale) bar.append(h("span.warn", { style: { padding: "2px 8px" } }, "时长已改，成片与当前设置不一致，重新运行后更新"));
        if (this.result?.timeline?.some(e => e.id === this.shot.id)) {
            bar.append(h("div.seg", {},
                h(`button${!result ? ".on" : ""}`, { onclick: () => { this.view = "material"; this.renderStage(); } }, "素材"),
                h(`button${result ? ".on" : ""}`, { onclick: () => { this.view = "result"; this.renderStage(); } }, "成片")));
        }
        bar.append(h("span.grow"), h("span.dim", {}, source.trim ? "空格播放 · ←/→ 逐帧 · I / O 设入点 / 出点" : "空格播放全片 · ←/→ 逐帧 · Shift 加速"));
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
        await this.painter.loadMask(layer.mask ? mediaUrl(layer.mask) : null);
        this.painter.setMode("mask");
        this.renderTabs();
        this.renderInspector();
        this.renderToolbar();
    }

    async saveMask() {
        const layer = this.shot.repaints[this.paint];
        layer.source = "paint";
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
        this.tab = "repaint";
        this.boxing = index;
        this.painter.setMode("box");
        this.painter.onbox = box => {
            this.shot.repaints[index].box = box.map(v => Number(v.toFixed(4)));
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
        // Each pass divides the canvas it ends on by its scale, rounded like the backend's Python round (halves to even).
        const round32 = v => { const q = v / 32, f = Math.floor(q); return Math.max(64, (q - f === 0.5 ? f + (f % 2) : Math.round(q)) * 32); };
        let size = [p.width, p.height];
        const chain = [];
        for (const [name, scale] of [["RTX 超分", p.superres], ["低噪重绘", p.refine.rerender], ["两段式", p.refine.twoPass]]) {
            if (!scale) continue;
            chain.unshift(`${name} ${size.join("×")}`);
            size = size.map(v => round32(v / scale));
        }
        const refine = chain.length ? ` · 画幅：首采 ${size.join("×")} → ${chain.join(" → ")}` : "";
        this.meta.textContent = `${p.width}×${p.height}${refine} · ${p.shots.length} 镜 · ${secs(total)} · 24fps`;
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
        this.project.shots.splice(this.selected + 1, 0, shot);
        this.selected += 1;
        this.commit(true);
    }

    /**
     * 自动分段: the backend reads this source shot's cuts and motion and picks where segments join; the shot is replaced
     * by them, each keeping its settings, with keyframes and repaint ranges moved to the segment they fall in. A cover
     * the video opens with becomes its own shot that shows, in the film, the generated frame it repeats: back in the
     * film, never in a segment's reference video.
     */
    async autoSplit() {
        const shot = this.shot, file = sourceFile(this.node, shot.source.media);
        const { shortest, longest, segments, cover: detect } = this.splitting;
        if (!file) return this.panel.say("自动分段需要上传的视频文件或 LoadVideo 里的视频。");
        if (longest < shortest) return this.panel.say("自动分段：最长时长不能短于最短时长。");
        this.panel.say("自动分段：正在分析源视频的剪辑点和动作……");
        let result;
        try {
            const response = await api.fetchApi("/yogurt_h3/director/split", { method: "POST", body: JSON.stringify({
                media: file, start: shot.source.start, frames: shot.frames, shortest, longest, segments, cover: detect, overlap: this.project.overlap }) });
            result = await response.json();
            if (!response.ok) throw new Error(result.error || response.statusText);
        } catch (error) {
            return this.panel.say(`自动分段失败：${error.message}`);
        }
        const last = frame => (frame === -1 ? shot.frames - 1 : frame), cover = result.cover;
        const parts = result.segments.map((segment, k) => {
            const part = clone(shot), offset = segment.start - shot.source.start, end = offset + segment.frames - 1;
            Object.assign(part, { id: uid(), frames: segment.frames, continue: k ? segment.carry : !cover && shot.continue });
            part.source.start = segment.start;
            part.keyframes = shot.keyframes.filter(key => last(key.at) >= offset && last(key.at) <= end)
                .map(key => ({ ...clone(key), at: key.at === -1 ? -1 : key.at - offset }));
            part.repaints = shot.repaints.filter(layer => last(layer.end) >= offset && layer.start <= end).map(layer => ({
                ...clone(layer), id: uid(), start: Math.max(layer.start - offset, 0), end: last(layer.end) >= end ? -1 : layer.end - offset }));
            return part;
        });
        const still = cover && Object.assign(clone(shot), { id: uid(), frames: cover.frames, prompt: "", keyframes: [], repaints: [] });
        if (still) {
            still.source.regen = false;
            const copy = shot.source.start + cover.copy, k = result.segments.findIndex(s => s.start <= copy && copy < s.start + s.frames);
            still.still = { shot: parts[k].id, frame: copy - result.segments[k].start };
        }
        this.project.shots.splice(this.selected, 1, ...(still ? [still] : []), ...parts);
        this.frame = 0;
        this.commit(true);
        const first = this.selected + (still ? 1 : 0);
        const joins = parts.slice(1).map(p => (p.continue ? "续接" : "剪辑点硬切"));
        const cuts = result.segments.map((s, k) => (s.cuts.length ? `第 ${first + k + 1} 镜剪辑点 ${s.cuts.map(secs).join("、")}` : "")).filter(Boolean);
        const masks = shot.repaints.some(layer => layer.source === "input") ? "连线遮罩按整镜对齐，分段后请改用文字识别或手绘遮罩。" : "";
        const held = still && parts.findIndex(p => p.id === still.still.shot);
        const kept = still ? `开头 ${cover.frames} 帧是封面（与 ${secs(cover.copy)} 的画面相同），作为第 ${this.selected + 1} 镜不参与生成，成片里换成第 ${first + held + 1} 镜生成后的这一帧；` : "";
        this.panel.say(`自动分段：${kept}${parts.length} 段（${parts.map(p => secs(p.frames)).join(" / ")}）${joins.length ? `，接缝 ${joins.join("、")}` : ""}。`
            + `${cuts.length ? ` ${cuts.join("；")}。` : ""}${result.note}${masks}`);
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
                    this.stopPlay();
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
        shot.repaints.forEach(layer => span(layer.start, layer.end, layer.crop ? "#7aa7ff" : "#ff8a2a"));
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
            repaint: this.shot.repaints.length };
        this.tabs.replaceChildren(...TABS.map(([id, label]) => h(`button${this.tab === id ? ".on" : ""}`, {
            onclick: () => { this.stopTools(); this.tab = id; this.renderTabs(); this.renderInspector(); this.renderStage(); },
        }, counts[id] ? `${label} ${counts[id]}` : label)));
    }

    /** Rewritten prompts arrived: redraw the panes that show them, unless the user is editing in the side pane. */
    showPrompts() {
        if ((this.tab === "story" || this.tab === "shot") && !this.pane.contains(document.activeElement)) this.renderInspector();
    }

    renderInspector() {
        const builders = { story: this.storyPane, shot: this.shotPane, keys: this.keysPane, refs: this.refsPane, repaint: this.repaintPane };
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

    /** What a media item looks like here: an upload's own file, or what is known about a connected socket. */
    preview(media) {
        if (!media) return null;
        if (media.input) return inputPreview(this.node, media.input);
        return { url: mediaUrl(media), video: isVideo(media), audio: isAudio(media), name: media.filename };
    }

    /**
     * Media slot: click to upload, drop a file, pick connected material or capture the stage frame.
     * ``graph`` lists the socket kinds offered as connected material (e.g. ["video", "image"]).
     */
    slot(media, { accept = "image/*", tag, graph = ["image"], capture = false, onpick }) {
        const shown = this.preview(media);
        const el = h(`div.slot${media ? ".filled" : ""}`, {
            title: !media ? "点击上传，或把文件拖到这里" : media.input && !shown ? `${media.input}：上游不是加载节点，运行一次后显示画面` : mediaLabel(media) });
        if (shown?.video) el.append(h("video", { src: shown.url, muted: true, preload: "metadata" }));
        else if (shown?.audio) el.append(h("span", {}, `♪ ${shown.name}`));
        else if (shown?.url) el.style.backgroundImage = `url("${shown.url}")`;
        else el.append(h("span", {}, media ? mediaLabel(media) : "+ 上传"));
        if (tag) el.append(h("span.tag", {}, tag));
        if (media?.input && shown && !shown.audio) el.append(h("span.src", {}, media.input));
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
        const inputs = graphInputs(this.node, ...graph);
        if (inputs.length) {
            // The slot's own file stays on the item while a socket is picked, so switching back restores it.
            const own = media?.input ? media.file : media;
            extras.push(h("select", {
                onchange: e => { if (e.target.value) onpick({ input: e.target.value, ...(own && { file: own }) }); else if (own) onpick(own); },
            }, h("option", { value: "" }, own ? `上传：${own.filename}` : "连线素材…"),
                ...inputs.map(i => h("option", { value: i.input, selected: media?.input === i.input }, i.label))));
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
                this.slot(shot.source.media, { accept: "video/*", graph: ["video", "image"], onpick: m => { shot.source.media = m; this.commit(true); } }),
                this.privacy(shot.source)));
            parts.push(this.field("起点（秒）", this.number(shot.source, "start", { min: 0, step: 1 / FPS, scale: FPS, after: () => this.renderStage() })));
            parts.push(h("div.dim", {}, shot.source.media?.input && !this.preview(shot.source.media)?.video ? "这个连线源视频不是直接接的 LoadVideo，导演台里放不了，按秒设置起点。"
                : "剪辑素材：在舞台下方的素材条上拖动选区，选出这一镜用到的那一段；拖选区右缘改时长，或播放到某一帧按 I / O 设入点 / 出点。"));
            parts.push(this.field("改写方式", this.seg([[false, "保留原画 · 只改遮罩"], [true, "整体改写（参考 <Video 1>）"]], shot.source.regen,
                v => { shot.source.regen = v; this.commit(true); })));
            if (shot.source.regen) {
                parts.push(this.field("改写强度（1 = 只参考源视频，越低越贴近原画）", this.slider(shot.source, "strength", 0.05, 1, 0.05)));
            } else {
                parts.push(h("div.note", {}, "保留原画：遮罩外像素原样输出。在“局部重绘”画遮罩、在“局部精修”框区域。"));
            }
            const held = shot.still && this.project.shots.findIndex(s => s.id === shot.still.shot);
            if (held >= 0) parts.push(h("div.note", {}, `封面：成片里显示第 ${held + 1} 镜第 ${shot.still.frame + 1} 帧的生成结果。`));
        }
        parts.push(this.field("时长（秒）", h("div.row", {}, this.number(shot, "frames", { min: 5, max: FPS * 600, step: 0.125, scale: FPS, after: () => this.commit(true) }),
            h("span.dim", {}, `生成 ${w.gen} 帧${w.head ? `（含续接 ${w.head} 帧）` : ""} · 输出 ${w.keep} 帧`))));
        if (shot.kind === "source" && shot.source.media) {
            const seconds = key => this.number(this.splitting, key, { min: FPS, max: 15 * FPS, step: 0.5, scale: FPS });
            parts.push(this.field("自动分段（每段一个镜头：剪辑点处硬切，不到 1 秒的快切镜头合成一段；比最长时长还长的镜头在镜头内续接切开，切出的每段不短于最短时长）",
                h("div.row", {}, "最短", seconds("shortest"), "秒 · 最长", seconds("longest"), "秒 · 段数",
                    this.number(this.splitting, "segments", { min: 0, max: 99 }), h("span.dim", {}, "0 = 自动"),
                    h("button", { onclick: () => this.autoSplit() }, "自动分段")),
                this.check(this.splitting, "cover", "识别开头的封面帧（单独成一镜不参与生成，成片里换成它重复的那一帧的生成结果）")));
        }
        const source = shot.kind === "source";
        if (!first && (!source || follows(this.project.shots[this.selected - 1], shot))) {
            parts.push(h("div.field", {}, this.check(shot, "continue", source ? "接着上一镜的源视频续接（latent 续接，接缝处人物与画面连贯）"
                : "接着上一镜继续拍（latent 续接，画面与声音连贯）", () => this.commit(true)),
                !w.carry && shot.continue ? h("span.warn", {}, source ? "上一镜的生成长度不在 17k+5 上或工程关闭了续接，本镜直接接上。"
                    : "上一镜过短或工程关闭了续接，本镜将硬切开始。") : null));
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
        if (shot.audio === "soundtrack") {
            parts.push(this.field("配乐音轨（全片共用：按时间线切到每个选了配乐的镜头）",
                this.slot(this.project.soundtrack, { accept: "audio/*,video/*", graph: ["audio", "video"], onpick: m => { this.project.soundtrack = m; this.commit(true); } })));
        }
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
            this.result?.report?.length ? [h("b", {}, "上次运行"), h("div.report", {}, this.result.report.join("\n"))] : null,
        ];
    }

    setKind(kind) {
        const shot = this.shot;
        if (shot.kind === kind) return;
        shot.kind = kind;
        shot.source = kind === "source" ? { media: null, start: 0, regen: false, strength: 1, ...SHARED } : null;
        shot.audio = kind === "source" ? "source" : "generate";
        shot.continue = kind === "generate";
        this.commit(true);
    }

    keysPane() {
        const shot = this.shot, w = this.win;
        const find = at => shot.keyframes.find(k => k.at === at);
        const put = (at, media) => {
            const kept = find(at) || { fit: "auto", ...SHARED };
            shot.keyframes = shot.keyframes.filter(k => k.at !== at);
            if (media) shot.keyframes.push({ ...kept, at, media });
            this.commit(true);
        };
        const ends = [[0, "首帧"], [-1, "尾帧"]].filter(([at]) => find(at));
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
            ends.length ? h("div.slots", {}, ...ends.map(([at, label]) => h("div.field", {},
                h("label.row", {}, label, this.fitSelect(find(at), "自动（首帧拉伸，其余裁剪）")), this.privacy(find(at))))) : null,
            h("b", {}, "中间关键帧"),
            ...middles.map(k => h("div.card", {},
                h("div.row", {}, h("span", {}, "第"), this.number(k, "at", { min: 1, max: w.keep - 2, after: () => this.commit(true) }), h("span", {}, `帧（${secs(k.at)}）`),
                    h("span.grow"), h("button.danger", { onclick: () => { shot.keyframes.splice(shot.keyframes.indexOf(k), 1); this.commit(true); } }, "删除")),
                this.slot(k.media, { capture: true, onpick: m => { if (m) { k.media = m; } else { shot.keyframes.splice(shot.keyframes.indexOf(k), 1); } this.commit(true); } }),
                h("label.row", {}, "适配画幅", this.fitSelect(k, "自动（裁剪）")), this.privacy(k))),
            h("button", {
                onclick: async () => {
                    const at = clamp(this.frame, 1, w.keep - 2);
                    let media = null;
                    try { media = await this.captureFrame(); } catch { media = null; }
                    if (!media) { const file = await pickFile("image/*"); if (!file) return; media = await upload(file); }
                    shot.keyframes.push({ at, media, fit: "auto", ...SHARED });
                    this.commit(true);
                },
            }, `+ 在当前帧（第 ${this.frame + 1} 帧）加关键帧`),
            h("div.dim", {}, "提示：“截取当前画面”可以把上一镜成片的最后一帧拿来做下一镜首帧，用于硬切后的人物衔接。"),
        ];
    }

    /** Whether the rewrite AI may see this image or video; a private one can be described in words instead (both optional). */
    privacy(item) {
        if (!item || !this.project.rewrite.enabled) return null;
        return h("div.field", {},
            this.check(item, "share", "发给改写 AI 看", () => this.renderInspector()),
            item.share ? null : h("textarea", {
                rows: 2, value: item.description || "", placeholder: "不发图片。可选：用文字描述它，代替图片告诉 AI",
                oninput: e => { item.description = e.target.value; this.commit(); },
            }));
    }

    /** How an image meets the canvas; "auto" is H3's own rule for that slot. */
    fitSelect(item, auto) {
        return h("select", { title: "适配画幅：裁剪填满 / 补黑边 / 拉伸变形", onchange: e => { item.fit = e.target.value; this.commit(true); } },
            ...[["auto", auto], ["crop", "裁剪填满"], ["pad", "补边（黑边）"], ["stretch", "拉伸"]].map(([value, label]) =>
                h("option", { value, selected: (item.fit || "auto") === value }, label)));
    }

    /** Upload slots for one reference set; ``offset`` is where its <Picture N> / <Video N> numbering starts. */
    refSlots(refs, offset) {
        const list = (items, options, build) => h("div.slots", {}, ...items.map(build),
            this.slot(null, { ...options, onpick: m => { if (m) { items.push(options.wrap ? options.wrap(m) : m); this.commit(true); } } }));
        return [
            h("span.dim", {}, `图片 ${refs.images.length}`),
            list(refs.images, { accept: "image/*", wrap: m => ({ media: m, fit: "auto", ...SHARED }) }, (m, i) => h("div.field", {},
                this.slot(m.media, { tag: `<Picture ${offset.images + i + 1}>`, onpick: v => { v ? (m.media = v) : refs.images.splice(i, 1); this.commit(true); } }),
                this.fitSelect(m, "原比例"), this.privacy(m))),
            h("span.dim", {}, `视频 ${refs.videos.length}`),
            list(refs.videos, { accept: "video/*", graph: ["video", "image"], wrap: m => ({ media: m, audio: false, start: 0, ...SHARED }) }, (v, i) => h("div.field", {},
                this.slot(v.media, { accept: "video/*", graph: ["video", "image"], tag: `<Video ${offset.videos + i + 1}>`,
                    onpick: m => { m ? (v.media = m) : refs.videos.splice(i, 1); this.commit(true); } }),
                this.check(v, "audio", "带原声", () => this.commit(true)), this.privacy(v),
                h("div.row", {}, "起点", this.number(v, "start", { min: 0, step: 1 / FPS, scale: FPS, after: () => this.renderStage() }), "秒",
                    this.preview(v.media)?.video ? h("button", { onclick: () => { this.stopTools(); this.trim = { clip: v, tag: `<Video ${offset.videos + i + 1}>` }; this.renderStage(); },
                        title: "在舞台下方的素材条上选择参考的片段" }, "截取") : null))),
            h("span.dim", {}, `音频 ${refs.audios.length}`),
            list(refs.audios, { accept: "audio/*,video/*", graph: ["audio", "video"] }, (m, i) => this.slot(m, { accept: "audio/*,video/*", graph: ["audio", "video"], tag: "音频",
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
            this.field("参考强度（整镜：参考图、参考视频和关键帧一起；1 = 官方默认，越低越不受约束）", this.slider(shot, "ref_strength", 0.3, 1, 0.05)),
            source ? h("div.dim", {}, "整体改写时源视频是 <Video 1>，参考视频接在它后面编号。") : null,
            shared.images || shared.videos ? h("div.dim", {}, "本镜参考的编号接在全局参考之后。") : null,
            ...this.refSlots(shot.refs, { images: shared.images, videos: shared.videos + source }),
        ];
    }

    detectorPicker(layer) {
        const picker = h("select", { onchange: e => { layer.detector = e.target.value; this.commit(); } });
        detectors().then(list => {
            picker.replaceChildren(...(list.length ? list : ["face_yolov8m.pt"]).map(name => h("option", { value: name, selected: layer.detector === name }, name)));
        });
        return picker;
    }

    /** Queue the director alone with a mask preview request: it runs SAM 3 / face detection, never H3. */
    async previewMask(layer) {
        try {
            const { output, workflow } = await app.graphToPrompt();
            const id = String(this.node.id), inputs = output[id]?.inputs;
            if (!inputs) throw new Error("找不到导演台节点。");
            inputs.project = JSON.stringify({ ...JSON.parse(inputs.project), mask_preview: { layer: layer.id } });
            const response = await api.fetchApi("/prompt", {
                method: "POST", headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ client_id: api.clientId, prompt: output, extra_data: { extra_pnginfo: { workflow } }, partial_execution_targets: [id] }),
            });
            if (!response.ok) {
                const body = await response.json().catch(() => ({}));
                throw new Error(body.error?.message || `HTTP ${response.status}`);
            }
            this.panel.say("已加入队列：识别重绘层的遮罩，完成后在“局部重绘”页显示。");
        } catch (error) {
            alert(`识别预览没能开始：${error.message}`);
        }
    }

    repaintPane() {
        const shot = this.shot, node = this.node, project = this.project;
        const linked = name => node.inputs?.some(i => i.name === name && i.link != null);
        const masks = graphInputs(node, "mask"), previews = node.properties?.yh3Masks || {}, english = node.properties?.yh3Targets || {};
        const editsSource = shot.kind === "source" && !shot.source.regen;
        const thumb = (url, label) => h("div.slot.filled", {
            style: { width: "84px", aspectRatio: `${project.width}/${project.height}`, backgroundImage: url ? `url("${url}")` : "none" } }, label);
        const cards = shot.repaints.map((layer, i) => {
            const set = (key, value) => { layer[key] = value; this.commit(true); };
            const [x, y, bw, bh] = layer.box;
            const where = h("span.dim", {}, `${Math.round(x * 100)}%,${Math.round(y * 100)}% · ${Math.round(bw * 100)}%×${Math.round(bh * 100)}%`);
            const foreign = layer.target && !/^[\x00-\x7f]*$/.test(layer.target);
            const sources = {
                paint: () => h("div.row", {}, thumb(layer.mask && mediaUrl(layer.mask), layer.mask ? "" : "无遮罩"),
                    h("button.primary", { onclick: () => this.startPaint(i) }, layer.mask ? "编辑遮罩" : "在画面上绘制遮罩")),
                input: () => (masks.length
                    ? h("div.row", {}, thumb(layer.input && this.preview({ input: layer.input })?.url, layer.input || "未选择"),
                        h("select", { onchange: e => set("input", e.target.value) }, h("option", { value: "" }, "选择遮罩输入…"),
                            ...masks.map(m => h("option", { value: m.input, selected: layer.input === m.input }, m.label))))
                    : h("div.warn", {}, "节点上还没有连接遮罩：把 MASK 接到导演台的 masks 输入。")),
                text: () => [
                    linked("sam3") && linked("sam3_clip") ? null : h("div.warn", {},
                        "导演台的 sam3 / sam3_clip 还没接：用 CheckpointLoaderSimple 加载 sam3.1_multiplex_fp16.safetensors，MODEL 接 sam3、CLIP 接 sam3_clip。"),
                    this.field("识别目标（英文短语最准，如 girl in red raincoat；多个用逗号分隔，写成 cup:2 最多取 2 个）",
                        h("input", { type: "text", value: layer.target, placeholder: "girl in red raincoat", onchange: e => set("target", e.target.value.trim()) })),
                    !foreign ? null : english[layer.target] ? h("div.dim", {}, `SAM 3 用：${english[layer.target]}`)
                        : linked("rewriter") ? h("div.dim", {}, "运行或预览时由改写模型翻成英文。")
                            : h("div.warn", {}, "SAM 3 只认英文：写英文，或接上改写模型自动翻译。"),
                    h("div.row", {}, this.field("识别阈值（低：更容易认到，也更容易认错）", this.slider(layer, "threshold", 0.1, 0.9, 0.05)),
                        this.field("最多目标（0 不限）", this.number(layer, "objects", { min: 0, max: 64 }))),
                ],
                face: () => [h("div.row", {}, h("button.primary", { onclick: () => this.startBox(i) }, "在画面上框住这张脸"), where),
                    this.field("人脸检测器（models/ultralytics/bbox）", this.detectorPicker(layer))],
                box: () => h("div.row", {}, h("button.primary", { onclick: () => this.startBox(i) }, "在画面上框选区域"), where),
            };
            const preview = previews[layer.id];
            const seen = layer.crop ? "成片（先运行一次）" : editsSource ? "源视频" : "本镜生成的画面（先运行一次）";
            return h(`div.card${this.paint === i || this.boxing === i ? ".active" : ""}`, {},
                h("header.row", {}, h("b", {}, `重绘 ${i + 1}`), this.check(layer, "enabled", "启用"), h("span.grow"),
                    h("button.danger", { onclick: () => { shot.repaints.splice(i, 1); this.commit(true); } }, "删除")),
                this.field("遮罩来源", this.seg(SOURCES, layer.source, v => set("source", v))),
                sources[layer.source](),
                h("div.row", {}, h("button", { onclick: () => this.previewMask(layer) }, "识别预览"), h("span.dim", {}, `在${seen}上看这层的遮罩`)),
                preview ? h("img.maskpreview", { src: mediaUrl(preview.image), onerror: e => e.target.remove() }) : null,
                preview ? h("div.dim", {}, `遮罩覆盖 ${Math.round(preview.coverage * 100)}% 的画面 · ${preview.frames} 帧`) : null,
                this.check(layer, "crop", "放大贴回：裁出遮罩区域、放大后重绘再贴回（适合脸、手、小物体；在二采后的成片上进行）", () => this.commit(true)),
                layer.crop ? this.field("清晰度（裁剪后放大到的像素量，MP）", this.slider(layer, "megapixels", 0.2, 1, 0.05)) : null,
                this.field(layer.crop ? "强度（放大贴回常用 0.3–0.5；越高细节越多，也越可能改样子）" : "重绘强度（1 = 遮罩内完全重新生成）",
                    this.slider(layer, "strength", 0.05, 1, 0.05)),
                this.range(layer),
                this.field("本层提示词（留空沿用镜头提示词）", h("textarea", { rows: 3, value: layer.prompt, oninput: e => { layer.prompt = e.target.value; this.commit(); } })),
                // A cropped box or face is pasted back with its own ellipse / feathered frame.
                layer.crop && boxed(layer) ? h("div.row", {}, this.reseed(layer))
                    : h("div.row", {}, this.field("遮罩扩展 px", this.number(layer, "grow", { min: 0, max: 256 })),
                        this.field("边缘羽化 px", this.number(layer, "feather", { min: 0, max: 256 })), this.reseed(layer)));
        });
        const master = widget(node, "detail"), crops = shot.repaints.some(layer => layer.crop);
        return [
            h("div.note", {}, editsSource
                ? "遮罩区域按提示词重新生成，遮罩外保持源视频原样。"
                : "只重新生成遮罩区域。不勾“放大贴回”时在首采上原位重绘，二采把它一起放大；接了 inpaint_patch（Fun ControlNet）时遮罩区更贴合周围画面。"
                    + "勾上则在成片上把遮罩区域裁出来放大重绘、再贴回。"),
            crops && master ? h("label.row", {}, h("input", { type: "checkbox", checked: !!master.value,
                onchange: e => { master.value = e.target.checked; changed(); this.project.detail.enabled = master.value; this.renderInspector(); } }),
                "放大贴回总开关（节点参数 detail）") : null,
            crops && master && !master.value ? h("div.warn", {}, "总开关已关闭：放大贴回的层运行时都会跳过。") : null,
            ...cards,
            h("button", { onclick: () => { shot.repaints.push(newLayer()); this.commit(true); this.startPaint(shot.repaints.length - 1); } }, "+ 新建重绘层"),
        ];
    }

    exportProject() {
        const blob = new Blob([JSON.stringify(projectData(this.project), null, 2)], { type: "application/json" });
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
