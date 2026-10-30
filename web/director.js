// Yogurt H3 导演台 node: compact timeline panel on the node, full editor in a modal.
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import {
    FPS, NODE, PHASES, STAGES, changed, clamp, effectiveRefs, features, filmSize, h, mediaLabel, mediaPreview, newShot, planWindows, readInputs, readProject,
    rewritable, secs, storePrompts, totalFrames, widget, writeProject,
} from "./core.js";
import { openEditor } from "./editor.js";

const panels = new Map();
// Backend stage -> the run phase it belongs to. Upscaling opens the next 二采 pass; segmenting stays in the current phase.
const PHASE_OF = { encoding: "encoding", sampling: "sampling", repainting: "sampling", refining: "two_pass", rerendering: "rerender",
    decoding: "decoding", detailing: "detailing", superres: "assembling", complete: "assembling" };
const REFINES = ["two_pass", "rerender"];
const clock = ms => { const s = Math.max(0, Math.round(ms / 1000)); return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`; };

class Panel {
    constructor(node) {
        this.node = node;
        this.states = {};
        this.run = null;
        this.listeners = new Set();
        this.selected = 0;
        this.meta = h("span.dim.grow");
        this.strip = h("div.strip");
        this.status = h("div.status", {}, "双击镜头或点“打开导演台”进行完整编辑。");
        this.quick = h("div.row");
        this.refs = h("div.refs", { hidden: true });
        this.story = h("textarea", {
            oninput: e => {
                const project = readProject(node);
                project.story = e.target.value;
                writeProject(node, project);
            },
        });
        this.prompt = h("textarea", {
            oninput: e => {
                const project = readProject(node);
                project.shots[this.selected].prompt = e.target.value;
                writeProject(node, project);
            },
        });
        this.promptLabel = h("span");
        this.result = h("div.result");
        this.live = h("img", { alt: "", hidden: true });
        this.phaseText = h("b");
        this.detailText = h("span.dim.grow");
        this.timeText = h("span.dim");
        this.stepText = h("span.dim");
        this.fill = h("i");
        this.runbox = h("div.runbox", { hidden: true }, this.live,
            h("div.info", {}, h("div.line", {}, this.phaseText, this.detailText, this.timeText), h("div.bar", {}, this.fill), h("div.line", {}, this.stepText)));
        this.root = h("div.yh3.yh3-panel", {},
            h("header", {}, h("strong", {}, "H3 导演台"), this.meta, h("button.primary", { onclick: () => this.open() }, "打开导演台")),
            this.strip, this.runbox, this.quick, this.refs,
            h("label.field.storyf", {}, h("span", {}, "全局提示词"), this.story),
            h("label.field", {}, this.promptLabel, this.prompt), this.result,
            h("div.row", {}, this.status, h("span.grow"),
                h("button", { onclick: () => this.addShot() }, "+ 镜头"),
                h("button.primary", { onclick: () => { app.queuePrompt(0, 1); this.say("已加入队列。"); } }, "▶ 运行")));
        for (const type of ["keydown", "keyup", "paste"]) this.root.addEventListener(type, e => e.stopPropagation());
    }

    listen(fn) {
        this.listeners.add(fn);
        return () => this.listeners.delete(fn);
    }

    notify() {
        this.listeners.forEach(fn => fn());
    }

    say(text) {
        this.status.textContent = text;
        this.status.title = text;
    }

    open(index = this.selected, tab) {
        if (!this.editor?.root?.isConnected) this.editor = openEditor(this.node, this, index, tab);
    }

    select(index) {
        this.selected = index;
        this.refresh();
    }

    addShot() {
        const project = readProject(this.node);
        const shot = newShot();
        shot.prompt = project.shots[this.selected]?.prompt || "";
        project.shots.splice(this.selected + 1, 0, shot);
        writeProject(this.node, project);
        this.select(this.selected + 1);
    }

    // ------------------------------------------------------------------ run progress

    /** A render starts with the phases it works through, each listing the shots it touches (Render.announce). */
    begin(phases) {
        clearInterval(this.timer);
        this.run = { phases: phases.filter(([, ids]) => ids.length), at: -1, shot: null, stage: null, sampled: false, step: null,
            start: Date.now(), end: null, error: null };
        this.states = {};
        this.timer = setInterval(() => this.showRun(), 1000);
        this.refresh();
    }

    /** Shot ``id`` entered backend stage ``stage``. */
    advance(id, stage) {
        const run = this.run;
        if (!run || run.end) return;
        let name = PHASE_OF[stage];
        if (stage === "upscaling") {
            // A 二采 pass upscales every shot before it samples any: an upscale opens the next pass unless this one is still upscaling.
            const current = run.phases[run.at]?.[0];
            name = REFINES.includes(current) && !run.sampled ? current : run.phases.slice(run.at + 1).find(([n]) => REFINES.includes(n))?.[0];
        }
        const at = name ? run.phases.findIndex(([n], k) => n === name && k >= run.at) : -1;
        if (at >= 0 && at !== run.at) Object.assign(run, { at, sampled: false });
        if (stage === "refining" || stage === "rerendering") run.sampled = true;
        Object.assign(run, { shot: id, stage, step: null });
    }

    finish(error = null) {
        if (!this.run || this.run.end) return;
        clearInterval(this.timer);
        Object.assign(this.run, { end: Date.now(), error });
        this.live.hidden = true;
        this.showRun();
        this.renderStrip();
    }

    /** Share of the run done: finished phases, plus the current phase up to the shot in work and its sampler steps. */
    progress() {
        const run = this.run;
        if (run.end && !run.error) return 1;
        let total = 0, done = 0;
        run.phases.forEach(([name, ids], k) => {
            const weight = PHASES[name][1];
            total += weight * ids.length;
            if (k < run.at) done += weight * ids.length;
            else if (k === run.at && !(REFINES.includes(name) && !run.sampled)) {
                const step = run.step?.max ? run.step.value / run.step.max : 0;
                done += weight * (Math.max(0, ids.indexOf(run.shot)) + step);
            }
        });
        return total ? Math.min(1, done / total) : 0;
    }

    showRun() {
        const run = this.run;
        this.runbox.hidden = !run;
        if (!run) return;
        const phase = run.phases[run.at], fraction = this.progress(), elapsed = (run.end || Date.now()) - run.start;
        const shots = readProject(this.node).shots, number = shots.findIndex(s => s.id === run.shot) + 1;
        this.runbox.classList.toggle("done", Boolean(run.end && !run.error));
        this.runbox.classList.toggle("error", Boolean(run.error));
        this.fill.style.width = `${(fraction * 100).toFixed(1)}%`;
        if (run.end) {
            this.phaseText.textContent = run.error || "完成";
            this.detailText.textContent = "";
            this.timeText.textContent = `用时 ${clock(elapsed)}`;
            this.stepText.textContent = run.error ? "" : `${shots.length} 镜，其中 ${shots.filter(s => this.cached(s.id)).length} 镜直接用缓存`;
            return;
        }
        this.phaseText.textContent = phase ? `${PHASES[phase[0]][0]} ${Math.max(1, phase[1].indexOf(run.shot) + 1)}/${phase[1].length}` : "准备中";
        this.detailText.textContent = number ? `第 ${number} 镜 · ${STAGES[run.stage] || run.stage}` : "";
        // The estimate settles once a few percent are done; before that it is mostly model loading.
        this.timeText.textContent = `已用 ${clock(elapsed)}${fraction > 0.05 ? ` · 约剩 ${clock(elapsed / fraction - elapsed)}` : ""}`;
        this.stepText.textContent = `总进度 ${Math.round(fraction * 100)}%${run.step?.max ? ` · 步 ${run.step.value}/${run.step.max}` : ""}`;
    }

    /** A shot no phase of this run touches before the film is assembled: everything it needs is cached. */
    cached(id) {
        return Boolean(this.run) && !this.run.phases.some(([name, ids]) => name !== "assembling" && ids.includes(id));
    }

    /** [data-state, label] of a shot block while a run is shown. */
    blockState(id) {
        const run = this.run;
        if (run.shot === id && !run.end) return ["running", `${STAGES[run.stage] || run.stage}${run.step?.max ? ` ${run.step.value}/${run.step.max}` : ""}`];
        if (this.cached(id)) return ["cached", "已缓存"];
        const next = run.phases.find(([, ids], k) => ids.includes(id) && (k > run.at || (k === run.at && ids.indexOf(id) > ids.indexOf(run.shot))));
        if (!next || (run.end && !run.error)) return [run.end && !run.error ? "complete" : "", run.end ? "完成" : "✓"];
        return ["waiting", `等待${PHASES[next[0]][0]}`];
    }

    // ------------------------------------------------------------------ rendering

    renderStrip() {
        const project = readProject(this.node), windows = planWindows(project);
        const prompts = this.node.properties?.yh3Prompts || {};
        const shown = shot => (rewritable(project, shot) && prompts[shot.id]) || shot.prompt;
        const running = this.run && !this.run.end;
        this.strip.replaceChildren(...windows.map(w => {
            const id = w.shot.id, idle = this.states[id];
            const [state, label] = this.run ? this.blockState(id)
                : [idle === "complete" ? "complete" : idle ? "running" : "", idle && idle !== "complete" ? STAGES[idle] || idle : null];
            const step = running && this.run.shot === id && this.run.step?.max ? this.run.step.value / this.run.step.max : null;
            return h(`button.block${w.shot.kind === "source" ? ".source" : ""}${w.index === this.selected ? ".selected" : ""}`, {
                style: { flex: `${w.keep} 1 0` }, "data-state": state,
                title: `${shown(w.shot) || "（无提示词）"}\n${secs(w.keep)}`,
                onclick: () => this.select(w.index), ondblclick: () => this.open(w.index),
            }, windows[w.index + 1]?.head ? h("i.carry", { style: { width: `${Math.min(100, 100 * windows[w.index + 1].head / w.keep)}%` } }) : null, h("i.state"),
                step != null ? h("i.fill", { style: { width: `${(step * 100).toFixed(1)}%` } }) : null,
                h("b", {}, `${w.index + 1} · ${secs(w.keep)}`),
                h("small", {}, label || features(w.shot, project).join(" ") || (w.shot.kind === "source" ? "源视频" : "生成")));
        }));
    }

    /** The selected shot's keyframes and references (global first, as H3 numbers them); click to edit them. */
    renderRefs(project, shot) {
        const refs = effectiveRefs(project, shot), node = this.node;
        // An integral source rewrite is <Video 1>; reference videos come after it.
        const videos = shot.kind === "source" && shot.source?.regen ? 1 : 0;
        const thumb = (media, tag, tab) => {
            const shown = mediaPreview(node, media);
            const el = h(`div.thumb${shown?.audio ? "" : ".picture"}`, { title: `${tag}　${mediaLabel(media)}`, onclick: () => this.open(this.selected, tab) },
                h("span.tag", {}, tag));
            if (shown?.video) el.append(h("video", { src: shown.url, muted: true, preload: "metadata" }));
            else if (shown?.url) el.style.backgroundImage = `url("${shown.url}")`;
            else el.append(shown?.audio ? `♪ ${shown.name}` : mediaLabel(media));
            return el;
        };
        const keyTag = at => (at === 0 ? "首帧" : at === -1 ? "尾帧" : `第 ${at + 1} 帧`);
        const items = [
            ...shot.keyframes.map(k => thumb(k.media, keyTag(k.at), "keys")),
            ...refs.images.map((m, i) => thumb(m.media, `<Picture ${i + 1}>`, "refs")),
            ...refs.videos.map((v, i) => thumb(v.media, `<Video ${videos + i + 1}>`, "refs")),
            ...refs.audios.map(m => thumb(m, "音频", "refs")),
        ];
        const pictures = [...shot.keyframes.map(k => k.media), ...refs.images.map(m => m.media), ...refs.videos.map(v => v.media)];
        if (pictures.some(m => m?.input && !mediaPreview(node, m))) {
            items.push(h("button", { onclick: () => readInputs(node, this), title: "只运行导演台的上游，取得连线素材的画面，不加载 H3" }, "读取连线预览"));
        }
        this.refs.hidden = !items.length;
        this.refs.replaceChildren(...items);
    }

    refresh() {
        const project = readProject(this.node);
        this.selected = clamp(this.selected, 0, project.shots.length - 1);
        const windows = planWindows(project), total = Math.max(1, totalFrames(project));
        const prompts = this.node.properties?.yh3Prompts || {};
        const film = filmSize(project), draft = [project.width, project.height];
        const size = film.join("×") === draft.join("×") ? film.join("×") : `首采 ${draft.join("×")} → ${film.join("×")}`;
        this.meta.textContent = `${size} · ${project.shots.length} 镜 · ${secs(total)}`;
        this.renderStrip();
        this.showRun();
        const shot = project.shots[this.selected], w = windows[this.selected];
        const seconds = h("input", {
            type: "number", min: 5 / FPS, step: 0.125, value: Number((shot.frames / FPS).toFixed(3)), style: { width: "72px" },
            onchange: e => {
                const p = readProject(this.node);
                p.shots[this.selected].frames = clamp(Math.round(Number(e.target.value) * FPS), 5, FPS * 600);
                writeProject(this.node, p);
                this.refresh();
            },
        });
        const carry = h("input", {
            type: "checkbox", checked: shot.continue, disabled: this.selected === 0 || shot.kind !== "generate",
            onchange: e => {
                const p = readProject(this.node);
                p.shots[this.selected].continue = e.target.checked;
                writeProject(this.node, p);
                this.refresh();
            },
        });
        this.quick.replaceChildren(h("b", {}, `第 ${this.selected + 1} 镜`), h("span.dim", {}, shot.kind === "source" ? "源视频" : "生成"),
            h("label.row", {}, "时长", seconds, "秒"), h("label.row", {}, carry, "续接上一镜"),
            h("span.dim", {}, `输出 ${w.keep} 帧`));
        this.renderRefs(project, shot);
        if (document.activeElement !== this.story) this.story.value = project.story;
        this.story.placeholder = project.rewrite.enabled ? "整部片的剧情、人物、风格和场景；改写模型据此写每一镜。"
            : "整部片的剧情、人物、风格和场景。只在打开自动改写后生效（改写模型据此写每一镜），不改写时只用各镜提示词。";
        if (document.activeElement !== this.prompt) this.prompt.value = shot.prompt;
        const auto = rewritable(project, shot);
        this.promptLabel.textContent = `第 ${this.selected + 1} 镜提示词`;
        this.prompt.placeholder = auto ? "本镜要点（自动改写开启，可留空）" : "本镜提示词";
        this.result.hidden = !auto;
        this.result.replaceChildren(h("b", {}, "改写结果　"), prompts[shot.id] || "（运行，或在导演台“剧本”页点“生成剧本”后显示）");
        this.notify();
    }
}

function setup(node) {
    const data = widget(node, "project");
    if (data) {
        data.hidden = true;
        data.computeSize = () => [0, -4];
        data.computeLayoutSize = () => ({ minHeight: 0, maxHeight: 0, minWidth: 0 });
    }
    const panel = new Panel(node);
    panels.set(node.id, panel);
    // The timeline drawn on the panel depends on these node widgets.
    for (const name of ["width", "height", "overlap", "two_pass", "two_pass_scale", "rerender", "rerender_scale", "rtx_vsr", "rtx_vsr_scale"]) {
        const w = widget(node, name), callback = w?.callback;
        if (w) w.callback = function () { const result = callback?.apply(this, arguments); panel.refresh(); return result; };
    }
    // Room for the run progress, references, both prompts and the rewrite result at their minimum heights.
    const dom = node.addDOMWidget("director", "yh3director", panel.root, { serialize: false, hideOnZoom: false, getMinHeight: () => 530 });
    dom.serialize = false;
    dom.computeLayoutSize = () => ({ minHeight: 530, minWidth: 460 });
    node.yh3Panel = panel;
    node.setSize([Math.max(node.size[0], 520), Math.max(node.size[1], 740)]);
    panel.refresh();
}

function panelFor(id) {
    return panels.get(Number(id)) ?? panels.get(String(id)) ?? app.graph?.getNodeById?.(Number(id))?.yh3Panel;
}

api.addEventListener("yogurt_h3.director", ({ detail }) => {
    const panel = panelFor(detail.node);
    if (!panel) return;
    if (detail.stage === "plan") {
        panel.begin(detail.phases);
        return;
    }
    panel.states[detail.shot] = detail.stage;
    panel.advance(detail.shot, detail.stage);
    const index = readProject(panel.node).shots.findIndex(s => s.id === detail.shot);
    panel.say(`第 ${index + 1} 镜：${STAGES[detail.stage] || detail.stage}`);
    if (detail.prompt != null) {
        storePrompts(panel.node, { [detail.shot]: detail.prompt });
        if (panel.editor?.root?.isConnected) panel.editor.showPrompts();
    }
    panel.refresh();
});

// Sampler steps of the shot in work (ComfyUI's own progress for the director node).
api.addEventListener("progress", ({ detail }) => {
    const panel = panelFor(detail?.node);
    if (!panel?.run || panel.run.end) return;
    panel.run.step = { value: detail.value, max: detail.max };
    panel.showRun();
    panel.renderStrip();
});

// The sampler's live preview, when ComfyUI's preview method is on.
api.addEventListener("b_preview_with_metadata", ({ detail }) => {
    const panel = panelFor(detail?.displayNodeId ?? detail?.nodeId);
    if (!panel?.run || panel.run.end) return;
    if (panel.liveUrl) URL.revokeObjectURL(panel.liveUrl);
    panel.liveUrl = URL.createObjectURL(detail.blob);
    panel.live.src = panel.liveUrl;
    panel.live.hidden = false;
});

api.addEventListener("execution_start", () => {
    for (const panel of panels.values()) {
        panel.states = {};
        clearInterval(panel.timer);
        panel.run = null;
        panel.live.hidden = true;
        panel.showRun();
    }
});

api.addEventListener("execution_error", ({ detail }) => {
    const panel = panelFor(detail?.node_id);
    if (!panel) return;
    panel.say(`出错：${detail.exception_message || "执行失败"}`);
    panel.finish(`出错：${detail.exception_message || "执行失败"}`);
});

api.addEventListener("execution_interrupted", () => {
    for (const panel of panels.values()) panel.finish("已中断");
});

app.registerExtension({
    name: "Yogurt.MiniMaxH3Director",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE) return;
        const created = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = created?.apply(this, arguments);
            setup(this);
            return result;
        };
        const configured = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const result = configured?.apply(this, arguments);
            this.yh3Panel?.refresh();
            return result;
        };
        const removed = nodeType.prototype.onRemoved;
        nodeType.prototype.onRemoved = function () {
            clearInterval(panels.get(this.id)?.timer);
            panels.delete(this.id);
            return removed?.apply(this, arguments);
        };
        const executed = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            const result = executed?.apply(this, arguments);
            const panel = this.yh3Panel;
            this.properties = this.properties || {};
            const mask = message?.yogurt_h3_mask?.[0];
            if (mask) {
                // A 识别预览 run: one 局部重绘 layer's mask over its frames and the English its text target became.
                this.properties.yh3Masks = { ...(this.properties.yh3Masks || {}), [mask.layer]: mask };
                this.properties.yh3Targets = { ...(this.properties.yh3Targets || {}), ...mask.targets };
                changed();
                if (panel) {
                    // The preview's progress marked its shot as busy; only a finished render keeps a mark.
                    for (const [id, state] of Object.entries(panel.states)) if (state !== "complete") delete panel.states[id];
                    panel.say("识别预览完成，结果在“局部重绘”页。");
                    panel.refresh();
                    if (panel.editor?.root?.isConnected && panel.editor.tab === "repaint") panel.editor.renderInspector();
                }
            }
            const inputs = message?.yogurt_h3_inputs?.[0];
            if (inputs) {
                // A 读取连线素材 run: pictures of the connected sockets.
                this.properties.yh3Inputs = { ...(this.properties.yh3Inputs || {}), ...inputs };
                changed();
                if (panel) {
                    panel.say(Object.keys(inputs).length ? "已读取连线素材。" : "没有可预览的连线素材。");
                    panel.refresh();
                    if (panel.editor?.root?.isConnected) panel.editor.renderInspector();
                }
            }
            const info = message?.yogurt_h3?.[0];
            if (info) {
                this.properties.yh3Result = info;
                if (info.prompts && Object.keys(info.prompts).length) storePrompts(this, info.prompts);
                // Pictures of the connected sockets this run received, for material the editor cannot open itself.
                this.properties.yh3Inputs = { ...(this.properties.yh3Inputs || {}), ...(info.inputs || {}) };
                this.properties.yh3Targets = { ...(this.properties.yh3Targets || {}), ...(info.targets || {}) };
                changed();
                if (panel) {
                    info.timeline.forEach(entry => { panel.states[entry.id] = "complete"; });
                    panel.finish();
                    panel.say(`完成：${info.video.filename}（${info.size[0]}×${info.size[1]}，${secs(info.timeline.reduce((s, e) => s + e.frames, 0))}）`);
                    panel.refresh();
                    if (panel.editor?.root?.isConnected) panel.editor.showPrompts();
                }
            }
            return result;
        };
    },
});
