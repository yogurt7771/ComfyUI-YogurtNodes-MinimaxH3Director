// Yogurt H3 导演台 node: compact timeline panel on the node, full editor in a modal.
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { FPS, NODE, STAGES, changed, clamp, features, h, newShot, planWindows, readProject, rewritable, secs, storePrompts, totalFrames, widget, writeProject } from "./core.js";
import { openEditor } from "./editor.js";

const panels = new Map();

class Panel {
    constructor(node) {
        this.node = node;
        this.states = {};
        this.listeners = new Set();
        this.selected = 0;
        this.meta = h("span.dim.grow");
        this.strip = h("div.strip");
        this.status = h("div.status", {}, "双击镜头或点“打开导演台”进行完整编辑。");
        this.quick = h("div.row");
        this.prompt = h("textarea", {
            placeholder: "本镜提示词", oninput: e => {
                const project = readProject(node);
                project.shots[this.selected].prompt = e.target.value;
                writeProject(node, project);
            },
        });
        this.result = h("div.result");
        this.root = h("div.yh3.yh3-panel", {},
            h("header", {}, h("strong", {}, "H3 导演台"), this.meta, h("button.primary", { onclick: () => this.open() }, "打开导演台")),
            this.strip, this.quick, this.prompt, this.result,
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

    open(index = this.selected) {
        if (!this.editor?.root?.isConnected) this.editor = openEditor(this.node, this, index);
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

    refresh() {
        const project = readProject(this.node);
        this.selected = clamp(this.selected, 0, project.shots.length - 1);
        const windows = planWindows(project), total = Math.max(1, totalFrames(project));
        const prompts = this.node.properties?.yh3Prompts || {};
        const shown = shot => (rewritable(project, shot) && prompts[shot.id]) || shot.prompt;
        this.meta.textContent = `${project.width}×${project.height} · ${project.shots.length} 镜 · ${secs(total)}`;
        this.strip.replaceChildren(...windows.map(w => {
            const state = this.states[w.shot.id];
            return h(`button.block${w.shot.kind === "source" ? ".source" : ""}${w.index === this.selected ? ".selected" : ""}`, {
                style: { flex: `${w.keep} 1 0` }, "data-state": state === "complete" ? "complete" : state ? "running" : "",
                title: `${shown(w.shot) || "（无提示词）"}\n${secs(w.keep)}`,
                onclick: () => this.select(w.index), ondblclick: () => this.open(w.index),
            }, windows[w.index + 1]?.head ? h("i.carry", { style: { width: `${Math.min(100, 100 * windows[w.index + 1].head / w.keep)}%` } }) : null, h("i.state"),
                h("b", {}, `${w.index + 1} · ${secs(w.keep)}`),
                h("small", {}, state && state !== "complete" ? STAGES[state] || state : features(w.shot, project).join(" ") || (w.shot.kind === "source" ? "源视频" : "生成")));
        }));
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
        if (document.activeElement !== this.prompt) this.prompt.value = shot.prompt;
        const auto = rewritable(project, shot);
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
    for (const name of ["width", "height", "overlap", "two_pass", "two_pass_scale", "rerender", "rerender_scale"]) {
        const w = widget(node, name), callback = w?.callback;
        if (w) w.callback = function () { const result = callback?.apply(this, arguments); panel.refresh(); return result; };
    }
    const dom = node.addDOMWidget("director", "yh3director", panel.root, { serialize: false, hideOnZoom: false, getMinHeight: () => 320 });
    dom.serialize = false;
    dom.computeLayoutSize = () => ({ minHeight: 320, minWidth: 460 });
    node.yh3Panel = panel;
    node.setSize([Math.max(node.size[0], 520), Math.max(node.size[1], 620)]);
    panel.refresh();
}

function panelFor(id) {
    return panels.get(Number(id)) ?? panels.get(String(id)) ?? app.graph?.getNodeById?.(Number(id))?.yh3Panel;
}

api.addEventListener("yogurt_h3.director", ({ detail }) => {
    const panel = panelFor(detail.node);
    if (!panel) return;
    panel.states[detail.shot] = detail.stage;
    const index = readProject(panel.node).shots.findIndex(s => s.id === detail.shot);
    panel.say(`第 ${index + 1} 镜：${STAGES[detail.stage] || detail.stage}`);
    if (detail.prompt != null) {
        storePrompts(panel.node, { [detail.shot]: detail.prompt });
        if (panel.editor?.root?.isConnected) panel.editor.showPrompts();
    }
    panel.refresh();
});

api.addEventListener("execution_start", () => {
    for (const panel of panels.values()) panel.states = {};
});

api.addEventListener("execution_error", ({ detail }) => {
    const panel = panelFor(detail?.node_id);
    if (panel) panel.say(`出错：${detail.exception_message || "执行失败"}`);
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
            panels.delete(this.id);
            return removed?.apply(this, arguments);
        };
        const executed = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            const result = executed?.apply(this, arguments);
            const info = message?.yogurt_h3?.[0];
            if (info) {
                this.properties = this.properties || {};
                this.properties.yh3Result = info;
                if (info.prompts && Object.keys(info.prompts).length) storePrompts(this, info.prompts);
                // Pictures of the connected sockets this run received, for material the editor cannot open itself.
                this.properties.yh3Inputs = info.inputs || {};
                changed();
                const panel = this.yh3Panel;
                if (panel) {
                    info.timeline.forEach(entry => { panel.states[entry.id] = "complete"; });
                    panel.say(`完成：${info.video.filename}（${info.size[0]}×${info.size[1]}，${secs(info.timeline.reduce((s, e) => s + e.frames, 0))}）`);
                    panel.refresh();
                    if (panel.editor?.root?.isConnected) panel.editor.showPrompts();
                }
            }
            return result;
        };
    },
});
