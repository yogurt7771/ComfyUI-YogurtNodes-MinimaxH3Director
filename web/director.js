import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const style = document.createElement("style");
style.textContent = `
.yh3 { --panel:#28313b; --ink:#ecf0f3; --muted:#aebbc6; --guide:#7dcfc2; --overlap:#e7b96b; --error:#f08080;
  color:var(--ink); background:var(--panel); padding:12px; box-sizing:border-box; font:12px/1.45 "Segoe UI",sans-serif; overflow:auto; height:100%; }
.yh3 * { box-sizing:border-box; } .yh3 header { display:flex; align-items:baseline; justify-content:space-between; gap:12px; }
.yh3 header strong { font:600 18px "Bahnschrift","Segoe UI",sans-serif; letter-spacing:.04em; }
.yh3 small,.yh3 .hint { color:var(--muted); } .yh3 .time { font:12px Consolas,monospace; }
.yh3 button,.yh3 input,.yh3 textarea { color:var(--ink); background:#202730; border:1px solid #566371; border-radius:3px; padding:6px 8px; font:inherit; }
.yh3 button { cursor:pointer; } .yh3 button:hover { border-color:var(--guide); } .yh3 :focus-visible { outline:2px solid var(--guide); outline-offset:2px; }
.yh3 .tools,.yh3 .fields { display:flex; flex-wrap:wrap; gap:8px; margin:9px 0; align-items:center; }
.yh3 .tools .primary { background:#345750; border-color:var(--guide); } .yh3 video { width:100%; height:180px; background:#171e26; object-fit:contain; }
.yh3 .rail { display:flex; gap:2px; height:64px; overflow-x:auto; margin:12px 0; padding:3px 0; }
.yh3 .segment { flex-shrink:0; min-width:58px; position:relative; padding:5px; border:1px solid #71808e; background:#344c5a; text-align:left; overflow:hidden; }
.yh3 .segment.active { border:2px solid var(--guide); } .yh3 .segment i { position:absolute; left:0; top:0; bottom:0; background:repeating-linear-gradient(135deg,#e7b96b44 0 4px,#e7b96b11 4px 8px); pointer-events:none; }
.yh3 .segment span { position:relative; display:block; white-space:nowrap; } .yh3 .segment[data-state=running] { border-color:var(--overlap); }
.yh3 .segment .edge { position:absolute; right:0; top:0; bottom:0; width:9px; background:#aebbc62a; cursor:ew-resize; touch-action:none; }
.yh3 .segment[data-state=failed] { border-color:var(--error); } .yh3 .segment[data-state=cached],.yh3 .segment[data-state=complete] { background:#345750; }
.yh3 label { display:flex; gap:5px; align-items:center; } .yh3 input[type=number] { width:76px; }
.yh3 textarea { width:100%; min-height:110px; resize:vertical; } .yh3 .target { flex:1; min-width:180px; }
.yh3 .message { min-height:18px; color:var(--overlap); white-space:pre-wrap; } .yh3 .error { color:var(--error); }
.yh3 .divider { border-top:1px solid #566371; margin-top:10px; padding-top:10px; }
`;
document.head.append(style);

const widget = (node, name) => node.widgets?.find(w => w.name === name);
const read = (node, name, fallback) => widget(node, name)?.value ?? fallback;
const time = frame => `${(frame / 24).toFixed(3)}s`;
const labels = { pending:"待生成", running:"生成中", cached:"读取缓存", complete:"已完成", failed:"失败", interrupted:"已中断" };

function inputNode(node, name) {
    const slot = node.inputs?.find(x => x.name === name);
    const link = slot?.link != null ? app.graph.links[slot.link] : null;
    return link ? app.graph.getNodeById(link.origin_id) : null;
}

function createEditor(node) {
    const root = document.createElement("section");
    root.className = "yh3";
    root.innerHTML = `<header><strong>H3 / 分段导演台</strong><small>24 fps · 斜纹为重叠续接</small></header>
      <div class="tools"><button class="primary" data-action="refresh">刷新计划</button><button data-action="reset">重建时间线</button>
      <button data-action="export">导出时间线</button><button data-action="import">导入时间线</button><input type="file" accept=".json" hidden></div>
      <video controls preload="metadata" aria-label="源视频预览"></video>
      <div class="rail" role="list" aria-label="分段时间线"></div><div class="time summary">连接 Load Video 与每段秒数，点击刷新计划。</div>
      <div class="tools"><button data-action="cut">在播放位置添加镜头切点</button><button data-action="uncut">移除最近切点</button></div>
      <div class="fields divider"><label>窗口帧数 <input name="window_frames" type="number" min="0" value="0" title="24帧为1秒；0继承全局。修改后刷新计划。"></label>
      <label>角色索引 <input name="role_index" type="number" min="0" value="0"></label>
      <label>种子偏移 <input name="seed_offset" type="number" value="0"></label><label><input name="bypass" type="checkbox">保留原片</label>
      <button data-action="rerun">标记本段重做</button><button data-action="preview">预览本段结果</button></div>
      <label>目标人物 <input class="target" name="target_description" placeholder="例如：画面左侧穿白衣的人；留空为主体人物"></label>
      <p class="hint">本段提示词：留空继承全局；填写则替换整段提示词。角色 0 为主参考图，1 起对应 additional_roles。</p>
      <textarea name="prompt" aria-label="本段提示词" placeholder="继承全局 H3 提示词"></textarea>
      <div class="message" role="status" aria-live="polite"></div>`;
    let plan = null, selected = 0;
    const states = new Map(), previews = new Map();
    const video = root.querySelector("video"), rail = root.querySelector(".rail"), msg = root.querySelector(".message");
    const fieldNames = ["window_frames", "role_index", "seed_offset", "bypass", "target_description", "prompt"];
    const say = (text, error = false) => { msg.textContent = text; msg.classList.toggle("error", error); };
    function writeTimeline(value) {
        const w = widget(node, "timeline_json");
        if (w) { w.value = JSON.stringify(value); w.callback?.(w.value); }
        app.graph.setDirtyCanvas(true, true);
    }
    function saved() { return JSON.parse(read(node, "timeline_json", "{}") || "{}"); }
    function saveEdits() {
        if (!plan) return;
        const previous = saved();
        const edits = new Map((previous.segments || []).filter(s => fieldNames.some(name => !!s[name])).map(s => [s.start_frame,s]));
        for (const s of plan.segments) {
            const edit = { start_frame:s.start_frame };
            for (const name of fieldNames) if (s[name]) edit[name] = s[name];
            if (Object.keys(edit).length > 1) edits.set(s.start_frame,edit);
            else edits.delete(s.start_frame);
        }
        writeTimeline({ ...previous, version:1, source_stamp:plan.source_stamp, segments:[...edits.values()] });
    }
    function choose(index, seek = true) {
        if (!plan?.segments.length) return;
        selected = Math.max(0, Math.min(index, plan.segments.length-1));
        const segment = plan.segments[selected];
        for (const name of fieldNames) {
            const field = root.querySelector(`[name="${name}"]`);
            if (field.type === "checkbox") field.checked = !!segment[name];
            else field.value = segment[name] ?? (field.type === "number" ? 0 : "");
        }
        for (const [i, button] of [...rail.children].entries()) button.classList.toggle("active", i === selected);
        root.querySelector(".summary").textContent = `段 ${selected+1}/${plan.segments.length} · 源片 ${time(segment.start_frame)}–${time(segment.start_frame+segment.content_frames)} · 新增 ${segment.output_frames} 帧 · 重叠 ${segment.overlap_frames} 帧 · 生成 ${segment.generation_frames} 帧`;
        if (seek && video.dataset.source) { video.src = video.dataset.source; video.currentTime = segment.output_start_frame/24; }
    }
    function render(newPlan) {
        if (plan) {
            const changedSource = plan.source_stamp !== newPlan.source_stamp || plan.width !== newPlan.width || plan.height !== newPlan.height;
            for (const number of new Set([...states.keys(), ...previews.keys()])) {
                if (changedSource || JSON.stringify(plan.segments[number-1]) !== JSON.stringify(newPlan.segments[number-1])) {
                    states.delete(number); previews.delete(number);
                }
            }
        }
        plan = newPlan;
        rail.replaceChildren();
        for (const [index, segment] of plan.segments.entries()) {
            const button = document.createElement("button");
            button.className = "segment";
            button.style.width = `${Math.max(70, segment.content_frames*0.8)}px`;
            button.dataset.state = states.get(index+1) || "pending";
            button.title = `第 ${index+1} 段：${time(segment.start_frame)}–${time(segment.start_frame+segment.content_frames)}`;
            const stripe = document.createElement("i");
            stripe.style.width = `${100*segment.overlap_frames/segment.content_frames}%`;
            const title = document.createElement("span"), status = document.createElement("span");
            title.textContent = `${String(index+1).padStart(2,"0")} / ${time(segment.output_start_frame)}`;
            status.textContent = labels[button.dataset.state]; status.className = "state";
            const edge = document.createElement("span"); edge.className="edge"; edge.title="拖动调整此段长度；也可使用窗口帧数输入框";
            edge.onpointerdown = event => {
                event.preventDefault(); event.stopPropagation(); edge.setPointerCapture(event.pointerId);
                const origin=event.clientX, initial=segment.content_frames;
                edge.onpointermove = move => { button.style.width=`${Math.max(58,(initial+(move.clientX-origin)/0.8)*0.8)}px`; };
                edge.onpointerup = async up => {
                    edge.onpointermove=null; edge.onpointerup=null;
                    segment.window_frames=Math.max(1,initial+Math.round((up.clientX-origin)/0.8));
                    selected=index; saveEdits();
                    try { await refresh(); } catch(error) { say(error.message,true); }
                };
                edge.onpointercancel = () => { edge.onpointermove=null; edge.onpointerup=null; render(plan); };
            };
            edge.onclick = event => event.stopPropagation();
            button.append(stripe, title, status, edge); button.onclick = () => choose(index);
            rail.append(button);
        }
        choose(selected, false);
    }
    async function refresh(reset = false) {
        const source = inputNode(node, "video");
        const filename = read(source ?? {}, "file", "");
        if (!filename) throw new Error("预览需要直接连接 Load Video。其他 VIDEO 节点仍可正常执行，运行后显示分段计划。");
        if (reset) {
            if (!window.confirm("清除当前时间线切点和逐段编辑？视频与缓存不会删除。")) return;
            writeTimeline({}); states.clear(); previews.clear();
        }
        const secondsNode = inputNode(node, "segment_seconds");
        const body = { file:filename, segment_seconds:read(secondsNode ?? {}, "value", read(node,"segment_seconds",6)) };
        for (const [name, fallback] of [["megapixels",0.41],["overlap_frames",22],["scene_detection",false],["scene_threshold",0.28],["timeline_json","{}"]]) body[name] = read(node,name,fallback);
        say("正在读取视频并计算时间线…");
        const response = await api.fetchApi("/yogurt_h3/plan", { method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify(body) });
        if (!response.ok) throw new Error(await response.text());
        const newPlan = await response.json();
        let cleanName = filename, type = "input";
        const suffix = filename.match(/ \[(input|output|temp)\]$/);
        if (suffix) { type = suffix[1]; cleanName = filename.slice(0,-suffix[0].length); }
        const url = api.apiURL(`/view?${new URLSearchParams({filename:cleanName,type})}`);
        video.src = url; video.dataset.source = url;
        render(newPlan); saveEdits();
        const unmatched = (saved().segments || []).filter(s => !plan.segments.some(p => p.start_frame === s.start_frame));
        say(`${plan.segments.length} 段 · ${time(plan.target_frames)} · ${plan.cuts.length} 个镜头切点。` +
            (unmatched.length ? ` ${unmatched.length} 项旧编辑不匹配当前段起点，已保留在导出 JSON 中，不会自动套到其他段。` : "修改后正常点击 ComfyUI 运行。"));
    }
    for (const name of fieldNames) root.querySelector(`[name="${name}"]`).addEventListener("change", event => {
        if (!plan) return;
        const field = event.target;
        plan.segments[selected][name] = field.type === "checkbox" ? field.checked : field.type === "number" ? Number(field.value) : field.value;
        for (let number=selected+1; number<=plan.segments.length; number++) { states.delete(number); previews.delete(number); }
        saveEdits(); render(plan);
    });
    root.addEventListener("click", async event => {
        const action = event.target.dataset.action;
        if (!action) return;
        try {
            if (action === "refresh" || action === "reset") await refresh(action === "reset");
            else if (action === "export") {
                const url = URL.createObjectURL(new Blob([JSON.stringify(saved(),null,2)],{type:"application/json"}));
                const a = document.createElement("a"); a.href=url; a.download="yogurt-h3-timeline.json"; a.click(); URL.revokeObjectURL(url);
            } else if (action === "import") root.querySelector('input[type="file"]').click();
            else if (action === "cut" || action === "uncut") {
                if (!plan) throw new Error("请先刷新时间线。");
                const edit = saved(), at = Math.round(video.currentTime*24), cuts = [...(edit.cuts || [])];
                if (action === "cut") { if (at>0 && at<plan.target_frames) cuts.push(at); }
                else if (cuts.length) cuts.splice(cuts.indexOf(cuts.reduce((a,b) => Math.abs(a-at)<Math.abs(b-at)?a:b)),1);
                edit.cuts = [...new Set(cuts)].sort((a,b)=>a-b); writeTimeline(edit); await refresh();
            } else if (action === "rerun") {
                const sampler = app.graph._nodes.find(n => n.comfyClass === "YogurtH3LongVideoSampler" && inputNode(n,"plan")?.id === node.id);
                const w = widget(sampler ?? {}, "rerun_segments");
                if (!w) throw new Error("没有找到连接此计划的 H3 长视频采样节点。");
                const items = new Set(String(w.value || "").split(",").filter(Boolean)); items.add(String(selected+1)); w.value=[...items].join(",");
                w.callback?.(w.value); say(`已标记第 ${selected+1} 段。点击运行后重做；完成后清空 rerun_segments 可恢复全部缓存复用。`);
            } else if (action === "preview") {
                const key = previews.get(selected+1);
                if (!key) throw new Error("本段尚无此次会话的结果。运行后可预览；续跑也会恢复缓存预览入口。");
                video.src = api.apiURL(`/yogurt_h3/preview/${key}?v=${Date.now()}`); video.load(); say("显示本段生成结果（无声预览，包含重叠头部）。");
            }
        } catch (error) { say(error.message, true); }
    });
    root.querySelector('input[type="file"]').onchange = async event => {
        try {
            const file = event.target.files[0]; if (!file) return;
            const data = JSON.parse(await file.text());
            if (data.version !== 1 || !Array.isArray(data.segments)) throw new Error("不是有效的 H3 时间线文件。");
            writeTimeline(data); await refresh();
        } catch (error) { say(error.message, true); }
    };
    node.yh3Plan = render;
    node.yh3State = (number, state, key) => {
        states.set(number,state); if (key) previews.set(number,key);
        const button = rail.children[number-1];
        if (button) { button.dataset.state=state; button.querySelector(".state").textContent=labels[state] || state; }
    };
    node.yh3Fail = state => { for (const [number,value] of states) if (value === "running") node.yh3State(number,state); };
    const dom = node.addDOMWidget("director_timeline", "div", root, { serialize:false });
    dom.computeSize = width => [width,720];
    const data = widget(node,"timeline_json");
    if (data) { data.type="hidden"; data.computeSize=()=>[0,-4]; }
    node.setSize([Math.max(node.size[0],740), Math.max(node.size[1],940)]);
}

app.registerExtension({
    name:"Yogurt.MiniMaxH3Director.Timeline",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "YogurtH3CharacterReplacePlanner") return;
        const created=nodeType.prototype.onNodeCreated, executed=nodeType.prototype.onExecuted;
        nodeType.prototype.onNodeCreated=function() { created?.apply(this,arguments); createEditor(this); };
        nodeType.prototype.onExecuted=function(message) { executed?.apply(this,arguments); if (message.yogurt_h3_plan?.[0]) this.yh3Plan?.(message.yogurt_h3_plan[0]); };
    },
    setup() {
        api.addEventListener("yogurt_h3_segment", ({detail}) => {
            const sampler=app.graph.getNodeById(detail.node);
            inputNode(sampler ?? {},"plan")?.yh3State?.(detail.segment,detail.state,detail.key);
        });
        for (const event of ["execution_error","execution_interrupted"]) api.addEventListener(event,() => {
            for (const node of app.graph._nodes) node.yh3Fail?.(event === "execution_error" ? "failed" : "interrupted");
        });
    },
});
