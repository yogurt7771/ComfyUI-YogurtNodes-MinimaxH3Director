"""Standalone browser fixture for the actual Director widget; no Comfy server or models."""

from fractions import Fraction
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import io
import json
from pathlib import Path
from urllib.parse import urlsplit

import av
import numpy as np


DIRECTORY = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("harness_planning", DIRECTORY.parent / "planning.py")
PLANNING = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PLANNING)

HTML = """<!doctype html><html lang="zh-CN"><meta charset="UTF-8"><title>Yogurt H3 独立界面预览</title>
<style>body{margin:0;background:#171e26;color:#edf2f6;font:14px 'Segoe UI',sans-serif}main{max-width:1100px;margin:24px auto;padding:0 20px}h1{font-size:22px}p{color:#b7c4cf}#editor{width:100%;height:760px}aside{padding:14px;background:#242e39;margin-top:16px}pre{white-space:pre-wrap;max-height:240px;overflow:auto;font-size:12px}button{padding:8px;margin:5px;color:white;background:#36534f;border:1px solid #8cbdaf;border-radius:3px}</style>
<main><h1>Yogurt H3 · 独立界面预览</h1><p>真实导演台组件，模拟视频与状态；未启动 ComfyUI 服务，未加载模型。</p>
<div><button id="simulate">模拟各段运行状态</button><button id="fail">模拟中断</button></div>
<div id="editor"></div><aside><strong>测试观察：保存的时间线与重做标记</strong><pre id="inspection"></pre></aside></main>
<script type="module">import {app} from '/scripts/app.js';import {api} from '/scripts/api.js';await import('/director.js');
document.querySelector('#simulate').onclick=()=>{['complete','cached','running','pending','pending','pending'].forEach((state,i)=>api.emit('yogurt_h3_segment',{node:4,segment:i+1,state,key:'demo'}));};
document.querySelector('#fail').onclick=()=>api.emit('execution_interrupted',{});
</script></html>"""

APP_JS = """const all=[];const widgets=values=>Object.entries(values).map(([name,value])=>({name,value,callback(){app.graph.setDirtyCanvas()}}));
class Planner{constructor(){this.id=1;this.comfyClass='YogurtH3CharacterReplacePlanner';this.size=[740,940];this.inputs=[{name:'video',link:1},{name:'segment_seconds',link:2}];this.widgets=widgets({width:864,height:480,overlap_frames:22,scene_detection:false,scene_threshold:0.28,timeline_json:'{}'});}addDOMWidget(name,type,element){document.querySelector('#editor').append(element);return {};}setSize(size){this.size=size;}}
export const app={graph:{_nodes:all,links:{1:{origin_id:2},2:{origin_id:3},3:{origin_id:1}},getNodeById(id){return all.find(n=>String(n.id)===String(id))},setDirtyCanvas(){const p=all.find(n=>n.id===1),s=all.find(n=>n.id===4);if(p)document.querySelector('#inspection').textContent='rerun_segments: '+s.widgets[0].value+'\\n'+JSON.stringify(JSON.parse(p.widgets.find(w=>w.name==='timeline_json').value),null,2);}},async registerExtension(extension){await extension.beforeRegisterNodeDef(Planner,{name:'YogurtH3CharacterReplacePlanner'});const planner=new Planner();all.push(planner,{id:2,widgets:widgets({file:'demo.mp4'})},{id:3,widgets:widgets({value:6})},{id:4,comfyClass:'YogurtH3CharacterReplaceSampler',inputs:[{name:'plan',link:3}],widgets:widgets({rerun_segments:''})});planner.onNodeCreated();extension.setup();window.harness={app,planner};document.querySelector('[data-action=refresh]').click();}};
"""

API_JS = """const events=new EventTarget();export const api={fetchApi:(url,opts)=>fetch(url,opts),apiURL:url=>url,addEventListener:(name,fn)=>events.addEventListener(name,fn),emit:(name,detail)=>events.dispatchEvent(new CustomEvent(name,{detail}))};"""


def demo_video():
    output = io.BytesIO()
    with av.open(output, "w", format="mp4") as container:
        stream = container.add_stream("h264", rate=24)
        stream.width, stream.height = 128, 72
        stream.pix_fmt = "yuv420p"
        for index in range(740):
            pixels = np.full((72, 128, 3), (40, 49, 59), dtype=np.uint8)
            x = index % 116
            pixels[16:56, x:x+12] = (125, 207, 194)
            frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
            frame.pts = index
            frame.time_base = Fraction(1, 24)
            container.mux(stream.encode(frame))
        container.mux(stream.encode(None))
    return output.getvalue()


VIDEO = demo_video()


class Handler(BaseHTTPRequestHandler):
    def reply(self, content, content_type):
        content = content.encode() if isinstance(content, str) else content
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(content)

    def do_GET(self):
        path = urlsplit(self.path).path
        if path == "/":
            self.reply(HTML, "text/html; charset=utf-8")
        elif path == "/scripts/app.js":
            self.reply(APP_JS, "text/javascript; charset=utf-8")
        elif path == "/scripts/api.js":
            self.reply(API_JS, "text/javascript; charset=utf-8")
        elif path == "/director.js":
            self.reply((DIRECTORY.parent / "web" / "director.js").read_bytes(), "text/javascript; charset=utf-8")
        elif path == "/view" or path.startswith("/yogurt_h3/preview/"):
            self.reply(VIDEO, "video/mp4")
        else:
            self.send_error(404)

    def do_POST(self):
        if urlsplit(self.path).path != "/yogurt_h3/plan":
            self.send_error(404)
            return
        data = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        timeline = json.loads(data.get("timeline_json", "{}"))
        cuts = timeline.get("cuts", [])
        segments = PLANNING.timeline_segments(740, data.get("segment_seconds", 6), data.get("overlap_frames", 22),
                                              cuts=cuts, edits=timeline.get("segments", []))
        plan = dict(target_frames=740, fps=24, source_duration=740/24, source_stamp="ui-harness-source",
                    cuts=cuts, segments=segments, width=864, height=480)
        self.reply(json.dumps(plan), "application/json")

    def log_message(self, *args):
        pass


if __name__ == "__main__":
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    print(f"http://127.0.0.1:{server.server_port}/", flush=True)
    server.serve_forever()
