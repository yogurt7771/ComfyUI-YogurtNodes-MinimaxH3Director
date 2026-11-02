"""Prompt rewriting through an installed agent CLI, on its own login instead of an API key.

Each run gets a throwaway folder holding the brief and its images, and read access only; what comes back is the
agent's final message. Nothing relies on the agent writing files: a sandboxed Codex on Windows writes files the
ComfyUI user cannot read, and Grok's permission rules do not keep writes inside the folder.
"""

import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import tempfile
import time

import comfy.model_management
import folder_paths
from comfy_api.latest import io

Rewriter = io.Custom("YOGURT_H3_REWRITER")

CLI_DEFAULT = "默认"
# Fixed text: a Windows .cmd shim passes its arguments through cmd.exe, so nothing from a project goes on the command line.
GO = "Read brief.md in the current folder and carry it out exactly."
MODEL = re.compile(r"[A-Za-z0-9][\w.:/\[\]@-]*")
BLOCK = re.compile(r"=====\s*SEGMENT (\d+)\s*=====\s*\n(.*?)\n\s*=====\s*END SEGMENT \1\s*=====", re.S)


def _flag(name, value):
    return [name, value] if value else []


def _codex(model, effort):
    return ["exec", GO, "--skip-git-repo-check", "--ephemeral", "-s", "read-only", "--color", "never",
            *_flag("-m", model), *_flag("-c", effort and f"model_reasoning_effort={effort}")]


def _claude(model, effort):
    return ["-p", GO, "--no-session-persistence", "--tools", "Read", "--setting-sources", "project,local", "--strict-mcp-config",
            *_flag("--model", model), *_flag("--effort", effort)]


def _grok(model, effort):
    return ["-p", GO, "--tools", "Read", "--no-subagents", "--disable-web-search",
            *_flag("-m", model), *_flag("--reasoning-effort", effort)]


AGENTS = {"codex": _codex, "claude": _claude, "grok": _grok}
EFFORTS = {"codex": ("low", "medium", "high", "xhigh"), "claude": ("low", "medium", "high", "xhigh", "max"), "grok": ("low", "medium", "high")}


def agent_config(provider, path, model="", effort=CLI_DEFAULT, vision=True):
    if provider not in AGENTS:
        raise ValueError(f"不支持的改写 CLI：{provider}。")
    program = shutil.which(str(path or provider).strip().strip('"'))
    if program is None:
        raise FileNotFoundError(f"找不到 {provider}：先安装它，或在节点的 path 里填完整路径。")
    # Workflows get shared and this value gets executed: run only the CLI itself, never a file an upload could have placed.
    writable = (folder_paths.get_input_directory(), folder_paths.get_output_directory(), folder_paths.get_temp_directory())
    if Path(program).stem.lower() != provider or any(Path(program).resolve().is_relative_to(Path(d).resolve()) for d in writable):
        raise ValueError(f"path 必须是 {provider} 本身，不能是别的程序，也不能放在 ComfyUI 的 input、output、temp 目录里。")
    model = str(model or "").strip()
    if model and not MODEL.fullmatch(model):
        raise ValueError(f"{provider} 的模型名称不合法：{model}")
    if effort != CLI_DEFAULT and effort not in EFFORTS[provider]:
        raise ValueError(f"{provider} 的 effort 只能是 {'/'.join((CLI_DEFAULT, *EFFORTS[provider]))}。")
    return {"provider": provider, "path": program, "model": model, "effort": "" if effort == CLI_DEFAULT else effort, "vision": bool(vision)}


def _schema(node_id, name, provider, login, model_tip):
    return io.Schema(
        node_id=node_id,
        display_name=f"Yogurt H3 改写 {name}",
        category="Yogurt/MiniMax H3",
        description=f"用本机已登录的 {name} 命令行写导演台提示词，走它自己的额度，不需要 API key；先在终端运行一次 {login}。"
                    "连接到导演台的 rewriter，在工程里打开“自动改写”。",
        inputs=[
            io.String.Input("path", default=provider, tooltip=f"{name} 可执行文件：默认从 PATH 查找，也可以填完整路径。文件名必须是 {provider}。"),
            io.String.Input("model", default="", tooltip=f"留空使用 {name} 自己配置的默认模型。{model_tip}"),
            io.Combo.Input("effort", options=[CLI_DEFAULT, *EFFORTS[provider]], default="medium", tooltip=f"推理强度，越高越慢；选“{CLI_DEFAULT}”时用 {name} 自己的配置。"),
            io.Boolean.Input("vision", default=True, tooltip="把首尾帧、参考图和源视频画面放进临时目录给它看。"),
        ],
        outputs=[Rewriter.Output("rewriter")],
    )


class YogurtH3Codex(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return _schema("YogurtH3Codex", "Codex", "codex", "codex login", "codex exec 的 -m 参数。")

    @classmethod
    def execute(cls, path, model, effort, vision) -> io.NodeOutput:
        return io.NodeOutput(agent_config("codex", path, model, effort, vision))


class YogurtH3Claude(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return _schema("YogurtH3Claude", "Claude", "claude", "claude auth login", "例如 opus、sonnet。")

    @classmethod
    def execute(cls, path, model, effort, vision) -> io.NodeOutput:
        return io.NodeOutput(agent_config("claude", path, model, effort, vision))


class YogurtH3Grok(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return _schema("YogurtH3Grok", "Grok", "grok", "grok login", "grok models 可以列出可用模型。")

    @classmethod
    def execute(cls, path, model, effort, vision) -> io.NodeOutput:
        return io.NodeOutput(agent_config("grok", path, model, effort, vision))


def _stop(process):
    # A .cmd shim runs the CLI as a grandchild, so the whole tree has to go.
    if os.name == "nt":
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(process.pid)], capture_output=True)
    else:
        os.killpg(process.pid, signal.SIGKILL)
    process.wait()


def run_agent(config, brief, files=None):
    """The agent's stdout. ``files`` ({relative path: bytes}) are placed next to the brief."""
    command = [config["path"], *AGENTS[config["provider"]](config["model"], config["effort"])]
    with tempfile.TemporaryDirectory(prefix="yogurt-h3-", ignore_cleanup_errors=True) as name, tempfile.TemporaryFile() as out, tempfile.TemporaryFile() as err:
        folder = Path(name)
        for relative, data in {"brief.md": brief.encode("utf-8"), **(files or {})}.items():
            target = folder / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        process = subprocess.Popen(command, cwd=folder, stdin=subprocess.DEVNULL, stdout=out, stderr=err,
                                   creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0, start_new_session=os.name != "nt")
        try:
            while process.poll() is None:
                comfy.model_management.throw_exception_if_processing_interrupted()
                time.sleep(0.5)
        finally:
            if process.poll() is None:
                _stop(process)
        out.seek(0)
        err.seek(0)
        reply = out.read().decode("utf-8", "replace")
        if process.returncode:
            detail = err.read().decode("utf-8", "replace").strip() or reply.strip()
            raise RuntimeError(f"{config['provider']} 运行失败（退出码 {process.returncode}）：{detail[-800:]}")
        return reply


def answers(reply):
    """{segment number: text} from the delimited blocks of a final message."""
    return {int(number): body.strip() for number, body in BLOCK.findall(reply)}
