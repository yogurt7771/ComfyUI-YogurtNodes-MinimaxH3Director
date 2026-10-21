"""Yogurt MiniMax H3 Director: a visual timeline director that renders finished MiniMax H3 videos."""

import asyncio

from aiohttp import web
from server import PromptServer

from .detail import detectors
from .director import YogurtH3Director
from .media import Media
from .project import load_project
from .rewrite import YogurtH3RewriteModel, rewrite_project, rewriter_config

WEB_DIRECTORY = "./web"

NODE_CLASS_MAPPINGS = {"YogurtH3Director": YogurtH3Director, "YogurtH3RewriteModel": YogurtH3RewriteModel}
NODE_DISPLAY_NAME_MAPPINGS = {"YogurtH3Director": "Yogurt H3 导演台", "YogurtH3RewriteModel": "Yogurt H3 改写模型"}

if getattr(PromptServer, "instance", None) is not None:
    @PromptServer.instance.routes.get("/yogurt_h3/director/detectors")
    async def list_detectors(request):
        return web.json_response(detectors())

    @PromptServer.instance.routes.post("/yogurt_h3/director/rewrite")
    async def preview_rewrite(request):
        """Rewrite the editor's project now, without queueing a render; results share the render's cache."""
        data = await request.json()

        def work():
            project = load_project(data["project"], data.get("settings"))
            return rewrite_project(project, rewriter_config(**data["rewriter"]), Media())

        try:
            _, prompts, report = await asyncio.to_thread(work)
        except Exception as error:  # surface provider and validation failures to the editor
            return web.json_response({"error": f"{type(error).__name__}: {error}"}, status=400)
        return web.json_response({"prompts": prompts, "report": report})

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
