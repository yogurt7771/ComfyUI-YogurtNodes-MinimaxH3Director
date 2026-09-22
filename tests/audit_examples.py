"""Compare saved example ports and widgets with installed schemas, without API calls."""

import ast
import asyncio
import inspect
import json
from pathlib import Path

from test_continuity_runtime import PLUGIN
import nodes
import execution
import folder_paths
from unittest.mock import patch
from comfy_api.latest import _io
from comfy_extras import nodes_audio, nodes_custom_sampler, nodes_lt, nodes_minimax_h3, nodes_primitive, nodes_video, nodes_preview_any, nodes_textgen


ROOT = Path(__file__).resolve().parents[1]
CLASSES = dict(nodes.NODE_CLASS_MAPPINGS)
CLASSES.update(PLUGIN.NODE_CLASS_MAPPINGS)
for module in (nodes_audio, nodes_custom_sampler, nodes_lt, nodes_minimax_h3, nodes_primitive, nodes_video, nodes_preview_any, nodes_textgen):
    for name, value in vars(module).items():
        if inspect.isclass(value) and value.__module__ == module.__name__ and hasattr(value, "define_schema"):
            CLASSES[value.define_schema().node_id] = value
    CLASSES.update(getattr(module, "NODE_CLASS_MAPPINGS", {}))

# Evaluate only the static schema method, never import Yogurt's client/config code.
tree = ast.parse((ROOT.parent / "ComfyUI-YogurtNodes/yogurt_nodes/llm/openrouter_image_understand.py").read_text(encoding="utf-8"))
source_class = next(item for item in tree.body if isinstance(item, ast.ClassDef) and item.name == "OpenRouterImageUnderstand")
source_class.body = [item for item in source_class.body if isinstance(item, ast.FunctionDef) and item.name == "INPUT_TYPES"
                     or isinstance(item, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "RETURN_TYPES" for target in item.targets)]
class Providers:
    @staticmethod
    def get_infrastructure_providers():
        return ["auto"]
context = {"OpenRouterClient": Providers}
exec(compile(ast.Module(body=[source_class], type_ignores=[]), "<OpenRouter schema only>", "exec"), context)
CLASSES["YogurtOpenRouterImageUnderstand"] = context["OpenRouterImageUnderstand"]


def widget_names(node_type, inputs):
    if node_type == "SaveVideo":
        return ["filename_prefix", "format", "codec"]
    names = []
    for name, entry in inputs.items():
        kind = entry[0]
        options = entry[1] if len(entry) > 1 else {}
        if not options.get("forceInput") and (isinstance(kind, list) or kind in ("STRING", "INT", "FLOAT", "BOOLEAN", "COMBO")):
            names.append(name)
            if options.get("control_after_generate") or name in ("seed", "noise_seed"):
                names.append("control_after_generate")
            if options.get("image_upload"):
                names.append("upload")
    return names


def convert(workflow):
    result = {}
    links = {item[0]: item for item in workflow["links"]}
    by_id = {node["id"]: node for node in workflow["nodes"]}
    # Like the frontend, muted nodes and links from them are not sent; a bypassed node forwards
    # each output from its first linked input of the same type.
    muted = {node["id"] for node in workflow["nodes"] if node.get("mode") == 2}
    bypassed = {node["id"] for node in workflow["nodes"] if node.get("mode") == 4}

    def source(link):
        node_id, slot = link[1], link[2]
        if node_id in muted:
            return None
        if node_id not in bypassed:
            return [str(node_id), slot]
        kind = by_id[node_id]["outputs"][slot]["type"]
        for item in by_id[node_id].get("inputs", []):
            if item["type"] == kind and item.get("link") is not None:
                return source(links[item["link"]])
        return None

    for node in workflow["nodes"]:
        if node["type"] == "MarkdownNote" or node["id"] in muted | bypassed:
            continue
        schema = CLASSES[node["type"]].INPUT_TYPES()
        entries = {**schema.get("required", {}), **schema.get("optional", {})}
        inputs = {name: value for name, value in zip(widget_names(node["type"], entries), node.get("widgets_values", []))
                  if name not in ("control_after_generate", "upload")}
        for item in node.get("inputs", []):
            if item.get("link") is not None and (origin := source(links[item["link"]])) is not None:
                inputs[item["name"]] = origin
        result[str(node["id"])] = {"class_type": node["type"], "inputs": inputs}
    return result


def audit():
    errors = []
    for path in sorted((ROOT / "example_workflows").glob("*.json")):
        workflow = json.loads(path.read_text(encoding="utf-8"))
        by_id = {node["id"]: node for node in workflow["nodes"]}
        by_link = {link[0]: link for link in workflow["links"]}
        print(path.name)
        for node in workflow["nodes"]:
            if node["type"] == "MarkdownNote":
                continue
            cls = CLASSES[node["type"]]
            schema = cls.INPUT_TYPES()
            inputs = {**schema.get("required", {}), **schema.get("optional", {})}
            returns = cls.RETURN_TYPES
            if len(node.get("outputs", [])) != len(returns):
                errors.append(f"{path.name}:{node['id']} output count")
            for index, output in enumerate(node.get("outputs", [])):
                if index < len(returns) and output["type"] != returns[index]:
                    errors.append(f"{path.name}:{node['id']} output {index} type {output['type']} != {returns[index]}")
                for link_id in output.get("links") or []:
                    link = by_link.get(link_id)
                    if not link or link[1:3] != [node["id"], index]:
                        errors.append(f"{path.name}:{node['id']} output link {link_id}")
            for index, input_ in enumerate(node.get("inputs", [])):
                if input_["name"].split(".")[0] not in inputs:
                    errors.append(f"{path.name}:{node['id']} unknown input {input_['name']}")
                if input_.get("link") is not None:
                    link = by_link.get(input_["link"])
                    if not link or link[3:5] != [node["id"], index]:
                        errors.append(f"{path.name}:{node['id']} input link {input_['link']}")
                    elif by_id[link[1]]["outputs"][link[2]]["type"] not in (input_["type"], "*", "ANY") and input_["type"] not in ("*", "ANY"):
                        errors.append(f"{path.name}:{node['id']} link type {input_['name']}")
            values = node.get("widgets_values", [])
            widgets = widget_names(node["type"], inputs)
            print(" ", node["id"], node["type"], "widgets", list(zip(widgets, values)), "counts", len(widgets), len(values))
        prompt = convert(workflow)
        with patch.dict(nodes.NODE_CLASS_MAPPINGS, CLASSES):
            validation = asyncio.run(execution.validate_prompt("example-audit", prompt, None))
        print("LOCAL MODEL AVAILABILITY", {key: value["class_type"] for key, value in validation[3].items()})
        artifact_names = {
            "diffusion_models": [node["inputs"]["unet_name"] for node in prompt.values() if node["class_type"] == "UNETLoader"],
            "text_encoders": [node["inputs"]["clip_name"] for node in prompt.values() if node["class_type"] == "CLIPLoader"],
            "vae": [node["inputs"]["vae_name"] for node in prompt.values() if node["class_type"] == "VAELoader"],
        }
        original_list = folder_paths.get_filename_list
        def fixture_names(category):
            return artifact_names[category] if category in artifact_names else original_list(category)
        with patch.dict(nodes.NODE_CLASS_MAPPINGS, CLASSES), patch.object(folder_paths, "get_filename_list", side_effect=fixture_names):
            structure = asyncio.run(execution.validate_prompt("example-schema-audit", prompt, None))
        print("STRUCTURAL API VALIDATION (model-name fixture only)", structure)
        if not structure[0] or structure[3]:
            errors.append(f"{path.name}: structural API validation failed")
    print("ERRORS", json.dumps(errors, ensure_ascii=False))
    return errors


if __name__ == "__main__":
    raise SystemExit(bool(audit()))
