import asyncio
import json
import unittest
from unittest.mock import patch

from audit_examples import ROOT, CLASSES, convert, widget_names, execution, nodes, folder_paths


class ExampleWorkflowTests(unittest.TestCase):
    def check_example(self, filename):
        workflow = json.loads((ROOT / "example_workflows" / filename).read_text(encoding="utf-8"))
        by_id = {node["id"]: node for node in workflow["nodes"]}
        self.assertEqual(len(by_id), len(workflow["nodes"]))
        by_link = {link[0]: link for link in workflow["links"]}
        self.assertEqual(len(by_link), len(workflow["links"]))
        for link_id, source_id, source_slot, target_id, target_slot, kind in workflow["links"]:
            source = by_id[source_id]["outputs"][source_slot]
            target = by_id[target_id]["inputs"][target_slot]
            self.assertIn(link_id, source["links"])
            self.assertEqual(target["link"], link_id)
            self.assertEqual(kind, source["type"])
            self.assertTrue(target["type"] in (kind, "*", "ANY"))
        for node in workflow["nodes"]:
            if node["type"] == "MarkdownNote":
                continue
            cls = CLASSES[node["type"]]
            schema = cls.INPUT_TYPES()
            inputs = {**schema.get("required", {}), **schema.get("optional", {})}
            for input_ in node.get("inputs", []):
                self.assertIn(input_["name"].split(".")[0], inputs)
            self.assertEqual([output["type"] for output in node.get("outputs", [])], list(cls.RETURN_TYPES))
            if node.get("mode") in (2, 4):
                continue
            names = widget_names(node["type"], inputs)
            values = node.get("widgets_values", [])
            self.assertLessEqual(len(values), len(names))
            for omitted in names[len(values):]:
                self.assertIn(omitted, schema.get("optional", {}))
            for name, value in zip(names, values):
                if name == "control_after_generate":
                    self.assertEqual(value, "fixed")
                elif name == "api_key":
                    self.assertEqual(value, "")
        prompt = convert(workflow)
        artifact_names = {
            "diffusion_models": [node["inputs"]["unet_name"] for node in prompt.values() if node["class_type"] == "UNETLoader"],
            "text_encoders": [node["inputs"]["clip_name"] for node in prompt.values() if node["class_type"] == "CLIPLoader"],
            "vae": [node["inputs"]["vae_name"] for node in prompt.values() if node["class_type"] == "VAELoader"],
        }
        original_list = folder_paths.get_filename_list
        def fixture_names(category):
            return artifact_names[category] if category in artifact_names else original_list(category)
        # This verifies API structure only; model files remain user-selected install requirements.
        with patch.dict(nodes.NODE_CLASS_MAPPINGS, CLASSES), patch.object(folder_paths, "get_filename_list", side_effect=fixture_names):
            valid, error, outputs, node_errors = asyncio.run(execution.validate_prompt("example-test", prompt, None))
        self.assertTrue(valid, error)
        self.assertEqual(node_errors, {})
        for node_id, node in prompt.items():
            if node["class_type"] == "SaveVideo":
                self.assertIn(node_id, outputs)

    def test_character_replace_example_ports_widgets_and_api_with_model_name_fixture(self):
        self.check_example("yogurt_h3_long_video_character_replace.json")

    def test_refine_example_ports_widgets_and_api_with_model_name_fixture(self):
        self.check_example("yogurt_h3_character_replace_refine.json")

    def test_native_modes_example_ports_widgets_and_api_with_model_name_fixture(self):
        self.check_example("yogurt_h3_native_modes.json")

    def test_story_example_ports_widgets_and_api_with_model_name_fixture(self):
        self.check_example("yogurt_h3_story_long_video.json")


if __name__ == "__main__":
    unittest.main()
