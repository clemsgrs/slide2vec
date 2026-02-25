import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATASET_FILE = ROOT / "slide2vec/data/dataset.py"


class DatasetWorkerCacheTests(unittest.TestCase):
    def setUp(self):
        self.src = DATASET_FILE.read_text(encoding="utf-8")
        self.tree = ast.parse(self.src)
        self.tile_dataset = next(
            node
            for node in self.tree.body
            if isinstance(node, ast.ClassDef) and node.name == "TileDataset"
        )

    def _get_method(self, name: str):
        return next(
            node
            for node in self.tile_dataset.body
            if isinstance(node, ast.FunctionDef) and node.name == name
        )

    def test_private_worker_cache_members_initialized(self):
        init_fn = self._get_method("__init__")
        attrs = set()
        for node in ast.walk(init_fn):
            if (
                isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Attribute)
                and isinstance(node.targets[0].value, ast.Name)
                and node.targets[0].value.id == "self"
                and isinstance(node.value, ast.Constant)
                and node.value.value is None
            ):
                attrs.add(node.targets[0].attr)
        self.assertIn("_wsi", attrs)
        self.assertIn("_worker_id", attrs)

    def test_worker_wsi_helper_exists_and_uses_worker_info(self):
        helper_fn = self._get_method("_get_worker_wsi")
        helper_src = ast.get_source_segment(self.src, helper_fn)
        self.assertIn("torch.utils.data.get_worker_info()", helper_src)
        self.assertIn("wsd.WholeSlideImage(self.path, backend=self.backend)", helper_src)

    def test_getitem_uses_helper_and_not_direct_constructor(self):
        getitem_fn = self._get_method("__getitem__")
        helper_calls = 0
        direct_ctor_calls = 0
        for node in ast.walk(getitem_fn):
            if isinstance(node, ast.Call):
                if (
                    isinstance(node.func, ast.Attribute)
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "self"
                    and node.func.attr == "_get_worker_wsi"
                ):
                    helper_calls += 1
                if (
                    isinstance(node.func, ast.Attribute)
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "wsd"
                    and node.func.attr == "WholeSlideImage"
                ):
                    direct_ctor_calls += 1

        self.assertGreater(helper_calls, 0)
        self.assertEqual(direct_ctor_calls, 0)


if __name__ == "__main__":
    unittest.main()
