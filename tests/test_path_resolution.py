import unittest

import importlib.util
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]


def load_paths_module():
    module_path = ROOT / "slide2vec" / "utils" / "paths.py"
    spec = importlib.util.spec_from_file_location("slide2vec_utils_paths", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class PathResolutionTests(unittest.TestCase):
    def test_resolve_coordinates_dir_prefers_read_coordinates_from(self):
        paths = load_paths_module()

        cfg = SimpleNamespace(
            output_dir="/tmp/experiment-output",
            tiling=SimpleNamespace(read_coordinates_from="/tmp/external-coordinates"),
        )

        self.assertEqual(
            paths.resolve_coordinates_dir(cfg),
            Path("/tmp/external-coordinates"),
        )

    def test_resolve_coordinates_dir_falls_back_to_output_coordinates_dir(self):
        paths = load_paths_module()

        cfg = SimpleNamespace(
            output_dir="/tmp/experiment-output",
            tiling=SimpleNamespace(read_coordinates_from=None),
        )

        self.assertEqual(
            paths.resolve_coordinates_dir(cfg),
            Path("/tmp/experiment-output/coordinates"),
        )


if __name__ == "__main__":
    unittest.main()
