import ast
import types
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EMBED_FILE = ROOT / "slide2vec/embed.py"


def load_functions(*fn_names):
    src = EMBED_FILE.read_text(encoding="utf-8")
    tree = ast.parse(src)
    fn_nodes = {
        node.name: node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
    }
    namespace = {}
    for name in fn_names:
        module = ast.Module(body=[fn_nodes[name]], type_ignores=[])
        code = compile(module, filename=str(EMBED_FILE), mode="exec")
        exec(code, namespace)
    return [namespace[name] for name in fn_names]


class EmbedPipelineModeTests(unittest.TestCase):
    def _cfg(self, mode: str):
        return types.SimpleNamespace(speed=types.SimpleNamespace(rank_sharding_mode=mode))

    def test_explicit_tile_mode(self):
        get_speed_option, decide_sharding_mode = load_functions(
            "get_speed_option", "decide_sharding_mode"
        )
        _ = get_speed_option
        cfg = self._cfg("tile")
        self.assertEqual(decide_sharding_mode(cfg, pending_count=100, world_size=8), "tile")

    def test_explicit_slide_mode(self):
        get_speed_option, decide_sharding_mode = load_functions(
            "get_speed_option", "decide_sharding_mode"
        )
        _ = get_speed_option
        cfg = self._cfg("slide")
        self.assertEqual(decide_sharding_mode(cfg, pending_count=1, world_size=8), "slide")

    def test_auto_mode_threshold(self):
        get_speed_option, decide_sharding_mode = load_functions(
            "get_speed_option", "decide_sharding_mode"
        )
        _ = get_speed_option
        cfg = self._cfg("auto")
        self.assertEqual(decide_sharding_mode(cfg, pending_count=8, world_size=8), "slide")
        self.assertEqual(decide_sharding_mode(cfg, pending_count=7, world_size=8), "tile")

    def test_invalid_mode_raises(self):
        get_speed_option, decide_sharding_mode = load_functions(
            "get_speed_option", "decide_sharding_mode"
        )
        _ = get_speed_option
        cfg = self._cfg("invalid")
        with self.assertRaises(ValueError):
            decide_sharding_mode(cfg, pending_count=10, world_size=8)


if __name__ == "__main__":
    unittest.main()
