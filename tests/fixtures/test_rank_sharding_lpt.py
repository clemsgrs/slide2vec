import ast
import copy
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


class RankShardingLptTests(unittest.TestCase):
    def test_deterministic_assignment(self):
        (assign_slides_lpt,) = load_functions("assign_slides_lpt")
        slides = [
            {"name": "a", "tile_count": 10},
            {"name": "b", "tile_count": 9},
            {"name": "c", "tile_count": 8},
            {"name": "d", "tile_count": 7},
        ]
        result_1 = assign_slides_lpt(copy.deepcopy(slides), world_size=2)
        result_2 = assign_slides_lpt(copy.deepcopy(slides), world_size=2)
        self.assertEqual(result_1, result_2)

    def test_balance_on_skewed_distribution(self):
        (assign_slides_lpt,) = load_functions("assign_slides_lpt")
        slides = [
            {"name": "big", "tile_count": 50},
            {"name": "m1", "tile_count": 10},
            {"name": "m2", "tile_count": 10},
            {"name": "m3", "tile_count": 10},
            {"name": "m4", "tile_count": 10},
        ]
        assignments = assign_slides_lpt(copy.deepcopy(slides), world_size=2)

        assigned_names = []
        rank_loads = {}
        for rank, rank_slides in assignments.items():
            assigned_names.extend(slide["name"] for slide in rank_slides)
            rank_loads[rank] = sum(slide["tile_count"] for slide in rank_slides)

        self.assertCountEqual(assigned_names, [slide["name"] for slide in slides])
        self.assertLessEqual(abs(rank_loads[0] - rank_loads[1]), 10)


if __name__ == "__main__":
    unittest.main()
