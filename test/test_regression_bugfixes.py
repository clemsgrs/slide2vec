import ast
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read_source(rel_path: str) -> str:
    return (ROOT / rel_path).read_text(encoding="utf-8")


def parse_source(rel_path: str) -> ast.AST:
    return ast.parse(read_source(rel_path))


class RegressionBugfixTests(unittest.TestCase):
    def test_main_uses_dedicated_process_groups_for_children(self):
        tree = parse_source("slide2vec/main.py")
        functions = {
            node.name: node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
        }
        for fn_name in ("run_feature_extraction", "run_feature_aggregation"):
            fn = functions[fn_name]
            popen_calls = [
                call
                for call in ast.walk(fn)
                if isinstance(call, ast.Call)
                and isinstance(call.func, ast.Attribute)
                and isinstance(call.func.value, ast.Name)
                and call.func.value.id == "subprocess"
                and call.func.attr == "Popen"
            ]
            self.assertTrue(popen_calls, f"No subprocess.Popen call found in {fn_name}")
            for call in popen_calls:
                kws = {kw.arg: kw.value for kw in call.keywords}
                self.assertIn(
                    "start_new_session",
                    kws,
                    f"{fn_name} must set start_new_session=True for safe killpg",
                )
                self.assertIsInstance(kws["start_new_session"], ast.Constant)
                self.assertTrue(kws["start_new_session"].value)

    def test_embed_and_aggregate_do_not_join_path_with_none(self):
        for rel_path in ("slide2vec/embed.py", "slide2vec/aggregate.py"):
            tree = parse_source(rel_path)
            bad_calls = []
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id == "Path"
                    and len(node.args) >= 2
                ):
                    second = node.args[1]
                    if (
                        isinstance(second, ast.Attribute)
                        and isinstance(second.value, ast.Name)
                        and second.value.id == "args"
                        and second.attr == "output_dir"
                    ):
                        bad_calls.append(node)
            self.assertFalse(
                bad_calls,
                f"{rel_path} should not call Path(cfg.output_dir, args.output_dir) directly",
            )

    def test_embed_has_no_barrier_calls_inside_try_block(self):
        tree = parse_source("slide2vec/embed.py")
        try_nodes = [node for node in ast.walk(tree) if isinstance(node, ast.Try)]
        self.assertTrue(try_nodes, "Expected at least one try/except block")

        barrier_calls_inside_try = []
        for try_node in try_nodes:
            for stmt in try_node.body:
                for node in ast.walk(stmt):
                    if (
                        isinstance(node, ast.Call)
                        and isinstance(node.func, ast.Attribute)
                        and node.func.attr == "barrier"
                    ):
                        barrier_calls_inside_try.append(node)

        self.assertFalse(
            barrier_calls_inside_try,
            "torch.distributed.barrier calls inside try blocks can deadlock when ranks diverge",
        )

    def test_region_model_factory_uses_tile_encoder_assignments(self):
        src = read_source("slide2vec/models/models.py")
        expected = {
            "conch": "tile_encoder = CONCH()",
            "musk": "tile_encoder = MUSK()",
            "phikonv2": "tile_encoder = PhikonV2()",
            "hibou": "tile_encoder = Hibou()",
            "kaiko": "tile_encoder = Kaiko(arch=options.arch)",
            "kaiko-midnight": "tile_encoder = Midnight12k()",
        }
        for model_name, assignment in expected.items():
            pattern = rf'elif options.name == "{re.escape(model_name)}":\n\s+{re.escape(assignment)}'
            self.assertRegex(
                src,
                pattern,
                f"Region-level branch for {model_name} should assign to tile_encoder",
            )

    def test_embed_reads_new_loader_config_keys(self):
        src = read_source("slide2vec/embed.py")
        expected_keys = [
            "embedding_pipeline",
            "rank_sharding_mode",
            "storage_mode",
            "num_workers_embedding",
            "prefetch_factor_embedding",
            "persistent_workers_embedding",
            "pin_memory_embedding",
            "loader_batch_timeout_sec",
            "log_perf_embedding",
        ]
        for key in expected_keys:
            self.assertIn(
                f"\"{key}\"",
                src,
                f"embed.py should reference speed.{key}",
            )


if __name__ == "__main__":
    unittest.main()
