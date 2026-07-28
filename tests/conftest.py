import importlib.util
import multiprocessing
import os
from pathlib import Path
import subprocess
import sys
import types

import pytest


if importlib.util.find_spec("transformers") is None:
    transformers_module = types.ModuleType("transformers")
    transformers_module.__path__ = []

    image_processing_utils = types.ModuleType("transformers.image_processing_utils")

    class BaseImageProcessor:
        pass

    image_processing_utils.BaseImageProcessor = BaseImageProcessor
    transformers_module.image_processing_utils = image_processing_utils

    sys.modules.setdefault("transformers", transformers_module)
    sys.modules["transformers.image_processing_utils"] = image_processing_utils


@pytest.fixture
def run_current_test_in_subprocess(request):
    """Re-run the current pytest case once in an isolated, timeout-bounded process."""

    def run(*, child_env_name: str, timeout: int = 25) -> None:
        env = os.environ.copy()
        env[child_env_name] = "1"
        subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "--no-cov", request.node.nodeid],
            cwd=Path(request.config.rootpath),
            env=env,
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )

    return run


@pytest.fixture
def assert_auto_worker_workflow_completes_in_subprocess(run_current_test_in_subprocess):
    """Run an auto-worker image workflow in a bounded child and assert clean shutdown."""

    def assert_workflow(*, child_env_name: str, workflow, expected_sample_ids: list[str]) -> None:
        if os.environ.get(child_env_name) == "1":
            artifacts = workflow()
            assert [artifact.sample_id for artifact in artifacts] == expected_sample_ids
            assert all(artifact.path.exists() for artifact in artifacts)
            assert multiprocessing.active_children() == []
            return
        run_current_test_in_subprocess(child_env_name=child_env_name)

    return assert_workflow


@pytest.fixture
def build_auto_worker_model(tmp_path, monkeypatch):
    """Build a public Model with a local loaded backend and default image execution."""

    def build(loaded):
        from slide2vec.api import ExecutionOptions, Model

        model = Model(name="virchow2", device="cpu")
        monkeypatch.setattr(model, "_load_backend", lambda: loaded)
        execution = ExecutionOptions(
            output_dir=tmp_path / "out",
            num_gpus=1,
            precision="fp32",
            num_workers_per_gpu=None,
        )
        return model, execution

    return build
