"""Each sharded torchrun worker must see only its own GPU (issue #326).

``import cucim`` opens a CUDA context on every visible device, so a rank that can see its
siblings' GPUs holds an idle context on each of them.
"""

from __future__ import annotations

import ast
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from slide2vec.distributed import pin_gpu, worker_entry
from slide2vec.distributed.worker_entry import resolve_worker_rank


def test_rank_sees_only_its_own_gpu():
    environ = {"LOCAL_RANK": "2"}
    pin_gpu.pin_to_own_gpu(environ)
    assert environ["CUDA_VISIBLE_DEVICES"] == "2"
    assert environ[pin_gpu.PINNED_ENV] == "1"


def test_rank_picks_its_entry_of_an_inherited_gpu_list():
    environ = {"LOCAL_RANK": "1", "CUDA_VISIBLE_DEVICES": "GPU-aaa, GPU-bbb,GPU-ccc"}
    pin_gpu.pin_to_own_gpu(environ)
    assert environ["CUDA_VISIBLE_DEVICES"] == "GPU-bbb"


def test_more_ranks_than_visible_gpus_is_an_error():
    with pytest.raises(RuntimeError, match="LOCAL_RANK=2"):
        pin_gpu.pin_to_own_gpu({"LOCAL_RANK": "2", "CUDA_VISIBLE_DEVICES": "0,1"})


def test_bootstrap_imports_nothing_that_could_initialise_cuda():
    # It runs by path, before the package import that freezes CUDA's visible-device list.
    tree = ast.parse(Path(pin_gpu.__file__).read_text())
    imported = {
        alias.name.split(".")[0]
        for node in ast.walk(tree) if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        (node.module or "").split(".")[0]
        for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
    }
    assert imported == {"os", "runpy", "sys"}


def test_pinned_worker_uses_cuda_0_but_keeps_its_rank_label(monkeypatch):
    assert worker_entry.PINNED_ENV == pin_gpu.PINNED_ENV
    monkeypatch.setenv("RANK", "3")
    monkeypatch.setenv("WORLD_SIZE", "4")
    monkeypatch.setenv("LOCAL_RANK", "3")
    monkeypatch.setenv(worker_entry.PINNED_ENV, "1")
    rank = resolve_worker_rank()
    assert rank.device == "cuda:0"
    assert (rank.global_rank, rank.local_rank) == (3, 3)


def test_unpinned_worker_uses_its_local_rank_device(monkeypatch):
    monkeypatch.setenv("LOCAL_RANK", "3")
    monkeypatch.delenv(worker_entry.PINNED_ENV, raising=False)
    assert resolve_worker_rank().device == "cuda:3"


@pytest.mark.parametrize("pin_gpus", [True, False])
def test_launcher_runs_the_bootstrap_by_path_only_when_pinning(tmp_path, pin_gpus):
    from slide2vec.runtime.distributed import run_torchrun_worker

    commands = []

    class _Done:
        pid = None
        stdout = stderr = None

        def __init__(self, command, **kwargs):
            commands.append(command)

        def poll(self):
            return 0

        def wait(self, timeout=None):
            return 0

    run_torchrun_worker(
        module="slide2vec.distributed.dense_worker", num_gpus=4, output_dir=tmp_path,
        request_path=tmp_path / "request.json", failure_title="failed",
        pin_gpus=pin_gpus, popen_factory=_Done,
    )
    (command,) = commands
    tail = command[command.index("--nproc_per_node=4") + 1:]
    if pin_gpus:
        assert Path(tail[0]) == Path(pin_gpu.__file__).resolve()
        assert tail[1] == "slide2vec.distributed.dense_worker"
    else:
        assert tail[:2] == ["-m", "slide2vec.distributed.dense_worker"]


@pytest.mark.parametrize("pinned, ordinal", [(True, 0), (False, 3)])
def test_enable_binds_a_pinned_rank_to_cuda_0(monkeypatch, pinned, ordinal):
    # pipeline_worker and direct_embed_worker go through distributed.enable(); once pinned,
    # cuda:<local_rank> no longer exists for any rank but 0.
    import torch

    import slide2vec.distributed as distributed

    for name in ("_RANK", "_WORLD_SIZE", "_LOCAL_RANK", "_LOCAL_WORLD_SIZE"):
        monkeypatch.setattr(distributed, name, -1)
    for key in ("RANK", "WORLD_SIZE", "LOCAL_RANK", "LOCAL_WORLD_SIZE"):
        monkeypatch.setenv(key, "4" if "SIZE" in key else "3")
    if pinned:
        monkeypatch.setenv(worker_entry.PINNED_ENV, "1")
    else:
        monkeypatch.delenv(worker_entry.PINNED_ENV, raising=False)
    monkeypatch.setattr(distributed, "_restrict_print_to_main_process", lambda: None)
    bound = []
    monkeypatch.setattr(torch.cuda, "set_device", bound.append)

    distributed.enable(overwrite=True)

    assert bound == [ordinal]
    assert distributed.get_device_ordinal() == ordinal
    assert distributed.get_local_rank() == 3


@pytest.mark.parametrize("worker", ["pipeline_worker", "direct_embed_worker"])
def test_enable_based_workers_build_the_model_on_the_device_ordinal(worker):
    source = (Path(pin_gpu.__file__).parent / f"{worker}.py").read_text()
    assert 'device=f"cuda:{distributed.get_device_ordinal()}"' in source
    assert 'progress_label=f"cuda:{local_rank}"' in source


def test_every_torchrun_stage_pins_its_workers():
    runtime = Path(pin_gpu.__file__).resolve().parents[1] / "runtime"
    launches = [
        node
        for path in runtime.glob("*.py")
        for node in ast.walk(ast.parse(path.read_text()))
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "run_torchrun_worker"
    ]
    assert len(launches) == 5
    for call in launches:
        (pin,) = [kw.value for kw in call.keywords if kw.arg == "pin_gpus"]
        assert pin.value is True


def test_workers_keep_the_parents_working_directory(tmp_path, monkeypatch):
    # A relative output_dir must mean the same place in the parent and in every rank, so the
    # checkout is made importable through PYTHONPATH rather than by moving the workers' cwd.
    import os

    import slide2vec
    from slide2vec.runtime.distributed import run_torchrun_worker

    monkeypatch.setenv("PYTHONPATH", "/somewhere/else")
    launches = []

    class _Done:
        pid = None
        stdout = stderr = None

        def __init__(self, command, **kwargs):
            launches.append((command, kwargs))

        def poll(self):
            return 0

        def wait(self, timeout=None):
            return 0

    monkeypatch.chdir(tmp_path)
    run_torchrun_worker(
        module="slide2vec.distributed.dense_worker", num_gpus=2, output_dir=Path("out"),
        request_path=Path("out") / "request.json", failure_title="failed", popen_factory=_Done,
    )
    ((command, kwargs),) = launches
    assert kwargs.get("cwd") is None
    assert command[command.index("--output-dir") + 1] == "out"
    package_root = str(Path(slide2vec.__file__).resolve().parents[1])
    assert kwargs["env"]["PYTHONPATH"].split(os.pathsep) == [package_root, "/somewhere/else"]


def test_bootstrap_prefers_its_checkout_and_keeps_relative_paths(tmp_path):
    # Minimal competing packages let a fresh interpreter exercise import precedence
    # without loading CUDA or model dependencies.
    package = tmp_path / "parent-checkout" / "slide2vec"
    bootstrap = package / "distributed" / "pin_gpu.py"
    bootstrap.parent.mkdir(parents=True)
    bootstrap.write_text(Path(pin_gpu.__file__).read_text())
    (package / "__init__.py").write_text("")

    work_dir = tmp_path / "other-checkout"
    shadow_package = work_dir / "slide2vec"
    shadow_package.mkdir(parents=True)
    (shadow_package / "__init__.py").write_text(
        "raise AssertionError('worker imported the competing checkout')\n"
    )
    (work_dir / "input.txt").write_text("relative input")
    (work_dir / "out").mkdir()
    (work_dir / "out" / "request.json").write_text(
        json.dumps({"input": "input.txt", "output": "out/report.json"})
    )
    # This module is only available in the working directory, which must remain
    # importable after the parent's package root.
    (work_dir / "probe_worker.py").write_text(
        "import json, os, sys\n"
        "from pathlib import Path\n"
        "import slide2vec\n"
        "request = json.loads(Path(sys.argv[1]).read_text())\n"
        "Path(request['output']).write_text(json.dumps({\n"
        "    'package': slide2vec.__file__,\n"
        "    'cwd': os.getcwd(),\n"
        "    'input': Path(request['input']).read_text(),\n"
        "    'visible': os.environ['CUDA_VISIBLE_DEVICES'],\n"
        "}))\n"
    )
    env = dict(os.environ)
    env.update(LOCAL_RANK="1", CUDA_VISIBLE_DEVICES="2,3", PYTHONPATH=str(package.parent))
    result = subprocess.run(
        [sys.executable, str(bootstrap), "probe_worker", "out/request.json"],
        cwd=work_dir, env=env, capture_output=True, text=True, timeout=15,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads((work_dir / "out" / "report.json").read_text()) == {
        "package": str(package / "__init__.py"),
        "cwd": str(work_dir),
        "input": "relative input",
        "visible": "3",
    }
