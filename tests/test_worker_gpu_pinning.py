"""Each sharded torchrun worker must see only its own GPU (issue #326).

``import cucim`` opens a CUDA context on every visible device, so a rank that can see its
siblings' GPUs holds an idle context on each of them.
"""

from __future__ import annotations

import ast
from pathlib import Path

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
