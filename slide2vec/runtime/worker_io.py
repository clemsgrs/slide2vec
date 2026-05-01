"""DataLoader worker stdio handling and CUDA-runtime detection helpers."""

import os
import tempfile
from typing import Any

import torch


def redirect_worker_output() -> None:
    """Redirect worker stdout/stderr to a shared cucim log file."""
    worker_log_path = os.path.join(
        tempfile.gettempdir(),
        "slide2vec-cucim-workers.log",
    )
    worker_log_fd = os.open(
        worker_log_path,
        os.O_WRONLY | os.O_CREAT | os.O_APPEND,
        0o644,
    )
    try:
        os.dup2(worker_log_fd, 1)
        os.dup2(worker_log_fd, 2)
    finally:
        os.close(worker_log_fd)


def configure_cucim_worker_stderr(loader_kwargs: dict[str, Any], *, backend: str) -> None:
    """Inject a worker_init_fn that silences cucim worker stderr."""
    if backend != "cucim" or int(loader_kwargs.get("num_workers", 0)) <= 0:
        return
    existing_worker_init = loader_kwargs.get("worker_init_fn")

    def _worker_init(worker_id: int) -> None:
        redirect_worker_output()
        if existing_worker_init is not None:
            existing_worker_init(worker_id)

    loader_kwargs["worker_init_fn"] = _worker_init


def should_suppress_cucim_dataloader_stderr(dataloader) -> bool:
    if int(getattr(dataloader, "num_workers", 0)) <= 0:
        return False
    collate_fn = getattr(dataloader, "collate_fn", None)
    reader = getattr(collate_fn, "_reader", None)
    return getattr(reader, "_backend", None) == "cucim"


def uses_cuda_runtime(device) -> bool:
    return str(device).startswith("cuda") and torch.cuda.is_available()
