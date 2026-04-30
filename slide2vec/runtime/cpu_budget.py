"""CPU budget reasoning for on-the-fly tiling and execution serialization."""

import logging
import os
from typing import Any, Sequence

from slide2vec.api import ExecutionOptions, PreprocessingConfig
from slide2vec.runtime import serialization as runtime_serialization
from slide2vec.runtime import tiling as runtime_tiling
from slide2vec.utils.utils import cpu_worker_limit, slurm_cpu_limit


def resolve_on_the_fly_num_workers(num_cucim_workers: int, num_gpus: int) -> tuple[int, str]:
    if int(num_cucim_workers) < 1:
        raise ValueError("num_cucim_workers must be at least 1")
    cpu_count = os.cpu_count() or 1
    worker_budget = max(1, cpu_worker_limit() // max(1, int(num_gpus)))
    details = [f"cpu_count={cpu_count}"]
    slurm_limit = slurm_cpu_limit()
    if slurm_limit is not None:
        details.append(f"slurm_cpu_limit={slurm_limit}")
    details.append(f"num_gpus={num_gpus}")
    effective_num_workers = max(1, worker_budget // num_cucim_workers)
    details.append(f"num_cucim_workers={num_cucim_workers}")
    return effective_num_workers, " // ".join(details)


def serialize_execution(
    execution: ExecutionOptions,
    *,
    preprocessing: PreprocessingConfig | None = None,
) -> dict[str, Any]:
    effective_num_workers_per_gpu = None
    if preprocessing is not None and preprocessing.on_the_fly and preprocessing.read_tiles_from is None:
        effective_num_workers_per_gpu, _ = resolve_on_the_fly_num_workers(
            preprocessing.num_cucim_workers,
            num_gpus=execution.num_gpus,
        )
    return runtime_serialization.serialize_execution(
        execution,
        effective_num_workers_per_gpu=effective_num_workers_per_gpu,
    )


def log_on_the_fly_worker_override_once(
    preprocessing: PreprocessingConfig,
    execution: ExecutionOptions,
    tiling_results: Sequence[Any],
) -> None:
    if not preprocessing.on_the_fly or preprocessing.read_tiles_from is not None:
        return
    if not any(
        runtime_tiling.resolve_slide_backend(preprocessing, tiling_result) == "cucim"
        for tiling_result in tiling_results
    ):
        return
    effective_num_workers_per_gpu, worker_context = resolve_on_the_fly_num_workers(
        preprocessing.num_cucim_workers,
        num_gpus=execution.num_gpus,
    )
    if effective_num_workers_per_gpu == execution.resolved_num_workers_per_gpu():
        return
    logging.getLogger(__name__).info(
        f"on-the-fly mode: setting DataLoader num_workers_per_gpu={effective_num_workers_per_gpu} "
        f"({worker_context}); "
        f"ignoring speed.num_workers_per_gpu={execution.num_workers_per_gpu}"
    )
