"""Bootstrap that shows a torchrun worker only its own GPU, then runs the worker (issue #326).

``import cucim`` opens a CUDA context on every visible device, so a rank that can see its
siblings' GPUs holds an idle context on each of them. CUDA reads ``CUDA_VISIBLE_DEVICES``
once, when it initialises, and ``import slide2vec`` already initialises it (``transformers``
calls ``torch.cuda.is_available()`` at import). A worker module therefore cannot narrow its
own visibility: ``python -m slide2vec.distributed.<worker>`` has imported the package before
the worker's first line runs. torchrun runs this file **by path** instead, so the narrowing
happens before anything is imported. Keep this file free of slide2vec and torch imports.

Only for workers that run no collectives: a pinned rank cannot see its siblings' GPUs.

Usage: ``torchrun ... pin_gpu.py <worker module> [worker args...]``
"""

import os
import runpy
import sys

PINNED_ENV = "SLIDE2VEC_WORKER_GPU_PINNED"  # Mirrored in worker_entry.PINNED_ENV.


def pin_to_own_gpu(environ) -> None:
    """Narrow ``CUDA_VISIBLE_DEVICES`` to the ``LOCAL_RANK``-th GPU this process inherited."""
    local_rank = int(environ["LOCAL_RANK"])
    visible = [gpu.strip() for gpu in environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if gpu.strip()]
    if visible and local_rank >= len(visible):
        raise RuntimeError(
            f"LOCAL_RANK={local_rank} but CUDA_VISIBLE_DEVICES lists only {len(visible)} GPU(s)"
        )
    environ["CUDA_VISIBLE_DEVICES"] = visible[local_rank] if visible else str(local_rank)
    environ[PINNED_ENV] = "1"


def main() -> None:
    pin_to_own_gpu(os.environ)
    module, *worker_args = sys.argv[1:]
    # Prefer the package that supplied this bootstrap over a competing checkout in cwd.
    # Keep cwd importable without changing how relative data and output paths resolve.
    package_root = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", ".."))
    sys.path[:1] = [package_root, os.getcwd()]
    sys.argv = [module, *worker_args]
    runpy.run_module(module, run_name="__main__", alter_sys=True)


if __name__ == "__main__":
    main()
