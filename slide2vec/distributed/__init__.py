import os

import torch

_RANK = -1
_WORLD_SIZE = -1
_LOCAL_RANK = -1
_LOCAL_WORLD_SIZE = -1


def is_enabled() -> bool:
    """
    Returns:
        True if distributed mode has been enabled (the process topology is known).
    """
    return _RANK >= 0


def get_global_size() -> int:
    """
    Returns:
        The number of processes in the job.
    """
    return _WORLD_SIZE if is_enabled() else 1


def get_global_rank() -> int:
    """
    Returns:
        The rank of the current process in the job.
    """
    return _RANK if is_enabled() else 0


def get_local_rank() -> int:
    """
    Returns:
        The rank of the current process on its machine.
    """
    if not is_enabled():
        return 0
    assert 0 <= _LOCAL_RANK < _LOCAL_WORLD_SIZE
    return _LOCAL_RANK


def get_local_size() -> int:
    """
    Returns:
        The number of processes on the current machine.
    """
    if not is_enabled():
        return 1
    assert 0 <= _LOCAL_RANK < _LOCAL_WORLD_SIZE
    return _LOCAL_WORLD_SIZE


def is_main_process() -> bool:
    """
    Returns:
        True if the current process is the main one.
    """
    return get_global_rank() == 0


def _restrict_print_to_main_process() -> None:
    """
    This function disables printing when not in the main process
    """
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_main_process() or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


# The process-topology variables torchrun exports. MASTER_ADDR / MASTER_PORT are
# deliberately ignored: they only locate a process-group rendezvous, and no process
# group is ever created (issue #219).
_TORCHRUN_ENV_VARS = (
    "RANK",
    "WORLD_SIZE",
    "LOCAL_RANK",
    "LOCAL_WORLD_SIZE",
)


def _collect_env_vars() -> dict[str, str]:
    return {
        env_var: os.environ[env_var]
        for env_var in _TORCHRUN_ENV_VARS
        if env_var in os.environ
    }


def _check_env_variable(key: str, new_value: str):
    # Only check for difference with preset environment variables
    if key in os.environ and os.environ[key] != new_value:
        raise RuntimeError(
            f"Cannot export environment variables as {key} is already set"
        )


class _TorchDistributedEnvironment:
    def __init__(self):
        self.rank = -1
        self.world_size = -1
        self.local_rank = -1
        self.local_world_size = -1

        env_vars = _collect_env_vars()
        if not env_vars:
            # Environment is not set
            pass
        elif len(env_vars) == len(_TORCHRUN_ENV_VARS):
            # Environment is fully set
            return self._set_from_preset_env()
        else:
            # Environment is partially set
            collected_env_vars = ", ".join(env_vars.keys())
            raise RuntimeError(f"Partially set environment: {collected_env_vars}")

        if torch.cuda.device_count() > 0:
            return self._set_from_local()

        raise RuntimeError(
            "Can't determine the process topology: no torchrun environment and no CUDA device"
        )

    # Single node job with preset environment (i.e. torchrun)
    def _set_from_preset_env(self):
        # logger.info("Initialization from preset environment")
        self.rank = int(os.environ["RANK"])
        self.world_size = int(os.environ["WORLD_SIZE"])
        assert self.rank < self.world_size
        self.local_rank = int(os.environ["LOCAL_RANK"])
        self.local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])
        assert self.local_rank < self.local_world_size

    # Single node and GPU job (i.e. local script run)
    def _set_from_local(self):
        # logger.info("Initialization from local")
        self.rank = 0
        self.world_size = 1
        self.local_rank = 0
        self.local_world_size = 1

    def export(self, *, overwrite: bool) -> "_TorchDistributedEnvironment":
        # Export the topology so child processes and later lookups see one consistent
        # view. There is no env:// rendezvous to feed (no process group is created), so
        # MASTER_ADDR / MASTER_PORT are not exported.
        env_vars = {
            "RANK": str(self.rank),
            "WORLD_SIZE": str(self.world_size),
            "LOCAL_RANK": str(self.local_rank),
            "LOCAL_WORLD_SIZE": str(self.local_world_size),
        }
        if not overwrite:
            for k, v in env_vars.items():
                _check_env_variable(k, v)

        os.environ.update(env_vars)
        return self


def enable(
    *,
    set_cuda_current_device: bool = True,
    overwrite: bool = False,
):
    """Enable distributed mode.

    Reads the process topology (RANK / WORLD_SIZE / LOCAL_RANK / LOCAL_WORLD_SIZE) that
    torchrun exports — or single-process defaults when launched bare — and records it for
    the ``get_*`` helpers. No ``torch.distributed`` process group is created: the workers
    run no collectives (each rank writes its own artifacts and nobody gathers), so an NCCL
    group would only add init cost and a 14-day-timeout hang surface for no benefit. See
    issue #219.

    Args:
        set_cuda_current_device: If True, call torch.cuda.set_device() to set the
            current PyTorch CUDA device to the one matching the local rank.
        overwrite: If True, overwrites already set variables. Else fails.
    """

    global _RANK, _WORLD_SIZE, _LOCAL_RANK, _LOCAL_WORLD_SIZE
    if is_enabled():
        raise RuntimeError("Distributed mode has already been enabled")
    torch_env = _TorchDistributedEnvironment()
    torch_env.export(overwrite=overwrite)

    if set_cuda_current_device:
        torch.cuda.set_device(torch_env.local_rank)

    # Finalize setup
    _RANK = torch_env.rank
    _WORLD_SIZE = torch_env.world_size
    _LOCAL_RANK = torch_env.local_rank
    _LOCAL_WORLD_SIZE = torch_env.local_world_size
    _restrict_print_to_main_process()
