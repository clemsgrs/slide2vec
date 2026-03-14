def initialize_wandb(*args, **kwargs):
    from .utils import initialize_wandb as impl

    return impl(*args, **kwargs)


def fix_random_seeds(*args, **kwargs):
    from .utils import fix_random_seeds as impl

    return impl(*args, **kwargs)


def get_sha(*args, **kwargs):
    from .utils import get_sha as impl

    return impl(*args, **kwargs)


def update_state_dict(*args, **kwargs):
    from .utils import update_state_dict as impl

    return impl(*args, **kwargs)


def setup_logging(*args, **kwargs):
    from .log_utils import setup_logging as impl

    return impl(*args, **kwargs)


__all__ = [
    "initialize_wandb",
    "fix_random_seeds",
    "get_sha",
    "update_state_dict",
    "setup_logging",
]
