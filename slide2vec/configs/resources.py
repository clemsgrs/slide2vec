from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


def config_resource(*parts: str):
    path = Path(__file__).resolve().parent
    for part in parts:
        path = path.joinpath(part)
    return path.with_suffix(".yaml")


def load_config(*parts: str):
    from omegaconf import OmegaConf

    resource = config_resource(*parts)
    with resource.open("r", encoding="utf-8") as handle:
        return OmegaConf.load(handle)


@contextmanager
def config_path(*parts: str) -> Iterator[Path]:
    yield config_resource(*parts)
