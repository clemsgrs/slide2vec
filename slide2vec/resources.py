from importlib.resources import as_file, files
from pathlib import Path
from typing import Iterator
from contextlib import contextmanager


def config_resource(*parts: str):
    return files("slide2vec").joinpath("configs", *parts).with_suffix(".yaml")


def load_config(*parts: str):
    from omegaconf import OmegaConf

    resource = config_resource(*parts)
    with resource.open("r", encoding="utf-8") as handle:
        return OmegaConf.load(handle)


@contextmanager
def config_path(*parts: str) -> Iterator[Path]:
    resource = config_resource(*parts)
    with as_file(resource) as resolved:
        yield resolved
