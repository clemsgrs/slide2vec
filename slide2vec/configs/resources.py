from pathlib import Path


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
