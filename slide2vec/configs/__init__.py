import pathlib

from omegaconf import OmegaConf


def load_config(config_name: str):
    config_filename = config_name + ".yaml"
    return OmegaConf.load(pathlib.Path(__file__).parent.resolve() / config_filename)


default_tiling_config = load_config("default_tiling")
default_model_config = load_config("default_model")


def load_and_merge_config(config_name: str):
    default_tiling_config = OmegaConf.create(default_tiling_config)
    default_model_config = OmegaConf.create(default_model_config)
    default_config = OmegaConf.merge(default_tiling_config, default_model_config)
    loaded_config = load_config(config_name)
    return OmegaConf.merge(default_config, loaded_config)