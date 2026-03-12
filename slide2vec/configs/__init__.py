import pathlib

from omegaconf import OmegaConf


CONFIG_ROOT = pathlib.Path(__file__).parent.resolve()


def load_config(*parts: str):
    config_path = CONFIG_ROOT.joinpath(*parts).with_suffix(".yaml")
    return OmegaConf.load(config_path)


default_preprocessing_config = load_config("preprocessing", "default")
default_model_config = load_config("models", "default")


def load_and_merge_config(config_name: str):
    default_preprocessing_cfg = OmegaConf.create(default_preprocessing_config)
    default_model_cfg = OmegaConf.create(default_model_config)
    default_config = OmegaConf.merge(default_preprocessing_cfg, default_model_cfg)
    loaded_config = load_config("models", config_name)
    return OmegaConf.merge(default_config, loaded_config)
