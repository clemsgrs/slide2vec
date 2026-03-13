import pathlib

from omegaconf import OmegaConf


CONFIG_ROOT = pathlib.Path(__file__).parent.resolve()


def load_config(*parts: str):
    config_path = CONFIG_ROOT.joinpath(*parts).with_suffix(".yaml")
    return OmegaConf.load(config_path)


default_preprocessing_config = load_config("preprocessing", "default")
default_model_config = load_config("models", "default")
