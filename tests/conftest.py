import importlib.util
import sys
import types


if importlib.util.find_spec("transformers") is None:
    transformers_module = types.ModuleType("transformers")
    transformers_module.__path__ = []

    image_processing_utils = types.ModuleType("transformers.image_processing_utils")

    class BaseImageProcessor:
        pass

    image_processing_utils.BaseImageProcessor = BaseImageProcessor
    transformers_module.image_processing_utils = image_processing_utils

    sys.modules.setdefault("transformers", transformers_module)
    sys.modules["transformers.image_processing_utils"] = image_processing_utils
