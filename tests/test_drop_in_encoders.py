"""Bring-your-own-encoder paths.

A drop-in module under ``slide2vec/encoders/models/``, the ``TorchTileEncoder`` base and
the ``register_encoder(encode_dim=...)`` shorthand. Ways this can fail, each pinned below:

- a new module file in the models directory is not imported, so its preset is missing;
- a built-in subpackage (moozy) stops being imported once the explicit import list is gone;
- the drop-in preset is visible in the parent but not in a fresh worker interpreter;
- the base ships the wrong transform: given images are not resized to ``input_size``, or
  declared tiles are resized;
- a module handed over in train mode is encoded in train mode;
- ``to`` records the device but does not move the module, or does not return ``self``;
- the shorthand produces metadata that differs from the explicit ``output_variants`` form;
- the shorthand is accepted together with, or instead of, the explicit form without error;
- the public names are not importable from ``slide2vec``;
- a module that reads the registry at module scope completes discovery over the
  built-ins imported so far, or deadlocks a concurrent direct model import against the
  discovery owner.
"""

import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap
import uuid

import pytest
import torch
from PIL import Image

from slide2vec import TorchTileEncoder, list_models, register_encoder
from slide2vec.encoders import encoder_registry, resolve_encoder_capabilities


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = REPOSITORY_ROOT / "slide2vec" / "encoders" / "models"

DROP_IN_MODULE = """
import torch

from slide2vec import TorchTileEncoder, register_encoder


@register_encoder(
    "drop-in-test",
    encode_dim=8,
    input_size=32,
    supports_variable_input_size=False,
    supported_spacing_um=0.5,
    precision="fp32",
    source="tests/test_drop_in_encoders.py",
)
class DropInTest(TorchTileEncoder):
    def __init__(self, *, output_variant: str | None = None):
        torch.manual_seed(0)
        model = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(), torch.nn.Linear(3, 8)
        )
        super().__init__(
            model,
            encode_dim=8,
            input_size=32,
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5),
            output_variant=output_variant,
        )
"""

CONSUMER_SCRIPT = """
import json
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from slide2vec import (
    ExecutionOptions,
    ImageSpec,
    Model,
    TileEncoder,
    TimmTileEncoder,
    TorchTileEncoder,
    list_models,
    register_encoder,
)
from slide2vec.encoders import encoder_registry
from slide2vec.runtime.serialization import serialize_model

builtins = set(json.loads(os.environ["BUILTIN_PRESETS"]))
assert set(list_models()) == builtins | {"drop-in-test"}, sorted(set(list_models()) ^ builtins)

model = Model.from_preset("drop-in-test", device="cpu")
assert model.level == "tile"
assert model.feature_dim == 8

out = Path(os.environ["DROP_IN_OUTPUT_DIR"])
images = out / "images"
images.mkdir(parents=True)
rng = np.random.default_rng(0)
specs = []
for name in ("a", "b"):
    pixels = rng.integers(0, 256, size=(48, 64, 3), dtype=np.uint8)
    Image.fromarray(pixels).save(images / f"{name}.png")
    specs.append(ImageSpec(sample_id=name, image_path=images / f"{name}.png"))

artifacts = model.embed_images(
    specs, execution=ExecutionOptions(output_dir=out, num_gpus=1, precision="fp32")
)
assert [artifact.sample_id for artifact in artifacts] == ["a", "b"]

encoder = encoder_registry.require("drop-in-test")()
for spec, artifact in zip(specs, artifacts):
    image = Image.open(spec.image_path).convert("RGB")
    given = encoder.get_transform()(image)
    assert tuple(given.shape) == (3, 32, 32), given.shape
    declared = encoder.get_normalization_transform()(image)
    assert tuple(declared.shape) == (3, 48, 64), declared.shape
    with torch.no_grad():
        expected = encoder.encode_tiles(given[None])[0]
    saved = torch.load(artifact.path, weights_only=True)
    assert tuple(saved.shape) == (8,), saved.shape
    torch.testing.assert_close(saved.float(), expected.float())

request_path = out / "request.json"
request_path.write_text(json.dumps({"model": serialize_model(model)}))
Path(os.environ["DROP_IN_REPORT"]).write_text(json.dumps({
    "presets": sorted(list_models()),
    "registered_class": [encoder.__class__.__module__, encoder.__class__.__qualname__],
    "embeddings": sorted(path.name for path in (out / "image_embeddings").glob("*.pt")),
}))
"""

WORKER_SCRIPT = """
import json
import os
from pathlib import Path

from slide2vec.distributed.worker_entry import model_from_request
from slide2vec.encoders import encoder_registry


class CpuWorkerRank:
    device = "cpu"


request = json.loads(Path(os.environ["DROP_IN_REQUEST_PATH"]).read_text())
model = model_from_request(request, rank=CpuWorkerRank())
registered_class = encoder_registry.require(model.name)
Path(os.environ["DROP_IN_WORKER_REPORT"]).write_text(json.dumps({
    "name": model.name,
    "feature_dim": model.feature_dim,
    "registered_class": [registered_class.__module__, registered_class.__qualname__],
}))
"""


REENTRANT_MODULE = """
from slide2vec import list_models

PRESETS_AT_IMPORT = list_models()
"""

# A direct model import holds the package import lock while a registry read in another
# thread wants to import the same package; the module above then reads the registry from
# inside the import. Both threads must finish, each with the registry's refusal.
REENTRANT_SCRIPT = """
import threading

results = {}


def direct_import():
    try:
        from slide2vec.encoders.models.uni import UNI  # noqa: F401
    except BaseException as error:
        results["import"] = error
    else:
        results["import"] = None


def registry_read():
    from slide2vec import list_models

    try:
        list_models()
    except BaseException as error:
        results["read"] = error
    else:
        results["read"] = None


threads = [
    threading.Thread(target=direct_import, daemon=True),
    threading.Thread(target=registry_read, daemon=True),
]
for thread in threads:
    thread.start()
for thread in threads:
    thread.join(timeout=120)
assert not any(thread.is_alive() for thread in threads), "a thread is still blocked"
for key in ("import", "read"):
    error = results[key]
    assert isinstance(error, RuntimeError), (key, repr(error))
    assert "still importing" in str(error), (key, str(error))

# The refusal is not cached as a completed discovery over a partial registry.
from slide2vec import list_models

try:
    list_models()
except RuntimeError as error:
    assert "still importing" in str(error)
else:
    raise AssertionError("a module-scope registry read was accepted on a later read")
"""


def _run(script: str, *, env: dict, cwd: Path) -> None:
    subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        cwd=cwd,
        env=env,
        check=True,
        timeout=300,
    )


def _repository_env() -> dict:
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in (str(REPOSITORY_ROOT), env.get("PYTHONPATH")) if p
    )
    return env


def _remove_module(module_path: Path) -> None:
    module_path.unlink(missing_ok=True)
    for cached in (MODELS_DIR / "__pycache__").glob(f"{module_path.stem}.*.pyc"):
        cached.unlink()


def test_drop_in_module_registers_and_encodes_in_fresh_interpreters(tmp_path: Path):
    module_name = f"zz_drop_in_{uuid.uuid4().hex[:8]}"
    module_path = MODELS_DIR / f"{module_name}.py"
    builtin_presets = list_models()
    assert "drop-in-test" not in builtin_presets
    assert any(encoder_registry.info(name)["source"].startswith("moozy") or name.startswith("moozy") for name in builtin_presets)

    env = _repository_env()
    env["BUILTIN_PRESETS"] = json.dumps(builtin_presets)
    env["DROP_IN_OUTPUT_DIR"] = str(tmp_path / "out")
    env["DROP_IN_REPORT"] = str(tmp_path / "report.json")
    env["DROP_IN_REQUEST_PATH"] = str(tmp_path / "out" / "request.json")
    env["DROP_IN_WORKER_REPORT"] = str(tmp_path / "worker-report.json")

    module_path.write_text(textwrap.dedent(DROP_IN_MODULE))
    try:
        _run(CONSUMER_SCRIPT, env=env, cwd=tmp_path)
        _run(WORKER_SCRIPT, env=env, cwd=tmp_path)
    finally:
        _remove_module(module_path)

    expected_class = [f"slide2vec.encoders.models.{module_name}", "DropInTest"]
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["presets"] == sorted(builtin_presets + ["drop-in-test"])
    assert report["registered_class"] == expected_class
    assert report["embeddings"] == ["a.pt", "b.pt"]
    assert json.loads((tmp_path / "worker-report.json").read_text()) == {
        "name": "drop-in-test",
        "feature_dim": 8,
        "registered_class": expected_class,
    }


def test_module_scope_registry_read_is_refused_without_deadlock(tmp_path: Path):
    module_path = MODELS_DIR / f"zz_reentrant_{uuid.uuid4().hex[:8]}.py"
    module_path.write_text(textwrap.dedent(REENTRANT_MODULE))
    try:
        _run(REENTRANT_SCRIPT, env=_repository_env(), cwd=tmp_path)
    finally:
        _remove_module(module_path)


def test_underscore_modules_are_not_imported_as_encoders():
    import slide2vec.encoders.models as models

    assert all(not name.startswith("_") for name in models.__all__)
    assert "moozy" in models.__all__


class _Tiny(TorchTileEncoder):
    def __init__(self, *, output_variant: str | None = None, training: bool = False):
        torch.manual_seed(0)
        model = torch.nn.Sequential(torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(), torch.nn.Linear(3, 4))
        model.train(training)
        super().__init__(
            model,
            encode_dim=4,
            input_size=16,
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5),
            output_variant=output_variant,
        )


def test_torch_tile_encoder_owns_eval_device_and_transforms():
    encoder = _Tiny(training=True)
    assert encoder._model.training is False
    assert encoder.encode_dim == 4
    assert encoder.to("cpu") is encoder
    assert encoder.device == torch.device("cpu")
    assert next(encoder._model.parameters()).device == torch.device("cpu")

    image = Image.fromarray((torch.rand(24, 40, 3) * 255).to(torch.uint8).numpy())
    given = encoder.get_transform()(image)
    declared = encoder.get_normalization_transform()(image)
    assert tuple(given.shape) == (3, 16, 16)
    assert tuple(declared.shape) == (3, 24, 40)
    assert given.dtype == declared.dtype == torch.float32
    assert declared.min() >= -1.0 and declared.max() <= 1.0  # (x - 0.5) / 0.5 on [0, 1]
    with torch.no_grad():
        assert tuple(encoder.encode_tiles(declared[None]).shape) == (1, 4)


def test_register_encoder_encode_dim_shorthand_matches_explicit_output_variants(
    isolated_encoder_registry,
):
    common = dict(
        input_size=16,
        supports_variable_input_size=False,
        supported_spacing_um=0.5,
        precision="fp32",
        source="tests",
    )
    register_encoder("drop-in-shorthand", encode_dim=4, **common)(_Tiny)
    register_encoder(
        "drop-in-explicit",
        output_variants={"default": {"encode_dim": 4}},
        default_output_variant="default",
        **common,
    )(_Tiny)

    shorthand = {k: v for k, v in encoder_registry.info("drop-in-shorthand").items() if k != "name"}
    explicit = {k: v for k, v in encoder_registry.info("drop-in-explicit").items() if k != "name"}
    assert shorthand == explicit
    capabilities = resolve_encoder_capabilities("drop-in-shorthand")
    assert (capabilities.pooled, capabilities.dense, capabilities.attention) == (True, False, False)


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"encode_dim": 4, "output_variants": {"default": {"encode_dim": 4}}},
        {"encode_dim": 4, "default_output_variant": "default"},
        {"output_variants": {"default": {"encode_dim": 4}}},
        {"encode_dim": 0},
        {"encode_dim": 4.0},
    ],
)
def test_register_encoder_requires_exactly_one_output_declaration(kwargs):
    with pytest.raises(ValueError):
        register_encoder(
            "drop-in-invalid",
            input_size=16,
            supports_variable_input_size=False,
            supported_spacing_um=0.5,
            **kwargs,
        )(_Tiny)
    assert "drop-in-invalid" not in encoder_registry
