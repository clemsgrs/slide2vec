"""Installed encoder plugins are discovered through the public model API."""

import os
from pathlib import Path
import subprocess
import sys
import textwrap


PLUGIN_MODULE = """
import os
from pathlib import Path
import time

import torch

from slide2vec.encoders import TileEncoder, register_encoder


def register_presets():
    discovered = Path(os.environ["PLUGIN_DISCOVERED_SENTINEL"])
    with discovered.open("x") as sentinel:
        sentinel.write("discovered")

    # A provider may use public registry reads while it registers. Built-ins must
    # already be visible, and the in-progress provider must not recursively run.
    from slide2vec import list_models
    assert "virchow2" in list_models()
    assert "private-alpha" not in list_models()
    time.sleep(0.1)

    @register_encoder(
        "private-alpha",
        output_variants={"default": {"encode_dim": 3}},
        default_output_variant="default",
        input_size=224,
        supports_variable_input_size=False,
        supported_spacing_um=0.5,
        precision="fp32",
        source="/models/private-alpha.pt",
    )
    class PrivateAlpha(TileEncoder):
        def __init__(self, *, output_variant=None):
            Path(os.environ["PLUGIN_CONSTRUCTED_SENTINEL"]).write_text("constructed")
            self._device = torch.device("cpu")

        @property
        def encode_dim(self):
            return 3

        @property
        def device(self):
            return self._device

        def to(self, device):
            self._device = torch.device(device)
            return self

        def get_transform(self):
            return lambda image: image

        def encode_tiles(self, batch):
            return batch[:, :3]

    @register_encoder(
        "private-beta",
        output_variants={"embedding": {"encode_dim": 5}},
        default_output_variant="embedding",
        input_size=256,
        supports_variable_input_size=True,
        variable_input_model_kwargs={"dynamic_img_size": True},
        supported_spacing_um=[0.25, 0.5],
        default_spacing_um=0.5,
        precision="fp16",
        source="private-org/private-beta",
    )
    class PrivateBeta(PrivateAlpha):
        @property
        def encode_dim(self):
            return 5
"""


PLUGIN_PYPROJECT = """
[build-system]
requires = ["setuptools"]
build-backend = "setuptools.build_meta"

[project]
name = "slide2vec-private-encoders"
version = "1.0.0"
requires-python = ">=3.10"

[project.entry-points."slide2vec.encoders"]
private = "private_encoder_plugin:register_presets"

[tool.setuptools]
py-modules = ["private_encoder_plugin"]
"""


PLUGIN_CONSUMER_SCRIPT = """
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import socket
from threading import Barrier

import slide2vec
from slide2vec import Model, list_models
from slide2vec.encoders import encoder_registry

discovered = Path(os.environ["PLUGIN_DISCOVERED_SENTINEL"])
constructed = Path(os.environ["PLUGIN_CONSTRUCTED_SENTINEL"])
assert not discovered.exists()
assert not constructed.exists()

def reject_network(*args, **kwargs):
    raise AssertionError("encoder discovery attempted network access")

socket.socket = reject_network

expected_models = [
    "conch", "conchv15", "dinov2-vitb14", "genbio-pathfm", "gigapath",
    "gigapath-slide", "gpfm", "h-optimus-0", "h-optimus-1", "h0-mini",
    "hibou-b", "hibou-l", "isight", "lunit", "mascaret", "midnight",
    "moozy", "moozy-slide", "mstar", "musk", "phaet", "phikon", "phikonv2",
    "prism", "prism2", "private-alpha", "private-beta", "prost40m", "rudolfv2",
    "rudolfv2-b", "rudolfv2-s", "titan", "uni", "uni2", "virchow", "virchow2",
]
barrier = Barrier(8)
def list_after_barrier():
    barrier.wait()
    return list_models()

with ThreadPoolExecutor(max_workers=8) as executor:
    listings = list(executor.map(lambda _: list_after_barrier(), range(8)))

assert listings == [expected_models] * 8
assert discovered.read_text() == "discovered"
assert not constructed.exists()

assert encoder_registry.info("private-beta") == {
    "name": "private-beta",
    "output_variants": {"embedding": {"encode_dim": 5}},
    "default_output_variant": "embedding",
    "level": "tile",
    "input_size": 256,
    "supports_variable_input_size": True,
    "variable_input_model_kwargs": {"dynamic_img_size": True},
    "patch_size": None,
    "tile_encoder": None,
    "tile_encoder_output_variant": None,
    "supported_spacing_um": [0.25, 0.5],
    "default_spacing_um": 0.5,
    "precision": "fp16",
    "source": "private-org/private-beta",
}

model = Model.from_preset("private-alpha", device="cpu")
assert model.name == "private-alpha"
assert model.level == "tile"
assert not constructed.exists()
assert model.feature_dim == 3
assert str(model.device) == "cpu"
assert constructed.read_text() == "constructed"

# Repeated public access is process-global and idempotent: the provider would fail
# on duplicate preset names if slide2vec invoked it a second time.
assert list_models() == expected_models
"""


def test_installed_distribution_contributes_presets_to_public_model_api(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.setenv("HF_TOKEN", "ci-token-must-not-reach-plugin-consumer")
    plugin = tmp_path / "plugin"
    plugin.mkdir()
    (plugin / "pyproject.toml").write_text(textwrap.dedent(PLUGIN_PYPROJECT))
    (plugin / "private_encoder_plugin.py").write_text(textwrap.dedent(PLUGIN_MODULE))

    target = tmp_path / "site-packages"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--quiet",
            "--no-deps",
            "--no-build-isolation",
            "--target",
            str(target),
            str(plugin),
        ],
        check=True,
    )

    env = os.environ.copy()
    env.pop("HF_TOKEN", None)
    repository_root = Path(__file__).resolve().parents[1]
    env["PYTHONPATH"] = os.pathsep.join((str(target), str(repository_root)))
    env["PLUGIN_DISCOVERED_SENTINEL"] = str(tmp_path / "discovered")
    env["PLUGIN_CONSTRUCTED_SENTINEL"] = str(tmp_path / "constructed")
    subprocess.run(
        [sys.executable, "-c", textwrap.dedent(PLUGIN_CONSUMER_SCRIPT)],
        cwd=tmp_path,
        env=env,
        check=True,
    )
