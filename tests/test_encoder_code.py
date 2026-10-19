"""The encoder code block the feature identity records under ``encoder_code``.

Ways it can go wrong, each covered below:

* computing the block instantiates the encoder or loads its weights;
* an edit to the file that defines the encoder leaves the block, or the comparison of
  a recorded identity, unchanged;
* an installed encoder records its file's digest instead of its distribution's version,
  or a namespace package shared by several distributions is attributed by its
  top-level name instead of by the ``RECORD`` that lists the file;
* an editable install, or a file slide2vec itself installed, records a version that
  does not change with the file;
* a class with no file (a notebook cell) raises instead of recording no fingerprint.

The installed-plugin path (a real ``pip install``) is covered in ``test_encoder_plugins.py``.
"""

import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import sys
import types

from PIL import Image
import pytest

from slide2vec import ExecutionOptions, ImageSpec, Model
from slide2vec.artifacts import load_metadata
from slide2vec.encoders import encoder_code

ENCODER_MODULE = '''
from slide2vec import TorchTileEncoder, register_encoder


@register_encoder(
    "{name}",
    encode_dim=8,
    input_size=32,
    supports_variable_input_size=False,
    supported_spacing_um=0.5,
    precision="fp32",
    source="/models/{name}.pt",
)
class CustomEncoder(TorchTileEncoder):
    def __init__(self, *, output_variant=None):
        raise AssertionError("encoder_code must not instantiate the encoder")
'''


WORKING_ENCODER_MODULE = '''
import torch

from slide2vec import TorchTileEncoder, register_encoder


@register_encoder(
    "{name}",
    encode_dim=3,
    input_size=32,
    supports_variable_input_size=False,
    supported_spacing_um=0.5,
    precision="fp32",
    source="/models/{name}.pt",
)
class CustomEncoder(TorchTileEncoder):
    def __init__(self, *, output_variant=None):
        model = torch.nn.Sequential(torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten())
        super().__init__(
            model, encode_dim=3, input_size=32, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
            output_variant=output_variant,
        )
'''


def _import_module_file(path: Path, module_name: str, monkeypatch) -> None:
    """Import *path* as *module_name*, as a script importing its encoder module does."""
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_encoder_code_of_a_module_file_encoder_names_its_class_file_digest_and_source(
    tmp_path, monkeypatch, isolated_encoder_registry
):
    path = tmp_path / "custom_encoders.py"
    path.write_text(ENCODER_MODULE.format(name="custom-code"), encoding="utf-8")
    _import_module_file(path, "custom_encoders", monkeypatch)

    assert encoder_code("custom-code") == {
        "object": "custom_encoders.CustomEncoder",
        "fingerprint": _sha256(path),
        "source": "/models/custom-code.pt",
    }


def test_an_edit_to_a_module_file_encoder_is_a_feature_identity_difference(
    tmp_path, monkeypatch, isolated_encoder_registry
):
    path = tmp_path / "edited_encoders.py"
    path.write_text(WORKING_ENCODER_MODULE.format(name="edited-code"), encoding="utf-8")
    _import_module_file(path, "edited_encoders", monkeypatch)
    image_path = tmp_path / "image.png"
    Image.new("RGB", (48, 48), color=(255, 255, 255)).save(image_path)
    model = Model.from_preset("edited-code", device="cpu")
    execution = ExecutionOptions(
        output_dir=tmp_path / "out", precision="fp32", num_gpus=1, num_workers_per_gpu=0
    )
    (artifact,) = model.embed_images(
        [ImageSpec(sample_id="image", image_path=image_path)], execution=execution
    )
    recorded = load_metadata(artifact.metadata_path)["compatibility"]
    recorded_code = {
        "object": "edited_encoders.CustomEncoder",
        "fingerprint": _sha256(path),
        "source": "/models/edited-code.pt",
    }

    assert recorded["encoder_code"] == recorded_code
    assert model.pooled_identity_differences(
        recorded, preprocessing=None, execution=execution
    ) == {}

    path.write_text(path.read_text(encoding="utf-8") + "\n# An edit.\n", encoding="utf-8")

    assert model.pooled_identity_differences(
        recorded, preprocessing=None, execution=execution
    ) == {"encoder_code": (recorded_code, {**recorded_code, "fingerprint": _sha256(path)})}


#: A namespace package (no ``__init__.py``) that two installed distributions share, as
#: ``protobuf`` and ``googleapis-common-protos`` share ``google``.
NAMESPACE = "slide2vec_test_namespace"


def _install(site: Path, distribution: str, version: str, module: str, *, direct_url=None):
    """Install ``<NAMESPACE>/<module>.py`` into *site* as *distribution*, listed in its RECORD."""
    package = site / NAMESPACE
    package.mkdir(parents=True, exist_ok=True)
    (package / f"{module}.py").write_text(
        ENCODER_MODULE.format(name=f"installed-{module}"), encoding="utf-8"
    )
    info = site / f"{distribution.replace('-', '_')}-{version}.dist-info"
    info.mkdir()
    (info / "METADATA").write_text(
        f"Metadata-Version: 2.1\nName: {distribution}\nVersion: {version}\n", encoding="utf-8"
    )
    (info / "RECORD").write_text(
        f"{NAMESPACE}/{module}.py,,\n{info.name}/METADATA,,\n{info.name}/RECORD,,\n",
        encoding="utf-8",
    )
    if direct_url is not None:
        (info / "direct_url.json").write_text(json.dumps(direct_url), encoding="utf-8")


@pytest.fixture
def site(tmp_path, monkeypatch, isolated_encoder_registry):
    """An extra site directory on ``sys.path``; its namespace modules are unloaded after."""
    site = tmp_path / "site-packages"
    site.mkdir()
    monkeypatch.syspath_prepend(str(site))
    yield site
    for name in [name for name in sys.modules if name.split(".", 1)[0] == NAMESPACE]:
        sys.modules.pop(name)


def _import_installed(module: str) -> None:
    importlib.import_module(f"{NAMESPACE}.{module}")


def test_encoder_code_of_an_installed_encoder_is_the_version_of_the_distribution_listing_it(site):
    """Distributions of a namespace package share its top-level name: their RECORD decides."""
    _install(site, "slide2vec-test-alpha", "1.0", "alpha")
    _install(site, "slide2vec-test-beta", "2.0", "beta")
    (site / NAMESPACE / "gamma.py").write_text(
        ENCODER_MODULE.format(name="installed-gamma"), encoding="utf-8"
    )
    for module in ("alpha", "beta", "gamma"):
        _import_installed(module)

    assert {
        module: encoder_code(f"installed-{module}")["fingerprint"]
        for module in ("alpha", "beta", "gamma")
    } == {
        "alpha": "1.0",
        "beta": "2.0",
        # A file no distribution lists is not installed code: its digest pins it.
        "gamma": _sha256(site / NAMESPACE / "gamma.py"),
    }


def test_encoder_code_of_an_editable_install_is_the_digest_of_its_file(site):
    """An editable install runs the work tree: its version does not change with an edit."""
    _install(
        site, "slide2vec-test-editable", "3.0", "delta",
        direct_url={"url": site.as_uri(), "dir_info": {"editable": True}},
    )
    _import_installed("delta")

    assert encoder_code("installed-delta")["fingerprint"] == _sha256(site / NAMESPACE / "delta.py")


def test_encoder_code_of_a_file_slide2vec_installed_is_its_digest(site):
    """A built-in's identity changes when a release changes its file, not on every release."""
    _install(site, "slide2vec", "99.0", "epsilon")
    _import_installed("epsilon")

    assert encoder_code("installed-epsilon")["fingerprint"] == _sha256(
        site / NAMESPACE / "epsilon.py"
    )


def test_encoder_code_of_a_built_in_encoder_is_the_digest_of_its_file():
    import slide2vec.encoders.models.virchow as virchow

    assert encoder_code("virchow2") == {
        "object": "slide2vec.encoders.models.virchow.Virchow2",
        "fingerprint": _sha256(Path(virchow.__file__)),
        "source": "paige-ai/Virchow2",
    }


def test_encoder_code_of_an_encoder_with_no_file_records_no_fingerprint(
    monkeypatch, isolated_encoder_registry
):
    """A class defined in a notebook cell lives in a module that has no file."""
    module = types.ModuleType("notebook_cell")
    monkeypatch.setitem(sys.modules, "notebook_cell", module)
    exec(compile(ENCODER_MODULE.format(name="cell-code"), "<cell>", "exec"), module.__dict__)

    assert encoder_code("cell-code") == {
        "object": "notebook_cell.CustomEncoder",
        "fingerprint": None,
        "source": "/models/cell-code.pt",
    }
