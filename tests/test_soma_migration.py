"""Release gates for soma's migration to ``Model.embed_images_dense`` (#260)."""

from __future__ import annotations

import base64
import gc
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
import torch
from numpy.typing import NDArray

PIL = pytest.importorskip("PIL")

from PIL import Image  # noqa: E402

from slide2vec.api import (  # noqa: E402
    DenseImageOptions,
    ExecutionOptions,
    ImageSpec,
    Model,
)
from slide2vec.artifacts import DenseImageArtifact  # noqa: E402
from slide2vec.encoders.base import TileEncoder  # noqa: E402
from slide2vec.encoders.registry import (  # noqa: E402
    encoder_registry,
    register_encoder,
)
from slide2vec.runtime.dense_image_reading import (  # noqa: E402
    raster_read_plan,
    read_dense_image,
    resolve_spacing_read_plan,
)
from slide2vec.runtime.dense_image_recipe import DenseImageRecipe  # noqa: E402
from slide2vec.runtime.dense_image_shard import run_dense_image_shard  # noqa: E402
from slide2vec.runtime.dense_regions import compute_dense_geometry  # noqa: E402
from slide2vec.runtime.types import LoadedModel  # noqa: E402


_FIXTURES = Path(__file__).parent / "fixtures"
_REPOSITORY = Path(__file__).resolve().parents[1]
_LITERAL_ENCODER_NAME = "issue260-literal-identity"


def _literal_raster_pixels(offset: int = 0) -> NDArray[np.uint8]:
    pixels = np.empty((16, 16, 3), dtype=np.uint8)
    pixels[:8, :8] = [10 + offset, 11 + offset, 12 + offset]
    pixels[:8, 8:] = [20 + offset, 21 + offset, 22 + offset]
    pixels[8:, :8] = [30 + offset, 31 + offset, 32 + offset]
    pixels[8:, 8:] = [40 + offset, 41 + offset, 42 + offset]
    return pixels


def _write_literal_raster(path: Path, *, offset: int = 0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(_literal_raster_pixels(offset)).save(path)


class _LiteralIdentityEncoder(TileEncoder, torch.nn.Module):
    """Expose RGB values as a literal ``(3, height, width)`` artifact grid."""

    def __init__(self, *, output_variant: str | None = None) -> None:
        torch.nn.Module.__init__(self)
        self.register_buffer("_device_anchor", torch.empty(0))

    @property
    def encode_dim(self) -> int:
        return 3

    @property
    def device(self) -> torch.device:
        return self._device_anchor.device

    def to(self, device: torch.device | str) -> "_LiteralIdentityEncoder":
        torch.nn.Module.to(self, device)
        return self

    @property
    def patch_size(self) -> tuple[int, int]:
        return (1, 1)

    def get_transform(self):
        return _rgb_tensor

    def get_normalization_transform(self):
        return _rgb_tensor

    def encode_tiles(self, batch: torch.Tensor) -> torch.Tensor:
        return batch.mean(dim=(-1, -2))

    def encode_tiles_dense(self, batch: torch.Tensor) -> torch.Tensor:
        return batch


@pytest.fixture
def literal_encoder_registry(monkeypatch) -> str:
    """Register the oracle encoder only for one public-API test."""
    monkeypatch.setattr(
        encoder_registry, "_entries", dict(encoder_registry._entries)
    )
    register_encoder(
        _LITERAL_ENCODER_NAME,
        output_variants={"default": {"encode_dim": 3}},
        default_output_variant="default",
        input_size=16,
        supports_variable_input_size=True,
        patch_size=1,
        supported_spacing_um=[0.25, 0.5],
        default_spacing_um=0.25,
        precision="fp32",
        source="fixed issue #260 soma oracle",
    )(_LiteralIdentityEncoder)
    return _LITERAL_ENCODER_NAME


class _LiteralPatchEncoder(torch.nn.Module):
    """Return the mean RGB value of each non-overlapping 8×8 patch."""

    patch_size = (8, 8)

    def get_normalization_transform(self):
        return _rgb_tensor

    def encode_tiles_dense(self, batch):
        return batch.unfold(2, 8, 8).unfold(3, 8, 8).mean(dim=(-1, -2))


def _rgb_tensor(image: Image.Image) -> torch.Tensor:
    pixels = np.array(image, dtype=np.float32, copy=True)
    return torch.from_numpy(pixels).permute(2, 0, 1)


def _literal_loaded() -> LoadedModel:
    return LoadedModel(
        name="literal-patch-encoder",
        level="tile",
        model=_LiteralPatchEncoder(),
        transforms=_rgb_tensor,
        feature_dim=3,
        device=torch.device("cpu"),
    )


def _literal_recipe() -> DenseImageRecipe:
    geometry = compute_dense_geometry(target_size=16, patch_size=8)
    return DenseImageRecipe(
        encoder_name="literal-patch-encoder",
        output_variant="default",
        reader_regime="raster",
        spacing_source="explicit",
        declared_spacing_um=0.5,
        source_spacing_um=0.5,
        effective_spacing_um=0.5,
        requested_backend="auto",
        backend="pil",
        tolerance=None,
        read_level=None,
        read_tile_size_px=None,
        requested_tile_size_px=None,
        target_size=geometry.target_size,
        patch_size=geometry.patch_size,
        encoded_size=geometry.encoded_size,
        pad=geometry.pad,
        grid_shape=geometry.grid_shape,
        pad_mode="reflect",
        image_pad_value=None,
        window_size=None,
        overlap=0.0,
        feature_kind="patch_features",
        attention_blocks=(-1,),
        attention_include_registers=False,
        precision="fp32",
        dtype="float32",
    )


def _literal_dense() -> DenseImageOptions:
    return DenseImageOptions(target_size=16, spacing_um=0.5)


def test_fixed_raster_recipe_preserves_literal_pillow_rgb_pixels(tmp_path):
    path = tmp_path / "literal.png"
    expected = _literal_raster_pixels()
    Image.fromarray(expected).save(path)

    image = read_dense_image(
        ImageSpec(sample_id="literal", image_path=path),
        plan=raster_read_plan(
            spacing_source="explicit",
            declared_spacing_um=0.5,
            requested_backend="auto",
        ),
        target_size=(16, 16),
    )

    np.testing.assert_array_equal(np.asarray(image), expected)


def test_fixed_raster_recipe_preserves_literal_dense_grid(tmp_path):
    path = tmp_path / "literal.png"
    _write_literal_raster(path)
    artifact = run_dense_image_shard(
        [ImageSpec(sample_id="literal", image_path=path)],
        loaded=_literal_loaded(),
        out_dir=tmp_path / "out",
        dense=_literal_dense(),
        recipe=_literal_recipe(),
        batch_size=1,
        num_workers=0,
    )[0]

    grid = torch.load(artifact.path, weights_only=True)
    expected = torch.tensor(
        [
            [[10.0, 20.0], [30.0, 40.0]],
            [[11.0, 21.0], [31.0, 41.0]],
            [[12.0, 22.0], [32.0, 42.0]],
        ],
        dtype=torch.float32,
    )
    torch.testing.assert_close(grid, expected, rtol=0, atol=0)


def _soma_oracle(
    tmp_path: Path, case: str
) -> tuple[ImageSpec, dict[str, Any], NDArray[np.uint8]]:
    manifest = cast(
        dict[str, Any],
        json.loads(
            (_FIXTURES / "soma_dense_reader_oracle.json").read_text(
                encoding="utf-8"
            )
        ),
    )
    source = tmp_path / "soma-reader-oracle.tiff"
    source.write_bytes(
        base64.b64decode(
            (_FIXTURES / "soma_dense_reader_source.b64").read_text().strip()
        )
    )
    captured = cast(dict[str, Any], manifest[case])
    expected = np.frombuffer(
        base64.b64decode(captured["bytes_base64"]),
        dtype=np.dtype(captured["dtype"]),
    ).reshape(captured["shape"])
    return ImageSpec(sample_id=case, image_path=source), manifest, expected


def _read_against_soma_oracle(
    tmp_path: Path, case: str
) -> NDArray[np.uint8]:
    spec, manifest, expected = _soma_oracle(tmp_path, case)
    captured = cast(dict[str, Any], manifest[case])
    plan = resolve_spacing_read_plan(
        spec,
        requested_spacing_um=float(captured["requested_spacing_um"]),
        spacing_source="explicit",
        requested_backend=str(manifest["oracle"]["backend"]),
        tolerance=float(manifest["oracle"]["tolerance"]),
    )
    captured_shape = tuple(int(value) for value in captured["shape"])
    assert len(captured_shape) == 3
    image = read_dense_image(
        spec,
        plan=plan,
        target_size=(captured_shape[0], captured_shape[1]),
    )
    actual = np.asarray(image)
    np.testing.assert_array_equal(actual, expected)
    return actual


def _embed_soma_oracle_through_public_api(
    tmp_path: Path, case: str, encoder_name: str
) -> tuple[torch.Tensor, dict[str, Any]]:
    spec, manifest, _ = _soma_oracle(tmp_path, case)
    captured = cast(dict[str, Any], manifest[case])
    target_size = tuple(int(value) for value in captured["shape"][:2])
    assert len(target_size) == 2
    artifact = Model.from_preset(
        encoder_name, device="cpu"
    ).embed_images_dense(
        [spec],
        dense=DenseImageOptions(
            target_size=(target_size[0], target_size[1]),
            spacing_um=float(captured["requested_spacing_um"]),
            backend=str(manifest["oracle"]["backend"]),
            tolerance=float(manifest["oracle"]["tolerance"]),
        ),
        execution=ExecutionOptions(
            output_dir=tmp_path / f"{case}-public-api",
            num_gpus=1,
            batch_size=2,
            num_workers_per_gpu=0,
            precision="fp32",
            output_dtype="fp32",
        ),
    )[0]
    expected = cast(dict[str, Any], captured["literal_artifact"])
    return torch.load(artifact.path, weights_only=True), expected


def _assert_literal_artifact(
    actual: torch.Tensor, expected: dict[str, Any]
) -> None:
    assert list(actual.shape) == expected["shape"]
    assert str(actual.dtype).removeprefix("torch.") == expected["dtype"]
    assert (
        hashlib.sha256(actual.numpy().tobytes(order="C")).hexdigest()
        == expected["sha256"]
    )


def test_native_hs2p_read_is_byte_identical_to_soma_1_8_0_oracle(tmp_path):
    actual = _read_against_soma_oracle(tmp_path, "native")

    assert actual.shape == (16, 16, 3)


def test_area_hs2p_read_is_byte_identical_to_soma_1_8_0_oracle(tmp_path):
    actual = _read_against_soma_oracle(tmp_path, "area_downsampled")

    assert actual.shape == (8, 8, 3)


def test_public_num_gpus_one_native_artifact_matches_soma_literal_oracle(
    tmp_path, literal_encoder_registry
):
    actual, expected = _embed_soma_oracle_through_public_api(
        tmp_path, "native", literal_encoder_registry
    )

    _assert_literal_artifact(actual, expected)


def test_public_num_gpus_one_area_artifact_matches_soma_literal_oracle(
    tmp_path, literal_encoder_registry
):
    actual, expected = _embed_soma_oracle_through_public_api(
        tmp_path, "area_downsampled", literal_encoder_registry
    )

    _assert_literal_artifact(actual, expected)


def _metadata(artifact: DenseImageArtifact) -> dict[str, Any]:
    return cast(
        dict[str, Any],
        json.loads(artifact.metadata_path.read_text(encoding="utf-8")),
    )


def test_dense_image_glossary_names_the_release_contract_terms():
    glossary = (_REPOSITORY / "docs" / "glossary.rst").read_text(
        encoding="utf-8"
    ).lower()

    for term in (
        "raster image",
        "spacing-readable image",
        "source-spacing declaration",
        "source spacing",
        "declared spacing",
        "effective spacing",
        "target size",
        "compatible artifact",
    ):
        assert term in glossary


def test_migration_guidance_lives_only_in_5_6_0_release_notes():
    release_notes_path = _REPOSITORY / "docs" / "release-notes" / "5.6.0.rst"
    release_notes = release_notes_path.read_text(encoding="utf-8")
    for required in (
        "soma",
        "Model.embed_images_dense",
        "compatible artifact",
        "resume",
        "provenance",
    ):
        assert required in release_notes

    evergreen = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (_REPOSITORY / "docs").rglob("*.rst")
        if "release-notes" not in path.parts
    )
    assert "replace its private dense image reader" not in evergreen
    assert "Artifacts written with hs2p 4.3 remain loadable" not in evergreen


@dataclass(frozen=True)
class _GpuParityRuns:
    one_artifacts: list[DenseImageArtifact]
    two_artifacts: list[DenseImageArtifact]
    one_grids: list[torch.Tensor]
    two_grids: list[torch.Tensor]


@pytest.fixture(scope="module")
def gpu_parity_runs(tmp_path_factory) -> _GpuParityRuns:
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        pytest.skip("the GPU release gate requires at least two visible CUDA devices")

    tmp_path = tmp_path_factory.mktemp("gpu-parity")
    specs = []
    y, x = np.indices((224, 224), dtype=np.uint16)
    for index, sample_id in enumerate(("a", "b", "c", "d")):
        pixels = np.stack(
            (
                (x + 3 * y + 11 * index) % 256,
                (7 * x + y + 13 * index) % 256,
                (5 * x + 9 * y + 17 * index) % 256,
            ),
            axis=-1,
        ).astype(np.uint8)
        path = tmp_path / "gpu-inputs" / f"{sample_id}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(pixels).save(path)
        specs.append(
            ImageSpec(
                sample_id=sample_id,
                image_path=path,
                spacing_at_level_0=0.5,
            )
        )

    dense = DenseImageOptions(target_size=224, spacing_um=0.5)
    one_artifacts = Model.from_preset("lunit", device="cuda:0").embed_images_dense(
        specs,
        dense=dense,
        execution=ExecutionOptions(
            output_dir=tmp_path / "one-gpu",
            num_gpus=1,
            batch_size=2,
            num_workers_per_gpu=0,
            precision="fp32",
            output_dtype="fp32",
        ),
    )
    gc.collect()
    torch.cuda.empty_cache()
    two_artifacts = Model.from_preset("lunit").embed_images_dense(
        specs,
        dense=dense,
        execution=ExecutionOptions(
            output_dir=tmp_path / "two-gpu",
            num_gpus=2,
            batch_size=2,
            num_workers_per_gpu=0,
            precision="fp32",
            output_dtype="fp32",
        ),
    )

    return _GpuParityRuns(
        one_artifacts=one_artifacts,
        two_artifacts=two_artifacts,
        one_grids=[
            torch.load(artifact.path, weights_only=True) for artifact in one_artifacts
        ],
        two_grids=[
            torch.load(artifact.path, weights_only=True) for artifact in two_artifacts
        ],
    )


@pytest.mark.gpu_integration
def test_gpu_parity_preserves_literal_sample_order(gpu_parity_runs):
    expected = ["a", "b", "c", "d"]

    assert [
        artifact.sample_id for artifact in gpu_parity_runs.one_artifacts
    ] == expected
    assert [
        artifact.sample_id for artifact in gpu_parity_runs.two_artifacts
    ] == expected


@pytest.mark.gpu_integration
def test_gpu_parity_preserves_literal_grid_shapes(gpu_parity_runs):
    expected = [(384, 28, 28)] * 4

    assert [tuple(grid.shape) for grid in gpu_parity_runs.one_grids] == expected
    assert [tuple(grid.shape) for grid in gpu_parity_runs.two_grids] == expected


@pytest.mark.gpu_integration
def test_gpu_parity_preserves_literal_grid_dtypes(gpu_parity_runs):
    expected = [torch.float32] * 4

    assert [grid.dtype for grid in gpu_parity_runs.one_grids] == expected
    assert [grid.dtype for grid in gpu_parity_runs.two_grids] == expected


def _normalized_gpu_metadata(artifact: DenseImageArtifact) -> dict[str, Any]:
    metadata = _metadata(artifact)
    metadata["sample_id"] = "<SAMPLE_ID>"
    metadata["image_path"] = "<IMAGE_PATH>"
    compatibility = cast(dict[str, Any], metadata["compatibility"])
    compatibility["sample_id"] = "<SAMPLE_ID>"
    compatibility["image_path"] = "<IMAGE_PATH>"
    return metadata


@pytest.mark.gpu_integration
def test_gpu_parity_preserves_literal_metadata(gpu_parity_runs):
    expected = json.loads(
        (_FIXTURES / "gpu_dense_artifact_metadata.json").read_text(encoding="utf-8")
    )
    one_metadata = [
        _metadata(artifact) for artifact in gpu_parity_runs.one_artifacts
    ]
    two_metadata = [
        _metadata(artifact) for artifact in gpu_parity_runs.two_artifacts
    ]
    expected_identity = [
        (sample_id, f"{sample_id}.png", sample_id, f"{sample_id}.png")
        for sample_id in ("a", "b", "c", "d")
    ]

    assert one_metadata == two_metadata
    assert [
        (
            metadata["sample_id"],
            Path(metadata["image_path"]).name,
            metadata["compatibility"]["sample_id"],
            Path(metadata["compatibility"]["image_path"]).name,
        )
        for metadata in one_metadata
    ] == expected_identity
    assert [
        _normalized_gpu_metadata(artifact)
        for artifact in gpu_parity_runs.one_artifacts
    ] == [expected] * 4
    assert [
        _normalized_gpu_metadata(artifact)
        for artifact in gpu_parity_runs.two_artifacts
    ] == [expected] * 4


@pytest.mark.gpu_integration
def test_gpu_parity_outputs_are_finite(gpu_parity_runs):
    assert all(torch.isfinite(grid).all() for grid in gpu_parity_runs.one_grids)
    assert all(torch.isfinite(grid).all() for grid in gpu_parity_runs.two_grids)


@pytest.mark.gpu_integration
def test_gpu_parity_cosine_is_at_least_point_9999(gpu_parity_runs):
    similarities = [
        torch.nn.functional.cosine_similarity(
            one.float().reshape(-1), two.float().reshape(-1), dim=0
        )
        for one, two in zip(
            gpu_parity_runs.one_grids, gpu_parity_runs.two_grids
        )
    ]

    assert all(float(similarity) >= 0.9999 for similarity in similarities)
