"""Tests for dense grids over pre-cropped images: the artifacts + the CPU loop (issue #235).

Two layers, both CPU-testable:

1. ``slide2vec.artifacts.write_dense_image`` — one ``(d, gh, gw)`` payload + one geometry
   sidecar per image, written atomically and sidecar-last, so the sidecar is an unambiguous
   done-marker (the same contract ``write_dense_region`` follows).
2. ``run_dense_image_shard`` — the device-agnostic encode+write loop each rank runs over its
   shard: decode each image, apply the encoder's **normalization-only** transform itemwise in
   the loader workers, stack, pad up to the patch multiple, encode through
   ``encode_dense_sliding``, persist.

The anchor test pins the persisted grid against the direct composition
``compute_dense_geometry`` + ``encode_dense_sliding`` — the equivalence downstream consumers
migrate onto — for both ``feature_kind`` values.
"""

from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
timm = pytest.importorskip("timm")
PIL = pytest.importorskip("PIL")

from PIL import Image  # noqa: E402

from slide2vec.api import DenseImageOptions, ImageSpec  # noqa: E402
from slide2vec.artifacts import dense_image_paths, write_dense_image  # noqa: E402
from slide2vec.encoders.base import TimmTileEncoder  # noqa: E402
from slide2vec.runtime.dense_image_shard import run_dense_image_shard  # noqa: E402
from slide2vec.runtime.dense_image_recipe import DenseImageRecipe  # noqa: E402
from slide2vec.runtime.dense_regions import (  # noqa: E402
    compute_dense_geometry,
    pad_image_to_encoded,
)
from slide2vec.runtime.dense_sliding import encode_dense_sliding  # noqa: E402
from slide2vec.runtime.sharding import plan_contiguous_shards  # noqa: E402
from slide2vec.runtime.types import LoadedModel  # noqa: E402


def _encoder() -> TimmTileEncoder:
    return TimmTileEncoder(
        "vit_tiny_patch16_224", pretrained=False, num_classes=0, dynamic_img_size=True
    )


def _loaded(encoder: TimmTileEncoder) -> LoadedModel:
    """A ``LoadedModel`` under a dense contract: the normalization-only transform."""
    return LoadedModel(
        name="fake-encoder",
        level="tile",
        model=encoder,
        transforms=encoder.get_normalization_transform(),
        feature_dim=int(encoder.encode_dim),
        device=torch.device("cpu"),
    )


def _write_image(path: Path, *, width: int, height: int) -> None:
    rng = np.random.default_rng(abs(hash((width, height, path.name))) % (2**32))
    pixels = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pixels).save(path)


def _spec(tmp_path, sample_id: str, *, width: int = 64, height: int = 64) -> ImageSpec:
    path = tmp_path / "images" / f"{sample_id}.png"
    _write_image(path, width=width, height=height)
    return ImageSpec(sample_id=sample_id, image_path=str(path))


def _dense(**kwargs) -> DenseImageOptions:
    return DenseImageOptions(**{"target_size": 64, **kwargs})


def _recipe(
    *,
    target_size=(64, 64),
    pad_mode="reflect",
    image_pad_value=None,
    window_size=None,
    overlap=0.0,
    feature_kind="patch_features",
    attention_blocks=(-1,),
    attention_include_registers=False,
    precision="fp32",
    dtype="float32",
) -> DenseImageRecipe:
    geometry = compute_dense_geometry(target_size=target_size, patch_size=(16, 16))
    return DenseImageRecipe(
        encoder_name="fake-encoder",
        output_variant="default",
        target_size=geometry.target_size,
        patch_size=geometry.patch_size,
        encoded_size=geometry.encoded_size,
        pad=geometry.pad,
        grid_shape=geometry.grid_shape,
        pad_mode=pad_mode,
        image_pad_value=image_pad_value,
        window_size=window_size,
        overlap=overlap,
        feature_kind=feature_kind,
        attention_blocks=attention_blocks,
        attention_include_registers=attention_include_registers,
        precision=precision,
        dtype=dtype,
    )


def _dense_dir(out_dir) -> Path:
    return out_dir / "dense_image_embeddings"


# --------------------------------------------------------------------------------------
# Layer 1: the artifact write (payload atomic, sidecar last)
# --------------------------------------------------------------------------------------


def test_dense_image_paths_rejects_sample_id_paths(tmp_path):
    with pytest.raises(ValueError, match="sample_id"):
        dense_image_paths(tmp_path, sample_id="/tmp/outside")


def test_dense_image_paths_rejects_symlink_escape(tmp_path):
    output_dir = tmp_path / "output"
    embeddings_dir = output_dir / "dense_image_embeddings"
    outside_dir = tmp_path / "outside"
    embeddings_dir.parent.mkdir(parents=True)
    outside_dir.mkdir()
    embeddings_dir.symlink_to(outside_dir, target_is_directory=True)

    with pytest.raises(ValueError, match="output_dir"):
        dense_image_paths(output_dir, sample_id="image-1")


def test_write_dense_image_does_not_publish_interrupted_sidecar(tmp_path, monkeypatch):
    """A crashed sidecar write leaves the payload but no done-marker."""
    original_write_text = Path.write_text

    def _interrupted_write(path, text, *, encoding):
        original_write_text(path, text[:1], encoding=encoding)
        raise OSError("simulated interrupted metadata write")

    monkeypatch.setattr(Path, "write_text", _interrupted_write)

    with pytest.raises(OSError, match="interrupted metadata write"):
        write_dense_image(
            torch.zeros((2, 2, 2)),
            output_dir=tmp_path,
            sample_id="image-1",
            metadata={"feature_dim": 2, "grid_shape": [2, 2]},
        )

    payload_path, sidecar_path = dense_image_paths(tmp_path, sample_id="image-1")
    assert payload_path.exists()
    assert not sidecar_path.exists()


# --------------------------------------------------------------------------------------
# Layer 2: run_dense_image_shard (device-agnostic encode+write loop)
# --------------------------------------------------------------------------------------


def test_run_dense_image_shard_writes_payload_and_sidecar_per_image(tmp_path):
    enc = _encoder()
    specs = [_spec(tmp_path, name) for name in ("a", "b", "c")]

    artifacts = run_dense_image_shard(
        specs,
        loaded=_loaded(enc),
        out_dir=tmp_path / "out",
        dense=_dense(),
        recipe=_recipe(),
        batch_size=2,
        num_workers=0,
    )

    assert [artifact.sample_id for artifact in artifacts] == ["a", "b", "c"]
    dense_dir = _dense_dir(tmp_path / "out")
    for artifact in artifacts:
        assert artifact.path == (dense_dir / f"{artifact.sample_id}.pt").resolve()
        assert artifact.metadata_path.exists()
        assert artifact.grid_shape == (4, 4)
        assert artifact.feature_dim == enc.encode_dim
        grid = torch.load(artifact.path, weights_only=True)
        assert tuple(grid.shape) == (enc.encode_dim, 4, 4)
    assert sorted(p.name for p in dense_dir.glob("*.pt")) == ["a.pt", "b.pt", "c.pt"]
    assert list(dense_dir.glob("*.tmp*")) == []


@pytest.mark.parametrize("feature_kind", ["patch_features", "cls_attention"])
@pytest.mark.parametrize("window_size", [None, 32])
def test_grid_matches_the_direct_dense_composition(tmp_path, feature_kind, window_size):
    """The equivalence downstream consumers migrate onto.

    A persisted grid is exactly ``compute_dense_geometry`` + the encoder's normalization
    transform + ``encode_dense_sliding``, at the same batch size and dtype — for the patch
    grid and the CLS-attention grid alike, whole-tile and sliding.
    """
    enc = _encoder()
    batch_size = 2
    specs = [_spec(tmp_path, f"image-{i}") for i in range(4)]

    artifacts = run_dense_image_shard(
        specs,
        loaded=_loaded(enc),
        out_dir=tmp_path / "out",
        dense=_dense(feature_kind=feature_kind, window_size=window_size),
        recipe=_recipe(feature_kind=feature_kind, window_size=window_size),
        batch_size=batch_size,
        num_workers=0,
    )

    geometry = compute_dense_geometry(target_size=64, patch_size=enc.patch_size)
    transform = enc.get_normalization_transform()
    encode_fn = (
        enc.encode_tiles_dense
        if feature_kind == "patch_features"
        else (lambda w: enc.encode_tiles_attention(w, blocks=(-1,), include_registers=False))
    )
    for start in range(0, len(specs), batch_size):
        chunk = specs[start : start + batch_size]
        items = []
        for spec in chunk:
            with Image.open(spec.image_path) as image:
                tensor = torch.as_tensor(transform(image.convert("RGB")))
            items.append(
                pad_image_to_encoded(
                    tensor, geometry, pad_mode="reflect", image_pad_value=None
                )
            )
        with torch.inference_mode():
            expected = encode_dense_sliding(
                enc,
                torch.stack(items),
                geometry=geometry,
                window_size=window_size,
                overlap=0.0,
                encode_fn=encode_fn,
            )
        for offset, artifact in enumerate(artifacts[start : start + batch_size]):
            persisted = torch.load(artifact.path, weights_only=True)
            torch.testing.assert_close(persisted, expected[offset], rtol=0, atol=0)


def test_run_dense_image_shard_sidecar_records_extraction_geometry(tmp_path):
    enc = _encoder()
    dense = _dense(target_size=60, window_size=32, overlap=0.25)
    specs = [_spec(tmp_path, "a", width=60, height=60)]

    artifacts = run_dense_image_shard(
        specs, loaded=_loaded(enc), out_dir=tmp_path / "out", dense=dense,
        recipe=_recipe(
            target_size=(60, 60), window_size=32, overlap=0.25
        ),
        batch_size=1, num_workers=0,
    )

    meta = json.loads(artifacts[0].metadata_path.read_text())
    assert meta == {
        "artifact_type": "dense_image_embeddings",
        "sample_id": "a",
        "image_path": str(specs[0].image_path),
        "format": "pt",
        "dtype": "float32",
        "feature_dim": enc.encode_dim,
        "grid_shape": [4, 4],          # 60 padded up to 64 → 4×4 tokens
        "target_size": [60, 60],
        "patch_size": [16, 16],
        "encoded_size": [64, 64],
        "pad": [4, 4],
        "encoder_name": "fake-encoder",
        "encoder_level": "tile",
        "encoder_input_regime": "declared",
        "pad_mode": "reflect",
        "image_pad_value": None,
        "window_size": 32,
        "overlap": 0.25,
        "feature_kind": "patch_features",
        "attention_blocks": [-1],
        "attention_include_registers": False,
        "compatibility": {
            "sample_id": "a",
            "image_path": str(Path(specs[0].image_path).resolve()),
            "encoder_name": "fake-encoder",
            "output_variant": "default",
            "target_size": [60, 60],
            "patch_size": [16, 16],
            "encoded_size": [64, 64],
            "pad": [4, 4],
            "grid_shape": [4, 4],
            "pad_mode": "reflect",
            "image_pad_value": None,
            "window_size": 32,
            "overlap": 0.25,
            "feature_kind": "patch_features",
            "attention_blocks": [-1],
            "attention_include_registers": False,
            "precision": "fp32",
            "dtype": "float32",
        },
    }


def test_run_dense_image_shard_supports_non_square_images(tmp_path):
    """The image *is* the region here, so a non-square declared geometry must work.

    Nothing on this path narrows the encoder input to a square: the dense contract checks
    each dimension against the patch geometry, and the square-only recorder the pooled
    Given path uses is never reached (dense never calls ``encode_tiles``).
    """
    enc = _encoder()
    specs = [_spec(tmp_path, "wide", width=96, height=64)]

    artifacts = run_dense_image_shard(
        specs,
        loaded=_loaded(enc),
        out_dir=tmp_path / "out",
        dense=_dense(target_size=(64, 96)),
        recipe=_recipe(target_size=(64, 96)),
        batch_size=1,
        num_workers=0,
    )

    assert artifacts[0].grid_shape == (4, 6)
    grid = torch.load(artifacts[0].path, weights_only=True)
    assert tuple(grid.shape) == (enc.encode_dim, 4, 6)
    meta = json.loads(artifacts[0].metadata_path.read_text())
    assert meta["target_size"] == [64, 96]
    assert meta["encoded_size"] == [64, 96]


def test_run_dense_image_shard_rejects_images_that_are_not_the_declared_size(tmp_path):
    """The declared geometry is the contract: an off-size image is an error, not a resize."""
    enc = _encoder()
    specs = [_spec(tmp_path, "small", width=32, height=32)]

    with pytest.raises(ValueError, match=r"'small': \(32, 32\)"):
        run_dense_image_shard(
            specs, loaded=_loaded(enc), out_dir=tmp_path / "out", dense=_dense(),
            recipe=_recipe(),
            batch_size=1, num_workers=0,
        )


def test_off_size_images_are_named_even_when_a_batch_is_mixed(tmp_path):
    """The error has to name the images, not report a list of shapes from ``torch.stack``.

    A mixed-size batch is the realistic failure: the stack would blow up first with no way
    back to a sample id, so the declared geometry is checked per item before stacking.
    """
    enc = _encoder()
    specs = [
        _spec(tmp_path, "right", width=64, height=64),
        _spec(tmp_path, "short", width=64, height=48),
        _spec(tmp_path, "narrow", width=32, height=64),
    ]

    with pytest.raises(ValueError) as excinfo:
        run_dense_image_shard(
            specs, loaded=_loaded(enc), out_dir=tmp_path / "out", dense=_dense(),
            recipe=_recipe(),
            batch_size=3, num_workers=0,
        )

    message = str(excinfo.value)
    assert "'short': (48, 64)" in message and "'narrow': (64, 32)" in message
    assert "right" not in message  # the conforming image is not accused


def test_run_dense_image_shard_skips_images_with_existing_sidecar(tmp_path):
    """Resume: an image whose sidecar exists is not re-decoded or re-encoded."""
    enc = _encoder()
    specs = [_spec(tmp_path, name) for name in ("a", "b", "c")]
    out_dir = tmp_path / "out"

    first = run_dense_image_shard(
        specs, loaded=_loaded(enc), out_dir=out_dir, dense=_dense(),
        recipe=_recipe(), batch_size=2, num_workers=0
    )
    mtimes = {a.sample_id: a.path.stat().st_mtime_ns for a in first}

    second = run_dense_image_shard(
        specs, loaded=_loaded(enc), out_dir=out_dir, dense=_dense(),
        recipe=_recipe(), batch_size=2, num_workers=0
    )

    assert [a.sample_id for a in second] == [a.sample_id for a in first]
    assert {a.sample_id: a.path.stat().st_mtime_ns for a in second} == mtimes


def test_run_dense_image_shard_reencodes_payload_missing_its_sidecar(tmp_path):
    """Crash-safety: a payload with no sidecar is incomplete and is re-encoded."""
    enc = _encoder()
    specs = [_spec(tmp_path, name) for name in ("a", "b")]
    out_dir = tmp_path / "out"
    run_dense_image_shard(
        specs, loaded=_loaded(enc), out_dir=out_dir, dense=_dense(),
        recipe=_recipe(), batch_size=2, num_workers=0
    )

    _, sidecar_path = dense_image_paths(out_dir, sample_id="b")
    payload_path, _ = dense_image_paths(out_dir, sample_id="b")
    sidecar_path.unlink()
    payload_mtime = payload_path.stat().st_mtime_ns

    run_dense_image_shard(
        specs, loaded=_loaded(enc), out_dir=out_dir, dense=_dense(),
        recipe=_recipe(), batch_size=2, num_workers=0
    )

    assert sidecar_path.exists()
    assert payload_path.stat().st_mtime_ns != payload_mtime


def test_incompatible_recompute_invalidates_done_marker_before_encoding(tmp_path):
    enc = _encoder()
    spec = _spec(tmp_path, "a")
    out_dir = tmp_path / "out"
    original_recipe = _recipe(overlap=0.0)
    run_dense_image_shard(
        [spec],
        loaded=_loaded(enc),
        out_dir=out_dir,
        dense=_dense(),
        recipe=original_recipe,
        batch_size=1,
        num_workers=0,
    )
    payload_path, sidecar_path = dense_image_paths(out_dir, sample_id="a")
    original_payload = payload_path.read_bytes()

    Path(spec.image_path).unlink()
    with pytest.raises(FileNotFoundError):
        run_dense_image_shard(
            [spec],
            loaded=_loaded(enc),
            out_dir=out_dir,
            dense=_dense(overlap=0.25),
            recipe=replace(original_recipe, overlap=0.25),
            batch_size=1,
            num_workers=0,
        )

    assert payload_path.read_bytes() == original_payload
    assert not sidecar_path.exists()


def test_interruption_after_payload_replacement_leaves_no_trusted_sidecar(
    tmp_path, monkeypatch
):
    enc = _encoder()
    spec = _spec(tmp_path, "a")
    out_dir = tmp_path / "out"
    original_recipe = _recipe(dtype="float32")
    run_dense_image_shard(
        [spec],
        loaded=_loaded(enc),
        out_dir=out_dir,
        dense=_dense(),
        recipe=original_recipe,
        batch_size=1,
        num_workers=0,
    )
    payload_path, sidecar_path = dense_image_paths(out_dir, sample_id="a")
    assert torch.load(payload_path, weights_only=True).dtype == torch.float32

    def _fail_sidecar(*args, **kwargs):
        raise RuntimeError("failure before sidecar publication")

    monkeypatch.setattr(Path, "write_text", _fail_sidecar)
    replacement_recipe = replace(original_recipe, dtype="float16")
    with pytest.raises(RuntimeError, match="failure before sidecar publication"):
        run_dense_image_shard(
            [spec],
            loaded=_loaded(enc),
            out_dir=out_dir,
            dense=_dense(),
            recipe=replacement_recipe,
            batch_size=1,
            output_dtype=torch.float16,
            num_workers=0,
        )

    assert torch.load(payload_path, weights_only=True).dtype == torch.float16
    assert not sidecar_path.exists()


def test_payload_temp_write_failure_leaves_old_payload_without_sidecar(
    tmp_path, monkeypatch
):
    enc = _encoder()
    spec = _spec(tmp_path, "a")
    out_dir = tmp_path / "out"
    original_recipe = _recipe(dtype="float32")
    run_dense_image_shard(
        [spec],
        loaded=_loaded(enc),
        out_dir=out_dir,
        dense=_dense(),
        recipe=original_recipe,
        batch_size=1,
        num_workers=0,
    )
    payload_path, sidecar_path = dense_image_paths(out_dir, sample_id="a")
    original_payload = payload_path.read_bytes()

    def _fail_payload_write(*args, **kwargs):
        raise OSError("payload temp write failed")

    monkeypatch.setattr(torch, "save", _fail_payload_write)
    with pytest.raises(OSError, match="payload temp write failed"):
        run_dense_image_shard(
            [spec],
            loaded=_loaded(enc),
            out_dir=out_dir,
            dense=_dense(),
            recipe=replace(original_recipe, dtype="float16"),
            batch_size=1,
            output_dtype=torch.float16,
            num_workers=0,
        )

    assert payload_path.read_bytes() == original_payload
    assert not sidecar_path.exists()
    assert not any(".tmp-" in path.name for path in payload_path.parent.iterdir())


def test_payload_publish_failure_leaves_old_payload_without_sidecar(
    tmp_path, monkeypatch
):
    enc = _encoder()
    spec = _spec(tmp_path, "a")
    out_dir = tmp_path / "out"
    original_recipe = _recipe(dtype="float32")
    run_dense_image_shard(
        [spec],
        loaded=_loaded(enc),
        out_dir=out_dir,
        dense=_dense(),
        recipe=original_recipe,
        batch_size=1,
        num_workers=0,
    )
    payload_path, sidecar_path = dense_image_paths(out_dir, sample_id="a")
    original_payload = payload_path.read_bytes()
    original_replace = os.replace

    def _fail_payload_replace(source, destination):
        if Path(destination) == payload_path:
            raise OSError("payload publish failed")
        return original_replace(source, destination)

    monkeypatch.setattr(os, "replace", _fail_payload_replace)
    with pytest.raises(OSError, match="payload publish failed"):
        run_dense_image_shard(
            [spec],
            loaded=_loaded(enc),
            out_dir=out_dir,
            dense=_dense(),
            recipe=replace(original_recipe, dtype="float16"),
            batch_size=1,
            output_dtype=torch.float16,
            num_workers=0,
        )

    assert payload_path.read_bytes() == original_payload
    assert not sidecar_path.exists()
    assert not any(".tmp-" in path.name for path in payload_path.parent.iterdir())


def test_sidecar_publish_failure_leaves_new_payload_without_sidecar(
    tmp_path, monkeypatch
):
    enc = _encoder()
    spec = _spec(tmp_path, "a")
    out_dir = tmp_path / "out"
    original_recipe = _recipe(dtype="float32")
    run_dense_image_shard(
        [spec],
        loaded=_loaded(enc),
        out_dir=out_dir,
        dense=_dense(),
        recipe=original_recipe,
        batch_size=1,
        num_workers=0,
    )
    payload_path, sidecar_path = dense_image_paths(out_dir, sample_id="a")
    original_replace = os.replace

    def _fail_sidecar_replace(source, destination):
        if Path(destination) == sidecar_path:
            raise OSError("sidecar publish failed")
        return original_replace(source, destination)

    monkeypatch.setattr(os, "replace", _fail_sidecar_replace)
    with pytest.raises(OSError, match="sidecar publish failed"):
        run_dense_image_shard(
            [spec],
            loaded=_loaded(enc),
            out_dir=out_dir,
            dense=_dense(),
            recipe=replace(original_recipe, dtype="float16"),
            batch_size=1,
            output_dtype=torch.float16,
            num_workers=0,
        )

    assert torch.load(payload_path, weights_only=True).dtype == torch.float16
    assert not sidecar_path.exists()
    assert not any(".tmp-" in path.name for path in payload_path.parent.iterdir())


def test_multi_rank_matches_single_rank(tmp_path):
    """Sharding equivalence: 3 shards vs 1 → identical file set and identical grids."""
    enc = _encoder()  # every rank loads the same checkpoint
    specs = [_spec(tmp_path, f"image-{i}") for i in range(7)]

    single_dir = tmp_path / "single"
    run_dense_image_shard(
        specs, loaded=_loaded(enc), out_dir=single_dir, dense=_dense(),
        recipe=_recipe(), batch_size=3, num_workers=0
    )

    multi_dir = tmp_path / "multi"
    for shard in plan_contiguous_shards(specs, 3):
        run_dense_image_shard(
            shard, loaded=_loaded(enc), out_dir=multi_dir, dense=_dense(),
            recipe=_recipe(), batch_size=3, num_workers=0
        )

    def _files(root):
        base = _dense_dir(root)
        return {p.relative_to(base).as_posix() for p in base.rglob("*") if p.is_file()}

    assert _files(single_dir) == _files(multi_dir)
    for name in _files(single_dir):
        if not name.endswith(".pt"):
            continue
        one = torch.load(_dense_dir(single_dir) / name, weights_only=True)
        many = torch.load(_dense_dir(multi_dir) / name, weights_only=True)
        # Same shard size (batch_size=3) on both sides, so the grids are bit-identical.
        torch.testing.assert_close(one, many, rtol=0, atol=0)


def test_run_dense_image_shard_honors_the_on_disk_grid_dtype(tmp_path):
    artifacts = run_dense_image_shard(
        [_spec(tmp_path, "a")],
        loaded=_loaded(_encoder()),
        out_dir=tmp_path / "out",
        dense=_dense(),
        recipe=_recipe(dtype="float16"),
        batch_size=1,
        output_dtype=torch.float16,
        num_workers=0,
    )
    grid = torch.load(artifacts[0].path, weights_only=True)
    assert grid.dtype == torch.float16
    assert json.loads(artifacts[0].metadata_path.read_text())["dtype"] == "float16"


def test_payload_size_is_independent_of_batch_size(tmp_path):
    """Each artifact holds one grid, not a view onto its whole batch's storage."""
    enc = _encoder()
    specs = [_spec(tmp_path, f"image-{i}") for i in range(4)]

    sizes = {}
    for batch_size in (1, 4):
        out_dir = tmp_path / f"out-{batch_size}"
        artifacts = run_dense_image_shard(
            specs, loaded=_loaded(enc), out_dir=out_dir, dense=_dense(),
            recipe=_recipe(),
            batch_size=batch_size, num_workers=0,
        )
        sizes[batch_size] = [artifact.path.stat().st_size for artifact in artifacts]

    assert sizes[1] == sizes[4]
