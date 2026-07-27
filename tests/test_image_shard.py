"""Tests for the given-geometry image artifacts + the CPU encode/write loop (issue #234).

Two layers, both CPU-testable:

1. ``slide2vec.artifacts.write_image_embedding`` — one payload + one sidecar per image,
   written atomically and sidecar-last, so the sidecar is an unambiguous done-marker.
2. ``run_image_shard`` — the device-agnostic encode+write loop each rank runs over its
   shard: decode each image, apply the encoder's shipped transform **itemwise** in the
   loader workers, stack, encode, persist. Exercised here with a random-weight timm
   encoder over real PNGs, including heterogeneously sized and non-square inputs.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
timm = pytest.importorskip("timm")
PIL = pytest.importorskip("PIL")

from PIL import Image  # noqa: E402

from slide2vec.api import ImageSpec  # noqa: E402
from slide2vec.artifacts import (  # noqa: E402
    image_embedding_paths,
    write_image_embedding,
)
from slide2vec.encoders.base import TimmTileEncoder  # noqa: E402
from slide2vec.runtime.image_shard import run_image_shard  # noqa: E402
from slide2vec.runtime.types import LoadedModel  # noqa: E402


def _encoder() -> TimmTileEncoder:
    return TimmTileEncoder("vit_tiny_patch16_224", pretrained=False, num_classes=0)


def _loaded(encoder: TimmTileEncoder) -> LoadedModel:
    """A ``LoadedModel`` under the Given contract: the encoder's shipped transform."""
    return LoadedModel(
        name="fake-encoder",
        level="tile",
        model=encoder,
        transforms=encoder.get_transform(),
        feature_dim=int(encoder.encode_dim),
        device=torch.device("cpu"),
    )


def _write_image(path, *, width: int, height: int) -> None:
    rng = np.random.default_rng(abs(hash((width, height, path.name))) % (2**32))
    pixels = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pixels).save(path)


def _spec(tmp_path, sample_id: str, *, width: int, height: int) -> ImageSpec:
    path = tmp_path / "images" / f"{sample_id}.png"
    _write_image(path, width=width, height=height)
    return ImageSpec(sample_id=sample_id, image_path=str(path))


# --------------------------------------------------------------------------------------
# Layer 1: the artifact write (payload atomic, sidecar last)
# --------------------------------------------------------------------------------------


def test_image_embedding_paths_rejects_sample_id_paths(tmp_path):
    with pytest.raises(ValueError, match="sample_id"):
        image_embedding_paths(tmp_path, sample_id="/tmp/outside", output_format="pt")


def test_image_embedding_paths_rejects_symlink_escape(tmp_path):
    output_dir = tmp_path / "output"
    embeddings_dir = output_dir / "image_embeddings"
    outside_dir = tmp_path / "outside"
    embeddings_dir.parent.mkdir(parents=True)
    outside_dir.mkdir()
    embeddings_dir.symlink_to(outside_dir, target_is_directory=True)

    with pytest.raises(ValueError, match="output_dir"):
        image_embedding_paths(output_dir, sample_id="image-1", output_format="pt")


def test_write_image_embedding_does_not_publish_interrupted_sidecar(tmp_path, monkeypatch):
    """A crashed sidecar write leaves the payload but no done-marker."""
    from pathlib import Path

    original_write_text = Path.write_text

    def _interrupted_write(path, text, *, encoding):
        original_write_text(path, text[:1], encoding=encoding)
        raise OSError("simulated interrupted metadata write")

    monkeypatch.setattr(Path, "write_text", _interrupted_write)

    with pytest.raises(OSError, match="interrupted metadata write"):
        write_image_embedding(
            torch.zeros((4,)),
            output_dir=tmp_path,
            sample_id="image-1",
            output_format="pt",
            metadata={"feature_dim": 4},
        )

    payload_path, sidecar_path = image_embedding_paths(
        tmp_path, sample_id="image-1", output_format="pt"
    )
    assert payload_path.exists()
    assert not sidecar_path.exists()


# --------------------------------------------------------------------------------------
# Layer 2: run_image_shard (device-agnostic encode+write loop)
# --------------------------------------------------------------------------------------


def test_run_image_shard_embeds_heterogeneously_sized_images_in_one_run(tmp_path):
    """BACH-sized 2048x1536 next to PCam-sized 96x96 next to a non-square input.

    Given-geometry inputs cannot be stacked before preprocessing, so the shipped transform
    runs itemwise and only the *transformed* items are stacked — which is what makes a
    single batch over these three images possible at all.
    """
    encoder = _encoder()
    specs = [
        _spec(tmp_path, "bach", width=2048, height=1536),
        _spec(tmp_path, "pcam", width=96, height=96),
        _spec(tmp_path, "wide", width=300, height=150),
    ]

    artifacts = run_image_shard(
        specs,
        loaded=_loaded(encoder),
        out_dir=tmp_path / "out",
        batch_size=3,
        output_precision="fp32",
        num_workers=0,
    )

    assert [artifact.sample_id for artifact in artifacts] == ["bach", "pcam", "wide"]
    embeddings_dir = tmp_path / "out" / "image_embeddings"
    for artifact in artifacts:
        assert artifact.path == (embeddings_dir / f"{artifact.sample_id}.pt").resolve()
        assert artifact.metadata_path.exists()
        assert artifact.feature_dim == encoder.encode_dim
        payload = torch.load(artifact.path, weights_only=True)
        assert tuple(payload.shape) == (encoder.encode_dim,)
    assert list(embeddings_dir.glob("*.tmp*")) == []


def test_run_image_shard_records_the_observed_encoder_input_size(tmp_path):
    """The Given regime's obligation: record what the encoder actually saw."""
    encoder = _encoder()
    specs = [_spec(tmp_path, "wide", width=300, height=150)]

    artifacts = run_image_shard(
        specs,
        loaded=_loaded(encoder),
        out_dir=tmp_path / "out",
        batch_size=1,
        output_precision="fp32",
        num_workers=0,
    )

    metadata = json.loads(artifacts[0].metadata_path.read_text())
    assert metadata["encoder_input_regime"] == "given"
    assert metadata["encoder_input_size_px"] == 224
    assert metadata["artifact_type"] == "image_embeddings"
    assert metadata["sample_id"] == "wide"
    assert metadata["feature_dim"] == encoder.encode_dim
    assert metadata["feature_dtype"] == "fp32"
    assert metadata["image_path"] == str(specs[0].image_path)


def test_run_image_shard_does_not_use_the_batched_transform_spec(tmp_path, monkeypatch):
    """The batched spec is exclusive to the uniform-size declared paths."""
    from slide2vec.runtime import batching, preprocessing

    def _forbidden(transforms):
        pytest.fail("the given-image path must preprocess itemwise")

    # Patched both where it is defined and where the batched pooled path bound it, so the
    # test cannot pass merely because one reference was missed.
    monkeypatch.setattr(preprocessing, "build_batch_transform_spec", _forbidden)
    monkeypatch.setattr(batching, "build_batch_transform_spec", _forbidden)
    specs = [_spec(tmp_path, "a", width=128, height=64)]

    run_image_shard(
        specs,
        loaded=_loaded(_encoder()),
        out_dir=tmp_path / "out",
        batch_size=2,
        output_precision="fp32",
        num_workers=0,
    )


def test_run_image_shard_skips_images_with_existing_sidecar(tmp_path):
    """Resume: an image whose sidecar exists is not re-decoded or re-encoded."""
    encoder = _encoder()
    specs = [_spec(tmp_path, name, width=64, height=64) for name in ("a", "b", "c")]
    out_dir = tmp_path / "out"

    first = run_image_shard(
        specs, loaded=_loaded(encoder), out_dir=out_dir, batch_size=2,
        output_precision="fp32", num_workers=0,
    )
    payload_mtimes = {a.sample_id: a.path.stat().st_mtime_ns for a in first}

    second = run_image_shard(
        specs, loaded=_loaded(encoder), out_dir=out_dir, batch_size=2,
        output_precision="fp32", num_workers=0,
    )

    assert [a.sample_id for a in second] == [a.sample_id for a in first]
    assert {a.sample_id: a.path.stat().st_mtime_ns for a in second} == payload_mtimes


def test_run_image_shard_reencodes_payload_missing_its_sidecar(tmp_path):
    """Crash-safety: a payload with no sidecar is incomplete and is re-encoded."""
    encoder = _encoder()
    specs = [_spec(tmp_path, name, width=64, height=64) for name in ("a", "b")]
    out_dir = tmp_path / "out"
    run_image_shard(specs, loaded=_loaded(encoder), out_dir=out_dir, batch_size=2,
                    output_precision="fp32", num_workers=0)

    _, sidecar_path = image_embedding_paths(out_dir, sample_id="b", output_format="pt")
    sidecar_path.unlink()

    run_image_shard(specs, loaded=_loaded(encoder), out_dir=out_dir, batch_size=2,
                    output_precision="fp32", num_workers=0)

    assert sidecar_path.exists()


def test_multi_rank_matches_single_rank(tmp_path):
    """Sharding equivalence: 3 shards vs 1 → identical file set and identical payloads."""
    from slide2vec.runtime.sharding import plan_contiguous_shards

    encoder = _encoder()  # every rank loads the same checkpoint
    specs = [_spec(tmp_path, f"image-{i}", width=64 + i, height=96) for i in range(7)]

    single_dir = tmp_path / "single"
    run_image_shard(specs, loaded=_loaded(encoder), out_dir=single_dir, batch_size=3,
                    output_precision="fp32", num_workers=0)

    multi_dir = tmp_path / "multi"
    for shard in plan_contiguous_shards(specs, 3):
        run_image_shard(shard, loaded=_loaded(encoder), out_dir=multi_dir, batch_size=3,
                        output_precision="fp32", num_workers=0)

    def _files(root):
        base = root / "image_embeddings"
        return {p.relative_to(base).as_posix() for p in base.rglob("*") if p.is_file()}

    assert _files(single_dir) == _files(multi_dir)
    for name in _files(single_dir):
        if not name.endswith(".pt"):
            continue
        one = torch.load(single_dir / "image_embeddings" / name, weights_only=True)
        many = torch.load(multi_dir / "image_embeddings" / name, weights_only=True)
        torch.testing.assert_close(one, many)


def test_run_image_shard_reencodes_when_the_output_format_changes(tmp_path):
    """The sidecar name carries no format, so resume must check this run's payload too."""
    encoder = _encoder()
    specs = [_spec(tmp_path, "a", width=64, height=64)]
    out_dir = tmp_path / "out"
    run_image_shard(specs, loaded=_loaded(encoder), out_dir=out_dir, batch_size=1,
                    output_precision="fp32", num_workers=0)

    artifacts = run_image_shard(specs, loaded=_loaded(encoder), out_dir=out_dir, batch_size=1,
                               output_precision="fp32", output_format="npz", num_workers=0)

    assert artifacts[0].path.suffix == ".npz"
    assert artifacts[0].path.exists()
    payload = np.load(artifacts[0].path)["features"]
    assert payload.shape == (encoder.encode_dim,)


def test_payload_size_is_independent_of_batch_size(tmp_path):
    """Each artifact holds one vector, not a view onto its whole batch's storage."""
    encoder = _encoder()
    specs = [_spec(tmp_path, f"image-{i}", width=64, height=64) for i in range(4)]

    sizes = {}
    for batch_size in (1, 4):
        out_dir = tmp_path / f"out-{batch_size}"
        artifacts = run_image_shard(
            specs, loaded=_loaded(encoder), out_dir=out_dir, batch_size=batch_size,
            output_precision="fp32", num_workers=0,
        )
        sizes[batch_size] = [artifact.path.stat().st_size for artifact in artifacts]

    assert sizes[1] == sizes[4]


def test_run_image_shard_honors_the_on_disk_feature_dtype(tmp_path):
    artifacts = run_image_shard(
        [_spec(tmp_path, "a", width=64, height=64)],
        loaded=_loaded(_encoder()),
        out_dir=tmp_path / "out",
        batch_size=1,
        output_precision="fp16",
        num_workers=0,
    )
    payload = torch.load(artifacts[0].path, weights_only=True)
    assert payload.dtype == torch.float16
    assert json.loads(artifacts[0].metadata_path.read_text())["feature_dtype"] == "fp16"
