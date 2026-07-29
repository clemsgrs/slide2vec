"""Tests for the CPU dense encode/write loop (issue #217).

``run_dense_shard`` is the device-agnostic encode+write loop: read each ROI, encode the
dense grid (reusing ``iter_regions_dense``), and write ``<x>_<y>.pt`` +
``<x>_<y>.meta.json`` per ROI. Tested on CPU with a fake WSI backend
(``_open_wsi_backend`` monkeypatch) + a random-weight encoder. The ROI-granularity split
itself is the shared ``plan_contiguous_shards`` (see ``test_sharding.py``).

The real 4-rank equivalence check runs on CPU: world_size=1 vs 4 shards, same file
set, grids within a per-grid cosine tolerance (docs/adr/0002).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
timm = pytest.importorskip("timm")

from slide2vec.api import DenseOptions  # noqa: E402
from slide2vec.artifacts import region_dense_paths, write_dense_region  # noqa: E402
from slide2vec.data import tile_reader  # noqa: E402
from slide2vec.encoders.base import TimmTileEncoder  # noqa: E402
from slide2vec.runtime.dense_shard import RegionSpec, run_dense_shard  # noqa: E402
from slide2vec.runtime.sharding import plan_contiguous_shards  # noqa: E402


def _encoder(**kwargs) -> TimmTileEncoder:
    return TimmTileEncoder(
        "vit_tiny_patch16_224", pretrained=False, num_classes=0,
        dynamic_img_size=True, **kwargs,
    )


def _canned_region(location, size) -> np.ndarray:
    """Deterministic RGB region for a location, of the requested ``(width, height)``."""
    width, height = int(size[0]), int(size[1])
    x, y = int(location[0]), int(location[1])
    rng = np.random.default_rng(abs(hash((x, y))) % (2**32))
    return rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)


class _FakeBackend:
    """Serves deterministic region arrays and records every read (both read paths)."""

    def __init__(self) -> None:
        self.read_regions_calls: list[list[tuple[int, int]]] = []
        self.read_region_calls: list[tuple[int, int]] = []

    @property
    def locations_read(self) -> list[tuple[int, int]]:
        flat = [loc for batch in self.read_regions_calls for loc in batch]
        return flat + list(self.read_region_calls)

    def read_regions(self, locations, level, size, num_workers):
        locs = [(int(x), int(y)) for x, y in locations]
        self.read_regions_calls.append(locs)
        return [_canned_region(loc, size) for loc in locs]

    def read_region(self, location, level, size):
        loc = (int(location[0]), int(location[1]))
        self.read_region_calls.append(loc)
        return _canned_region(loc, size)


class _FakeBackendFactory:
    """Stands in for ``_open_wsi_backend``: hands out one shared fake per image path."""

    def __init__(self) -> None:
        self.open_count = 0
        self.backends: dict[str, _FakeBackend] = {}

    def __call__(self, image_path, backend, gpu_decode):
        self.open_count += 1
        return self.backends.setdefault(str(image_path), _FakeBackend())

    @property
    def locations_read(self) -> list[tuple[int, int]]:
        out: list[tuple[int, int]] = []
        for b in self.backends.values():
            out.extend(b.locations_read)
        return out


@pytest.fixture
def fake_backend(monkeypatch) -> _FakeBackendFactory:
    factory = _FakeBackendFactory()
    monkeypatch.setattr(tile_reader, "_open_wsi_backend", factory)
    return factory


def _spec(x, y, *, sample_id="s0", image_path="s0.tif", annotation=None,
          requested=64, read=None, read_level=0, backend="cucim") -> RegionSpec:
    return RegionSpec(
        sample_id=sample_id,
        image_path=image_path,
        x=int(x),
        y=int(y),
        read_level=int(read_level),
        read_tile_size_px=int(read if read is not None else requested),
        requested_tile_size_px=int(requested),
        backend=backend,
        annotation=annotation,
    )


# --------------------------------------------------------------------------------------
# run_dense_shard: the device-agnostic encode+write loop, CPU-tested
# --------------------------------------------------------------------------------------


def test_region_dense_paths_rejects_sample_id_paths(tmp_path):
    with pytest.raises(ValueError, match="sample_id"):
        region_dense_paths(
            tmp_path,
            sample_id="/tmp/outside",
            annotation=None,
            x=0,
            y=0,
        )


def test_region_dense_paths_rejects_drive_relative_sample_id(tmp_path):
    with pytest.raises(ValueError, match="sample_id"):
        region_dense_paths(
            tmp_path,
            sample_id="C:outside",
            annotation=None,
            x=0,
            y=0,
        )


def test_region_dense_paths_rejects_symlink_escape(tmp_path):
    output_dir = tmp_path / "output"
    slide_dir = output_dir / "dense_embeddings" / "slide-1"
    outside_dir = tmp_path / "outside"
    slide_dir.parent.mkdir(parents=True)
    outside_dir.mkdir()
    slide_dir.symlink_to(outside_dir, target_is_directory=True)

    with pytest.raises(ValueError, match="output_dir"):
        region_dense_paths(
            output_dir,
            sample_id="slide-1",
            annotation=None,
            x=0,
            y=0,
        )


def test_region_dense_paths_rejects_annotation_paths(tmp_path):
    with pytest.raises(ValueError, match="annotation"):
        region_dense_paths(
            tmp_path,
            sample_id="slide-1",
            annotation="../../outside",
            x=0,
            y=0,
        )


def test_write_dense_region_does_not_publish_interrupted_sidecar(tmp_path, monkeypatch):
    original_write_text = Path.write_text

    def _interrupted_write(path, text, *, encoding):
        original_write_text(path, text[:1], encoding=encoding)
        raise OSError("simulated interrupted metadata write")

    monkeypatch.setattr(Path, "write_text", _interrupted_write)

    with pytest.raises(OSError, match="interrupted metadata write"):
        write_dense_region(
            torch.zeros((2, 2, 2)),
            output_dir=tmp_path,
            sample_id="slide-1",
            annotation=None,
            x=0,
            y=0,
            metadata={"feature_dim": 2, "grid_shape": [2, 2]},
        )

    payload_path, sidecar_path = region_dense_paths(
        tmp_path,
        sample_id="slide-1",
        annotation=None,
        x=0,
        y=0,
    )
    assert payload_path.exists()
    assert not sidecar_path.exists()


def _dense(**kwargs) -> DenseOptions:
    return DenseOptions(spacing_um=0.5, **{"target_size": 64, **kwargs})


def _slide_dir(out_dir, sample_id="s0", annotation=None):
    sub = "dense_embeddings" if annotation is None else f"dense_embeddings/{annotation}"
    return out_dir / sub / sample_id


def test_run_dense_shard_writes_payload_and_sidecar_per_region(fake_backend, tmp_path):
    """Completeness: N ROIs in → N ``.pt`` + N ``.meta.json`` out, zero duplicates."""
    enc = _encoder()
    coords = [(0, 0), (64, 0), (0, 64)]
    regions = [_spec(x, y) for x, y in coords]

    artifacts = run_dense_shard(
        regions, model=enc, out_dir=tmp_path, dense=_dense(), batch_size=2, device="cpu",
    )

    assert len(artifacts) == 3
    slide_dir = _slide_dir(tmp_path)
    for (x, y), art in zip(coords, artifacts):
        payload = slide_dir / f"{x}_{y}.pt"
        sidecar = slide_dir / f"{x}_{y}.meta.json"
        assert payload.exists() and sidecar.exists()
        assert art.path == payload.resolve()
        assert art.metadata_path == sidecar.resolve()
        assert art.grid_shape == (4, 4)
        assert art.feature_dim == enc.encode_dim
        assert (art.x, art.y) == (x, y)
    # Exactly N payloads/sidecars in the slide directory — no duplicates, no temp files.
    assert sorted(p.name for p in slide_dir.glob("*.pt")) == ["0_0.pt", "0_64.pt", "64_0.pt"]
    assert len(list(slide_dir.glob("*.meta.json"))) == 3
    assert list(slide_dir.glob("*.tmp*")) == []


def _multi_slide_regions() -> list[RegionSpec]:
    """Two slides, seven ROIs — a 4-way split crosses the slide boundary (boundary re-open)."""
    a = [_spec(x, y, sample_id="a", image_path="a.tif") for x, y in [(0, 0), (64, 0), (0, 64)]]
    b = [
        _spec(x, y, sample_id="b", image_path="b.tif")
        for x, y in [(0, 0), (64, 0), (0, 64), (64, 64)]
    ]
    return a + b


def _dense_dir_files(out_dir) -> set:
    root = out_dir / "dense_embeddings"
    return {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}


def test_multi_rank_matches_single_rank(fake_backend, tmp_path):
    """Equivalence (D4): 4 shards vs 1 → identical file set, grids cosine ≥ 1 − 1e-4."""
    regions = _multi_slide_regions()
    enc = _encoder()  # every rank loads the same checkpoint → one shared encoder here

    single_dir = tmp_path / "single"
    run_dense_shard(regions, model=enc, out_dir=single_dir, dense=_dense(),
                    batch_size=3, device="cpu")

    multi_dir = tmp_path / "multi"
    shards = plan_contiguous_shards(regions, 4)
    for shard in shards:  # ranks run the identical loop over their contiguous shard
        run_dense_shard(shard, model=enc, out_dir=multi_dir, dense=_dense(),
                        batch_size=3, device="cpu")

    # Same set of payloads + sidecars regardless of how ROIs were sharded.
    assert _dense_dir_files(single_dir) == _dense_dir_files(multi_dir)

    for rel in _dense_dir_files(single_dir):
        if not rel.endswith(".pt"):
            continue
        g1 = torch.load(single_dir / "dense_embeddings" / rel, weights_only=True).float().numpy().ravel()
        g4 = torch.load(multi_dir / "dense_embeddings" / rel, weights_only=True).float().numpy().ravel()
        cos = float(np.dot(g1, g4) / (np.linalg.norm(g1) * np.linalg.norm(g4) + 1e-12))
        assert cos >= 1 - 1e-4, f"{rel}: cos={cos}"


def test_run_dense_shard_skips_regions_with_existing_sidecar(fake_backend, tmp_path):
    """Resume/crash-safety (D9): a ROI whose sidecar exists is not re-read or re-encoded."""
    enc = _encoder()
    regions = [_spec(x, y) for x, y in [(0, 0), (64, 0), (0, 64)]]

    first = run_dense_shard(regions, model=enc, out_dir=tmp_path, dense=_dense(),
                            batch_size=2, device="cpu")
    reads_after_first = len(fake_backend.locations_read)

    # Re-running with every sidecar already on disk reads nothing and still returns all N.
    second = run_dense_shard(regions, model=enc, out_dir=tmp_path, dense=_dense(),
                             batch_size=2, device="cpu")
    assert len(fake_backend.locations_read) == reads_after_first  # no new reads
    assert [(a.x, a.y) for a in second] == [(a.x, a.y) for a in first]
    assert [a.path for a in second] == [a.path for a in first]


def test_run_dense_shard_reencodes_payload_missing_its_sidecar(fake_backend, tmp_path):
    """Crash-safety (D6): a ``.pt`` with no sidecar is re-encoded, never counted as done."""
    enc = _encoder()
    regions = [_spec(x, y) for x, y in [(0, 0), (64, 0), (0, 64)]]
    run_dense_shard(regions, model=enc, out_dir=tmp_path, dense=_dense(),
                    batch_size=2, device="cpu")

    # Simulate a crash after the payload landed but before the sidecar: drop (64,0)'s sidecar.
    slide_dir = _slide_dir(tmp_path)
    (slide_dir / "64_0.meta.json").unlink()
    reads_before = len(fake_backend.locations_read)

    run_dense_shard(regions, model=enc, out_dir=tmp_path, dense=_dense(),
                    batch_size=2, device="cpu")

    new_reads = fake_backend.locations_read[reads_before:]
    assert new_reads == [(64, 0)]  # only the incomplete ROI is re-read/re-encoded
    assert (slide_dir / "64_0.meta.json").exists()  # sidecar restored


def test_run_dense_shard_namespaces_annotation_subdir(fake_backend, tmp_path):
    """A real annotation class lands under ``dense_embeddings/<class>/<sample_id>/`` (D5)."""
    enc = _encoder()
    regions = [_spec(0, 0, annotation="tumor"), _spec(64, 0, annotation="tumor")]
    run_dense_shard(regions, model=enc, out_dir=tmp_path, dense=_dense(),
                    batch_size=2, device="cpu")
    slide_dir = _slide_dir(tmp_path, annotation="tumor")
    assert (slide_dir / "0_0.pt").exists()
    assert (slide_dir / "64_0.meta.json").exists()
    assert not (tmp_path / "dense_embeddings" / "s0").exists()  # not the flat root


def test_run_dense_shard_keeps_merged_structural_identity_on_fresh_and_resume(
    fake_backend,
    tmp_path,
):
    region = _spec(0, 0, annotation="merged")
    encoder = _encoder()

    fresh = run_dense_shard(
        [region],
        model=encoder,
        out_dir=tmp_path,
        dense=_dense(),
        batch_size=1,
        device="cpu",
        num_workers=0,
    )
    resumed = run_dense_shard(
        [region],
        model=encoder,
        out_dir=tmp_path,
        dense=_dense(),
        batch_size=1,
        device="cpu",
        num_workers=0,
    )

    expected_path = tmp_path / "dense_embeddings" / "s0" / "0_0.pt"
    metadata = json.loads(expected_path.with_suffix(".meta.json").read_text())
    assert fresh[0].path == expected_path.resolve()
    assert fresh[0].annotation is None
    assert metadata["annotation"] is None
    assert resumed[0].path == expected_path.resolve()
    assert resumed[0].annotation is None
    assert fake_backend.open_count == 1


def test_run_dense_shard_sidecar_records_extraction_geometry_only(fake_backend, tmp_path):
    """The sidecar carries the extraction geometry + encode params slide2vec owns — and no
    caller ``extra`` passthrough (D7)."""
    enc = _encoder()
    dense = _dense(target_size=60, window_size=32, overlap=0.25, pad_mode="reflect")
    run_dense_shard([_spec(128, 256, requested=60)], model=enc, out_dir=tmp_path,
                    dense=dense, batch_size=1, device="cpu")
    meta = json.loads((_slide_dir(tmp_path) / "128_256.meta.json").read_text())
    assert meta == {
        "artifact_type": "dense_embeddings",
        "sample_id": "s0",
        "annotation": None,
        "x": 128,
        "y": 256,
        "format": "pt",
        "dtype": "float32",
        "feature_dim": enc.encode_dim,
        "grid_shape": [4, 4],          # 60 padded up to 64 → 4×4 tokens
        "target_size": [60, 60],
        "patch_size": [16, 16],
        "encoded_size": [64, 64],
        "pad": [4, 4],
        "spacing_um": 0.5,
        "tolerance": 0.05,
        "backend": "cucim",
        "read_level": 0,
        "read_tile_size_px": 60,
        "requested_tile_size_px": 60,
        "pad_mode": "reflect",
        "image_pad_value": None,
        "window_size": 32,
        "overlap": 0.25,
        "feature_kind": "patch_features",
        "attention_blocks": [-1],
        "attention_include_registers": False,
    }


def test_run_dense_shard_supports_cls_attention_feature_kind(fake_backend, tmp_path):
    enc = _encoder()
    regions = [_spec(0, 0)]
    artifacts = run_dense_shard(
        regions, model=enc, out_dir=tmp_path, dense=_dense(feature_kind="cls_attention"),
        batch_size=1, device="cpu",
    )
    assert len(artifacts) == 1
    assert artifacts[0].grid_shape == (4, 4)
    meta = json.loads(artifacts[0].metadata_path.read_text())
    assert meta["feature_kind"] == "cls_attention"


def test_run_dense_shard_spanning_two_slides_opens_each(fake_backend, tmp_path):
    """A contiguous shard crossing a slide boundary encodes both slides into their own dirs."""
    enc = _encoder()
    regions = [
        _spec(0, 0, sample_id="a", image_path="a.tif"),
        _spec(64, 0, sample_id="a", image_path="a.tif"),
        _spec(0, 0, sample_id="b", image_path="b.tif"),
    ]
    run_dense_shard(regions, model=enc, out_dir=tmp_path, dense=_dense(),
                    batch_size=2, device="cpu")
    assert (_slide_dir(tmp_path, "a") / "0_0.pt").exists()
    assert (_slide_dir(tmp_path, "a") / "64_0.pt").exists()
    assert (_slide_dir(tmp_path, "b") / "0_0.pt").exists()
    assert set(fake_backend.backends) == {"a.tif", "b.tif"}  # both slides opened
