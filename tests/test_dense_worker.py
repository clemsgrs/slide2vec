"""Smoke test for the torchrun dense worker entry (issue #217, layer 3).

The worker is deliberately near logic-free: read ``RANK`` / ``WORLD_SIZE`` from the env (no
NCCL), load the JSON request + coordinates npz, then call the two CPU-tested layers. This
exercises that wiring on CPU — env-driven rank selection, request/npz decode, and the fact
that a rank only encodes its own contiguous shard — with the model load and WSI reads faked.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
timm = pytest.importorskip("timm")

import slide2vec.api as api  # noqa: E402
from slide2vec.api import DenseOptions, ExecutionOptions  # noqa: E402
from slide2vec.data import tile_reader  # noqa: E402
from slide2vec.distributed import dense_worker  # noqa: E402
from slide2vec.encoders.base import TimmTileEncoder  # noqa: E402
from slide2vec.runtime import dense_stage  # noqa: E402
from slide2vec.runtime.dense_shard import RegionSpec  # noqa: E402
from slide2vec.runtime.serialization import (  # noqa: E402
    serialize_dense_options,
    serialize_execution,
)


def _canned_region(location, size) -> np.ndarray:
    width, height = int(size[0]), int(size[1])
    x, y = int(location[0]), int(location[1])
    rng = np.random.default_rng(abs(hash((x, y))) % (2**32))
    return rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)


class _FakeBackend:
    def __init__(self) -> None:
        self.locations: list[tuple[int, int]] = []

    def read_regions(self, locations, level, size, num_workers):
        locs = [(int(x), int(y)) for x, y in locations]
        self.locations.extend(locs)
        return [_canned_region(loc, size) for loc in locs]

    def read_region(self, location, level, size):
        loc = (int(location[0]), int(location[1]))
        self.locations.append(loc)
        return _canned_region(loc, size)


def test_dense_worker_encodes_only_its_rank_shard(monkeypatch, tmp_path):
    from slide2vec.runtime.dense_image_reading import DenseImageReadPlan
    backends: dict[str, _FakeBackend] = {}
    monkeypatch.setattr(
        tile_reader, "_open_wsi_backend",
        lambda image_path, backend, gpu_decode: backends.setdefault(str(image_path), _FakeBackend()),
    )

    encoder = TimmTileEncoder("vit_tiny_patch16_224", pretrained=False, num_classes=0,
                              dynamic_img_size=True)
    declared: list = []

    def _load_backend():
        # A rank declares its own dense encoder-input contract before loading, so that the
        # variable-input constructor settings its ROI geometry implies are applied.
        assert declared, "the rank must declare its dense contract before it loads"
        return SimpleNamespace(model=encoder, device="cpu")

    monkeypatch.setattr(
        api.Model, "from_preset",
        classmethod(lambda cls, name, **kwargs: SimpleNamespace(
            _declare_dense_encoder_input=(
                lambda dense, *, emit_run_info: declared.append(dense)
            ),
            _load_backend=_load_backend,
        )),
    )

    plan = DenseImageReadPlan(
        reader_regime="spacing-readable",
        spacing_source="explicit",
        declared_spacing_um=0.5,
        source_spacing_um=0.25,
        spacing_at_level_0=None,
        read_spacing_um=0.5,
        effective_spacing_um=0.5,
        requested_backend="auto",
        backend="cucim",
        tolerance=0.05,
        read_level=0,
        is_within_tolerance=True,
        read_size=(64, 64),
        output_size=(64, 64),
    )
    specs = [RegionSpec("s0", "s0.tif", i * 64, 0, plan) for i in range(4)]
    request = {
        "model": {"name": "fake", "output_variant": None, "allow_non_recommended_settings": False},
        "dense": serialize_dense_options(DenseOptions(spacing_um=0.5, target_size=64)),
        "execution": serialize_execution(ExecutionOptions(output_dir=tmp_path, num_gpus=2, precision="fp32", batch_size=2)),
        "output_dir": str(tmp_path),
        "progress_events_path": None,
        **dense_stage.build_dense_worker_request(specs, coordinates_npz_path=tmp_path / "coords.npz"),
    }
    request_path = tmp_path / "dense_request.json"
    request_path.write_text(json.dumps(request), encoding="utf-8")

    # World of 2 ranks: plan_contiguous_shards([0,1,2,3], 2) => rank 1 owns ROIs [2, 3].
    monkeypatch.setenv("RANK", "1")
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("LOCAL_RANK", "0")

    rc = dense_worker.main(["--output-dir", str(tmp_path), "--request-path", str(request_path)])
    assert rc == 0

    slide_dir = tmp_path / "dense_embeddings" / "s0"
    assert sorted(p.name for p in slide_dir.glob("*.pt")) == ["128_0.pt", "192_0.pt"]
    # This rank read only its own shard's coordinates.
    assert backends["s0.tif"].locations == [(128, 0), (192, 0)]
