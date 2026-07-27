"""Tests for the parent-side dense orchestration (issue #217, layer 3 glue).

``embed_regions_dense`` flattens the caller's ``SlideRegions`` into the slide-ordered flat
ROI list, resume-filters it (D9), resolves each slide's read plan once, then either runs the
encode/write loop in-process (``num_gpus=1``) or fans it out over torchrun ranks
(``num_gpus>1``). These tests run on CPU: the WSI reads go through the fake
``_open_wsi_backend`` seam, the per-slide read-plan resolution is stubbed (no real slide), and
the torchrun launch is captured rather than executed.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
timm = pytest.importorskip("timm")

from slide2vec.api import DenseOptions, ExecutionOptions, SlideRegions  # noqa: E402
from slide2vec.data import tile_reader  # noqa: E402
from slide2vec.encoders.base import TimmTileEncoder  # noqa: E402
from slide2vec.runtime import dense_stage  # noqa: E402


def _encoder() -> TimmTileEncoder:
    return TimmTileEncoder("vit_tiny_patch16_224", pretrained=False, num_classes=0,
                           dynamic_img_size=True)


def _canned_region(location, size) -> np.ndarray:
    width, height = int(size[0]), int(size[1])
    x, y = int(location[0]), int(location[1])
    rng = np.random.default_rng(abs(hash((x, y))) % (2**32))
    return rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)


class _FakeBackend:
    def __init__(self) -> None:
        self.locations: list[tuple[int, int]] = []
        self.num_workers: list[int] = []

    def read_regions(self, locations, level, size, num_workers):
        locs = [(int(x), int(y)) for x, y in locations]
        self.locations.extend(locs)
        self.num_workers.append(int(num_workers))
        return [_canned_region(loc, size) for loc in locs]

    def read_region(self, location, level, size):
        loc = (int(location[0]), int(location[1]))
        self.locations.append(loc)
        return _canned_region(loc, size)


@pytest.fixture
def fake_backend(monkeypatch):
    backends: dict[str, _FakeBackend] = {}

    def _open(image_path, backend, gpu_decode):
        return backends.setdefault(str(image_path), _FakeBackend())

    monkeypatch.setattr(tile_reader, "_open_wsi_backend", _open)
    return backends


@pytest.fixture
def stub_read_plan(monkeypatch):
    """Stub per-slide read-plan resolution so no real slide is opened for metadata."""
    monkeypatch.setattr(
        dense_stage, "resolve_slide_read_plan",
        lambda image_path, dense: (0, int(dense.target_size), "cucim"),
    )


class _FakeModel:
    """Minimal Model stand-in exposing what the in-process dense path reads.

    ``_load_backend`` refuses to hand out a backend until the dense encoder-input contract
    has been declared, mirroring the real ``Model``: dense declares its effective encoder
    input like every other route that reaches the encoder.
    """

    def __init__(self, encoder, device="cpu") -> None:
        self._encoder = encoder
        self._device = device
        self.name = "fake-encoder"
        self._output_variant = None
        self._requested_device = device
        self.allow_non_recommended_settings = False
        self.declared_dense = None

    def _declare_dense_encoder_input(self, dense, *, emit_run_info):
        self.declared_dense = dense
        return dense

    def _load_backend(self):
        assert self.declared_dense is not None, "dense must declare before it loads"
        return SimpleNamespace(model=self._encoder, device=self._device)

    @property
    def device(self):
        return self._device


def _regions(sample_id="s0", image_path="s0.tif", coords=((0, 0), (64, 0), (0, 64)),
             annotation=None) -> SlideRegions:
    return SlideRegions(
        sample_id=sample_id, image_path=image_path,
        coordinates=np.asarray(coords, dtype=np.int64), annotation=annotation,
    )


def _dense(**kwargs) -> DenseOptions:
    return DenseOptions(spacing_um=0.5, **{"target_size": 64, **kwargs})


def test_embed_regions_dense_num_gpus_one_runs_in_process(fake_backend, stub_read_plan, tmp_path, monkeypatch):
    """``num_gpus=1`` encodes in-process — no torchrun subprocess — and writes every ROI."""
    # Any attempt to launch torchrun in the single-GPU path is a bug.
    monkeypatch.setattr(
        dense_stage, "run_torchrun_worker",
        lambda **kwargs: pytest.fail("num_gpus=1 must not launch torchrun"),
    )
    model = _FakeModel(_encoder())
    execution = ExecutionOptions(
        output_dir=tmp_path,
        num_gpus=1,
        num_workers_per_gpu=7,
        precision="fp32",
    )

    artifacts = dense_stage.embed_regions_dense(
        model, [_regions()], dense=_dense(), execution=execution,
    )

    assert len(artifacts) == 3
    slide_dir = tmp_path / "dense_embeddings" / "s0"
    assert sorted(p.name for p in slide_dir.glob("*.pt")) == ["0_0.pt", "0_64.pt", "64_0.pt"]
    for art in artifacts:
        assert art.metadata_path.exists()
        assert art.grid_shape == (4, 4)
    assert {num_workers for backend in fake_backend.values() for num_workers in backend.num_workers} == {7}


def test_embed_regions_dense_num_gpus_gt_one_launches_dense_worker(fake_backend, stub_read_plan, tmp_path, monkeypatch):
    """``num_gpus>1`` launches ``slide2vec.distributed.dense_worker`` under torchrun, handing
    it the coordinates as an npz (D10b) — the parent itself encodes nothing. The fake torchrun
    stands in for the ranks (reconstruct specs → shard → encode) so collection sees N grids."""
    from slide2vec.runtime.dense_shard import run_dense_shard
    from slide2vec.runtime.sharding import plan_contiguous_shards
    from slide2vec.runtime.serialization import deserialize_dense_options

    caller_dir = tmp_path / "caller"
    caller_dir.mkdir()
    monkeypatch.chdir(caller_dir)
    captured = {}
    rank_encoder = _encoder()  # every rank loads the same checkpoint

    def _fake_run(*, module, num_gpus, output_dir, request_path, **kwargs):
        request = json.loads(Path(request_path).read_text())
        with np.load(request["coordinates_npz_path"], allow_pickle=False) as payload:
            coords = np.asarray(payload["coordinates"])
        captured.update(
            module=module,
            num_gpus=num_gpus,
            output_dir=output_dir,
            request_path=request_path,
            request=request,
            coords=coords,
        )
        # Simulate the ranks: rebuild the flat specs, shard, encode each shard on CPU.
        specs = dense_stage.region_specs_from_request(request)
        dense = deserialize_dense_options(request["dense"])
        for shard in plan_contiguous_shards(specs, num_gpus):
            run_dense_shard(shard, model=rank_encoder, out_dir=Path(output_dir), dense=dense,
                            batch_size=2, device="cpu")

    monkeypatch.setattr(dense_stage, "run_torchrun_worker", _fake_run)
    monkeypatch.setattr(dense_stage, "validate_multi_gpu_execution", lambda *a, **k: None)
    # No in-process encode may happen on the multi-GPU path.
    monkeypatch.setattr(
        dense_stage, "_run_dense_in_process",
        lambda *a, **k: pytest.fail("num_gpus>1 must not encode in-process"),
    )
    model = _FakeModel(_encoder())
    execution = ExecutionOptions(output_dir=Path("output"), num_gpus=3, precision="fp32")

    artifacts = dense_stage.embed_regions_dense(
        model,
        [_regions(coords=((0, 0), (64, 0))), _regions(sample_id="s1", image_path="s1.tif", coords=((0, 0),))],
        dense=_dense(), execution=execution,
    )

    assert captured["module"] == "slide2vec.distributed.dense_worker"
    assert captured["num_gpus"] == 3
    expected_output_dir = (caller_dir / "output").resolve()
    assert Path(captured["output_dir"]) == expected_output_dir
    assert Path(captured["request_path"]).is_absolute()
    assert captured["request"]["execution"]["output_dir"] == str(expected_output_dir)
    assert all(Path(slide["image_path"]).is_absolute() for slide in captured["request"]["slides"])
    # All three ROIs travelled to the npz, slide-ordered.
    assert captured["coords"].tolist() == [[0, 0], [64, 0], [0, 0]]
    assert [s["sample_id"] for s in captured["request"]["slides"]] == ["s0", "s1"]
    assert captured["request"]["dense"]["target_size"] == 64
    # Collection returns one artifact per input ROI, read back off disk (nobody gathered grids).
    assert len(artifacts) == 3
    assert (expected_output_dir / "dense_embeddings" / "s0" / "0_0.pt").exists()
    assert (expected_output_dir / "dense_embeddings" / "s1" / "0_0.pt").exists()


def test_embed_regions_dense_resume_skips_existing_and_logs(fake_backend, stub_read_plan, tmp_path, caplog):
    """Resume (D9): pre-existing ROIs are filtered before dispatch; the skip count is logged."""
    model = _FakeModel(_encoder())
    execution = ExecutionOptions(output_dir=tmp_path, num_gpus=1, precision="fp32")

    dense_stage.embed_regions_dense(model, [_regions(coords=((0, 0), (64, 0)))],
                                    dense=_dense(), execution=execution)
    reads_after_first = sum(len(b.locations) for b in fake_backend.values())

    with caplog.at_level("INFO", logger="slide2vec.runtime.dense_stage"):
        artifacts = dense_stage.embed_regions_dense(
            model, [_regions(coords=((0, 0), (64, 0), (0, 64)))], dense=_dense(), execution=execution,
        )

    assert len(artifacts) == 3
    reads_after_second = sum(len(b.locations) for b in fake_backend.values())
    assert reads_after_second - reads_after_first == 1  # only the one new ROI is read
    assert "2/3 regions already on disk, encoding 1" in caplog.text


def test_embed_regions_dense_all_present_does_not_dispatch(fake_backend, stub_read_plan, tmp_path, monkeypatch):
    """A fully-resumed run encodes nothing (no in-process encode, no torchrun) yet returns all N."""
    model = _FakeModel(_encoder())
    execution = ExecutionOptions(output_dir=tmp_path, num_gpus=1, precision="fp32")
    dense_stage.embed_regions_dense(model, [_regions()], dense=_dense(), execution=execution)

    monkeypatch.setattr(dense_stage, "_run_dense_in_process",
                        lambda *a, **k: pytest.fail("nothing to encode"))
    monkeypatch.setattr(dense_stage, "run_torchrun_worker",
                        lambda **k: pytest.fail("nothing to encode"))
    artifacts = dense_stage.embed_regions_dense(model, [_regions()], dense=_dense(), execution=execution)
    assert len(artifacts) == 3


def test_model_embed_regions_dense_delegates_and_requires_output_dir(monkeypatch, tmp_path):
    """The public API coerces execution, requires an output dir, and delegates to the stage."""
    import slide2vec.runtime.dense_stage as stage_mod
    from slide2vec.api import Model

    seen = {}

    def _fake_stage(model, regions, *, dense, execution):
        seen.update(model=model, regions=regions, dense=dense, execution=execution)
        return ["artifact"]

    monkeypatch.setattr(stage_mod, "embed_regions_dense", _fake_stage)
    model = Model(name="virchow2")  # constructing a Model does not load weights

    # Missing output_dir is rejected up front, before any work.
    with pytest.raises(ValueError):
        model.embed_regions_dense([_regions()], dense=_dense(), execution=ExecutionOptions(num_gpus=1))

    result = model.embed_regions_dense(
        [_regions()], dense=_dense(), execution=ExecutionOptions(output_dir=tmp_path, num_gpus=1),
    )
    assert result == ["artifact"]
    assert seen["model"] is model
    assert seen["execution"].output_dir == tmp_path


def test_region_specs_round_trip_through_request(tmp_path):
    """The npz + request rebuild the exact flat spec list the parent sharded (worker inverse)."""
    from slide2vec.runtime.dense_shard import RegionSpec

    specs = [
        RegionSpec("s0", "s0.tif", 0, 0, 0, 64, 64, "cucim", None),
        RegionSpec("s0", "s0.tif", 64, 0, 0, 64, 64, "cucim", None),
        RegionSpec("s1", "s1.tif", 10, 20, 1, 96, 64, "cucim", "tumor"),
    ]
    request = dense_stage.build_dense_worker_request(
        specs, coordinates_npz_path=tmp_path / "coords.npz"
    )
    assert dense_stage.region_specs_from_request(request) == specs
