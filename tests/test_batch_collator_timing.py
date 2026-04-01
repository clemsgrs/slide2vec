from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


def test_batch_tile_collator_emits_worker_and_reader_timing(monkeypatch: pytest.MonkeyPatch):
    torch = pytest.importorskip("torch")
    from slide2vec.data import dataset

    timings = {
        "reader_open_ms": 1.25,
        "reader_read_ms": 8.5,
    }

    class FakeReader:
        def __init__(self, tar_path: Path, tile_size_px: int):
            self.tar_path = tar_path
            self.tile_size_px = tile_size_px

        def read_batch_with_timing(self, tile_indices):
            tensor = torch.zeros((len(tile_indices), 3, self.tile_size_px, self.tile_size_px), dtype=torch.uint8)
            return tensor, dict(timings)

    monkeypatch.setattr(dataset, "TarTileReader", FakeReader)

    collator = dataset.BatchTileCollator(
        tar_path=Path("/tmp/fake.tiles.tar"),
        tiling_result=SimpleNamespace(requested_tile_size_px=4),
    )

    indices, tensor, timing = collator([2, 5])

    assert indices.tolist() == [2, 5]
    assert tuple(tensor.shape) == (2, 3, 4, 4)
    assert timing["reader_open_ms"] == pytest.approx(1.25)
    assert timing["reader_read_ms"] == pytest.approx(8.5)
    assert timing["worker_batch_ms"] >= 0.0


def test_on_the_fly_collator_emits_worker_and_reader_timing(monkeypatch: pytest.MonkeyPatch):
    torch = pytest.importorskip("torch")
    import slide2vec.data.tile_reader as tile_reader

    class FakeReader:
        ordered_indices = None

        def __init__(self, image_path, tiling_result, *, backend: str, num_cucim_workers: int, gpu_decode: bool, use_supertiles: bool):
            self.tile_size = int(tiling_result.effective_tile_size_px)

        def read_batch_with_timing(self, tile_indices):
            tensor = torch.zeros((len(tile_indices), 3, self.tile_size, self.tile_size), dtype=torch.uint8)
            return tensor, {"reader_open_ms": 2.0, "reader_read_ms": 7.25}

    monkeypatch.setattr(tile_reader, "WSITileReader", FakeReader)

    collator = tile_reader.OnTheFlyBatchTileCollator(
        image_path=Path("/tmp/fake.svs"),
        tiling_result=SimpleNamespace(effective_tile_size_px=4),
        backend="cucim",
        num_cucim_workers=4,
        gpu_decode=False,
        use_supertiles=False,
    )

    indices, tensor, timing = collator([0, 4])

    assert indices.tolist() == [0, 4]
    assert tuple(tensor.shape) == (2, 3, 4, 4)
    assert timing["reader_open_ms"] == pytest.approx(2.0)
    assert timing["reader_read_ms"] == pytest.approx(7.25)
    assert timing["worker_batch_ms"] >= 0.0
