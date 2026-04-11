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
            self.tile_size = int(tiling_result.read_tile_size_px)

        def read_batch_with_timing(self, tile_indices):
            tensor = torch.zeros((len(tile_indices), 3, self.tile_size, self.tile_size), dtype=torch.uint8)
            return tensor, {"reader_open_ms": 2.0, "reader_read_ms": 7.25}

    monkeypatch.setattr(tile_reader, "WSITileReader", FakeReader)

    collator = tile_reader.OnTheFlyBatchTileCollator(
        image_path=Path("/tmp/fake.svs"),
        tiling_result=SimpleNamespace(read_tile_size_px=4),
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


def test_wsi_tile_reader_suppresses_native_stderr_for_cucim(monkeypatch: pytest.MonkeyPatch):
    torch = pytest.importorskip("torch")
    import slide2vec.data.tile_reader as tile_reader

    calls: list[str] = []

    class _FakeSuppress:
        def __enter__(self):
            calls.append("enter")
            return self

        def __exit__(self, *args):
            calls.append("exit")
            return False

    class FakeBackendReader:
        def read_regions(self, locations, level, size, *, num_workers=None):
            del locations, level, num_workers
            width, height = size
            return [np.zeros((height, width, 3), dtype=np.uint8), np.zeros((height, width, 3), dtype=np.uint8)]

    monkeypatch.setattr(tile_reader, "_open_wsi_backend", lambda *args, **kwargs: FakeBackendReader())
    monkeypatch.setattr(tile_reader, "suppress_c_stderr", lambda: _FakeSuppress())
    reader = tile_reader.WSITileReader(
        Path("/tmp/fake.svs"),
        SimpleNamespace(
            read_tile_size_px=4,
            read_level=0,
            x=np.array([0, 4]),
            y=np.array([0, 0]),
        ),
        backend="cucim",
        num_cucim_workers=4,
        gpu_decode=False,
        use_supertiles=False,
    )

    tensor, timing = reader.read_batch_with_timing(np.array([0, 1], dtype=np.int64))

    assert tuple(tensor.shape) == (2, 3, 4, 4)
    assert timing["reader_open_ms"] >= 0.0
    assert timing["reader_read_ms"] >= 0.0
    assert calls == ["enter", "exit"]


def test_on_the_fly_collator_filters_native_stderr_for_cucim(monkeypatch: pytest.MonkeyPatch):
    torch = pytest.importorskip("torch")
    import slide2vec.data.tile_reader as tile_reader

    calls: list[str] = []

    class FakeReader:
        ordered_indices = None
        _backend = "cucim"

        def __init__(self, image_path, tiling_result, *, backend: str, num_cucim_workers: int, gpu_decode: bool, use_supertiles: bool):
            self.tile_size = int(tiling_result.read_tile_size_px)

        def read_batch_with_timing(self, tile_indices):
            tensor = torch.zeros((len(tile_indices), 3, self.tile_size, self.tile_size), dtype=torch.uint8)
            return tensor, {"reader_open_ms": 2.0, "reader_read_ms": 7.25}

    def _fake_run_with_filtered_stderr(func, *, suppress_patterns=()):
        del suppress_patterns
        calls.append("filtered")
        return func()

    monkeypatch.setattr(tile_reader, "WSITileReader", FakeReader)
    monkeypatch.setattr(tile_reader, "run_with_filtered_stderr", _fake_run_with_filtered_stderr)

    collator = tile_reader.OnTheFlyBatchTileCollator(
        image_path=Path("/tmp/fake.svs"),
        tiling_result=SimpleNamespace(read_tile_size_px=4),
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
    assert calls == ["filtered"]
