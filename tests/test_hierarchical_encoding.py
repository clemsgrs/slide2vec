"""Hierarchical tile encoding: the local and the distributed-shard paths.

Both paths encode flat subtile indices in loader order, then place each embedding by its
flat index. A fake reader serves subtile ``i`` as a constant image of value ``i`` and
hands batches back in reverse order, so a misplaced row shows up as a wrong value.

Ways this can fail:

- local scatter puts a subtile's embedding in the wrong region or row-major slot;
- the shard path drops, duplicates or reorders its selection relative to its embeddings;
- merging shards (one of them empty) does not restore the region-major grid;
- a ragged final batch or an empty selection changes shapes or dtypes;
- the reader is built with the wrong geometry, or cucim reads lose the worker budget or
  stderr filtering;
- the local path still takes a partial selection and leaves unselected rows uninitialized;
- on a real slide, the local grid and the merged shard grid disagree.
"""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from slide2vec.api import ExecutionOptions, PreprocessingConfig
from slide2vec.runtime import embedding_pipeline
from slide2vec.runtime.distributed import merge_hierarchical_embedding_shards
from slide2vec.runtime.types import LoadedModel

# 3 regions of 2x2 subtiles, read at the requested spacing (no resize).
PREPROCESSING = replace(
    PreprocessingConfig(),
    backend="openslide",
    requested_spacing_um=0.5,
    requested_tile_size_px=224,
    region_tile_multiple=2,
    requested_region_size_px=448,
)

# Subtile value v -> feature (v / 255, 100 + v / 255), laid out region-major,
# row-major within the region.
EXPECTED_GRID = np.array(
    [[[v / 255.0, 100.0 + v / 255.0] for v in region] for region in [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]]],
    dtype=np.float32,
)


class _ReversedBatchesCollator:
    """Serves subtile ``i`` as a 2x2 image of value ``i``; batches come back reversed."""

    instances: list["_ReversedBatchesCollator"] = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.sampled_indices = None
        _ReversedBatchesCollator.instances.append(self)

    def build_batch_sampler(self, *, batch_size, dataset_indices):
        self.sampled_indices = np.asarray(dataset_indices).copy()
        positions = list(range(len(dataset_indices)))[::-1]
        return [positions[start:start + batch_size] for start in range(0, len(positions), batch_size)]

    def __call__(self, flat_indices):
        indices = torch.as_tensor(flat_indices, dtype=torch.long)
        images = indices.to(torch.uint8).view(-1, 1, 1, 1).expand(-1, 3, 2, 2).contiguous()
        return indices, images, {"worker_batch_ms": 0.0, "reader_open_ms": 0.0, "reader_read_ms": 0.0}


class _ValueEncoder:
    def encode_tiles(self, image):
        values = image[:, 0, 0, 0].to(torch.float32)
        return torch.stack((values, values + 100.0), dim=1)


def _loaded() -> LoadedModel:
    return LoadedModel(
        name="uni",
        level="tile",
        model=_ValueEncoder(),
        transforms=SimpleNamespace(transforms=[]),
        feature_dim=2,
        device=torch.device("cpu"),
    )


def _tiling_result(num_regions: int = 3):
    return SimpleNamespace(
        x=np.arange(num_regions, dtype=np.int64) * 448,
        y=np.zeros(num_regions, dtype=np.int64),
        base_spacing_um=0.5,
        level_downsamples=[1.0],
        read_level=0,
    )


def _slide():
    return SimpleNamespace(sample_id="slide-h", image_path="slide-h.tif", mask_path=None)


@pytest.fixture
def fake_reader(monkeypatch):
    _ReversedBatchesCollator.instances = []
    monkeypatch.setattr(embedding_pipeline, "OnTheFlyHierarchicalBatchCollator", _ReversedBatchesCollator)
    return _ReversedBatchesCollator


def _encode_local(loaded, tiling_result, *, batch_size=5):
    return embedding_pipeline.compute_hierarchical_embeddings_for_slide(
        loaded,
        _slide(),
        tiling_result,
        preprocessing=PREPROCESSING,
        execution=ExecutionOptions(batch_size=batch_size, num_workers_per_gpu=0, num_gpus=1),
    )


def _encode_shard(loaded, flat_indices, *, batch_size=2):
    return embedding_pipeline.compute_hierarchical_embedding_shard_for_slide(
        loaded,
        _slide(),
        _tiling_result(),
        preprocessing=PREPROCESSING,
        execution=ExecutionOptions(batch_size=batch_size, num_workers_per_gpu=0, num_gpus=1),
        flat_indices=flat_indices,
    )


def test_local_encoding_places_reordered_ragged_batches_region_major(fake_reader):
    loaded = _loaded()

    # 12 subtiles, batch size 5: batches [11..7], [6..2], [1, 0].
    result = _encode_local(loaded, _tiling_result())

    assert result.dtype == torch.float32
    np.testing.assert_allclose(result.numpy(), EXPECTED_GRID, rtol=0, atol=1e-6)
    (collator,) = fake_reader.instances
    np.testing.assert_array_equal(collator.sampled_indices, np.arange(12))
    assert loaded.encoder_input_size_px == 224


def test_reader_receives_the_resolved_hierarchical_geometry(fake_reader):
    _encode_local(_loaded(), _tiling_result())
    _encode_shard(_loaded(), np.array([3, 4], dtype=np.int64))

    local, shard = fake_reader.instances
    for collator in (local, shard):
        np.testing.assert_array_equal(collator.kwargs["region_index"], [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])
        np.testing.assert_array_equal(collator.kwargs["subtile_index_within_region"], [0, 1, 2, 3] * 3)
        assert collator.kwargs["read_region_size_px"] == 448
        assert collator.kwargs["read_tile_size_px"] == 224
        assert collator.kwargs["requested_tile_size_px"] == 224
        assert collator.kwargs["backend"] == "openslide"
        assert collator.kwargs["image_path"] == "slide-h.tif"


def test_local_encoding_no_longer_accepts_a_flat_index_selection(fake_reader):
    with pytest.raises(TypeError, match="flat_indices"):
        embedding_pipeline.compute_hierarchical_embeddings_for_slide(
            _loaded(),
            _slide(),
            _tiling_result(),
            preprocessing=PREPROCESSING,
            execution=ExecutionOptions(batch_size=4, num_workers_per_gpu=0, num_gpus=1),
            flat_indices=np.array([0, 1], dtype=np.int64),
        )


def test_local_encoding_of_a_slide_without_regions_is_an_empty_region_grid(fake_reader):
    loaded = _loaded()

    result = _encode_local(loaded, _tiling_result(num_regions=0))

    assert result.shape == (0, 4, 2)
    assert result.dtype == torch.float32
    assert loaded.encoder_input_size_px is None


def test_shard_encoding_returns_its_selection_in_loader_order(fake_reader):
    loaded = _loaded()

    # Selection [9, 2, 5, 11, 0], batch size 2, reversed: [0, 11], [5, 2], [9].
    indices, embeddings = _encode_shard(loaded, np.array([9, 2, 5, 11, 0], dtype=np.int64))

    assert isinstance(indices, np.ndarray)
    np.testing.assert_array_equal(indices, np.array([0, 11, 5, 2, 9], dtype=np.int64))
    np.testing.assert_allclose(
        embeddings.numpy(),
        np.array([[v / 255.0, 100.0 + v / 255.0] for v in [0, 11, 5, 2, 9]], dtype=np.float32),
        rtol=0,
        atol=1e-6,
    )
    (collator,) = fake_reader.instances
    np.testing.assert_array_equal(collator.sampled_indices, [9, 2, 5, 11, 0])
    assert loaded.encoder_input_size_px == 224


def test_shard_encoding_of_an_empty_selection_is_empty(fake_reader):
    loaded = _loaded()

    indices, embeddings = _encode_shard(loaded, np.array([], dtype=np.int64))

    assert indices.dtype == np.int64
    assert indices.shape == (0,)
    assert embeddings.shape == (0, 2)
    assert embeddings.dtype == torch.float32
    assert loaded.encoder_input_size_px is None


def test_reordered_partial_shards_with_an_empty_rank_merge_region_major(fake_reader):
    selections = [
        np.array([9, 2, 5, 11, 0], dtype=np.int64),
        np.array([], dtype=np.int64),
        np.array([10, 1, 3, 8, 4, 7, 6], dtype=np.int64),
    ]
    payloads = []
    for selection in selections:
        indices, embeddings = _encode_shard(_loaded(), selection, batch_size=3)
        payloads.append({"flat_index": indices, "tile_embeddings": embeddings})

    merged = merge_hierarchical_embedding_shards(payloads, num_regions=3, tiles_per_region=4)

    np.testing.assert_allclose(merged.numpy(), EXPECTED_GRID, rtol=0, atol=1e-6)


@pytest.mark.parametrize("path", ["local", "shard"])
def test_cucim_reads_use_the_worker_budget_and_filter_stderr(fake_reader, monkeypatch, path):
    loader_kwargs = {}
    real_dataloader = torch.utils.data.DataLoader

    def capturing_dataloader(dataset, **kwargs):
        loader_kwargs.update(kwargs)
        for worker_only in ("num_workers", "prefetch_factor", "worker_init_fn", "persistent_workers"):
            kwargs.pop(worker_only, None)
        return real_dataloader(dataset, num_workers=0, **kwargs)

    filtered_calls = []

    def recording_filter(fn):
        filtered_calls.append(fn)
        return fn()

    monkeypatch.setattr(torch.utils.data, "DataLoader", capturing_dataloader)
    monkeypatch.setattr(embedding_pipeline, "resolve_on_the_fly_num_workers", lambda workers, num_gpus: (3, ""))
    monkeypatch.setattr(embedding_pipeline, "run_with_filtered_stderr", recording_filter)
    cucim = replace(PREPROCESSING, backend="cucim")
    execution = ExecutionOptions(batch_size=4, num_workers_per_gpu=0, num_gpus=1)

    if path == "local":
        embedding_pipeline.compute_hierarchical_embeddings_for_slide(
            _loaded(), _slide(), _tiling_result(), preprocessing=cucim, execution=execution
        )
    else:
        embedding_pipeline.compute_hierarchical_embedding_shard_for_slide(
            _loaded(),
            _slide(),
            _tiling_result(),
            preprocessing=cucim,
            execution=execution,
            flat_indices=np.array([0, 5], dtype=np.int64),
        )

    assert loader_kwargs["num_workers"] == 3
    assert callable(loader_kwargs["worker_init_fn"])
    assert len(filtered_calls) == 1
    (collator,) = fake_reader.instances
    assert collator.kwargs["backend"] == "cucim"


def test_fixture_slide_local_and_merged_shard_encodings_agree():
    """Real reads from the fixture WSI, CPU, stand-in encoder (see the script's docstring)."""
    pytest.importorskip("openslide")
    from scripts.hierarchical_encoding_consistency import WORLD_SIZES, encode_fixture

    outputs = encode_fixture()

    assert outputs["local"].shape == (4, 4, 12)
    np.testing.assert_array_equal(outputs["world5_rank1_flat_index"], [4, 5, 6])
    assert outputs["world17_rank16_flat_index"].shape == (0,)
    for world_size in WORLD_SIZES:
        np.testing.assert_allclose(outputs[f"world{world_size}_merged"], outputs["local"], rtol=0, atol=1e-6)
