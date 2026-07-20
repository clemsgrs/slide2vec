from contextlib import nullcontext
import io
import tarfile
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image


def test_wsi_tile_reader_area_resizes_explicit_read_geometry_to_requested(monkeypatch):
    """A WSI read is canonicalized before it reaches model preprocessing."""
    from slide2vec.data import tile_reader

    values = np.array(
        [
            [0, 0, 4, 4],
            [0, 0, 4, 4],
            [8, 8, 12, 12],
            [8, 8, 12, 12],
        ],
        dtype=np.uint8,
    )
    raw_tile = np.repeat(values[:, :, np.newaxis], 3, axis=2)

    class Reader:
        def read_region(self, location, level, size):
            assert location == (10, 20)
            assert level == 0
            assert size == (4, 4)
            return raw_tile

    monkeypatch.setattr(tile_reader, "_open_wsi_backend", lambda *args: Reader())
    reader = tile_reader.WSITileReader(
        "slide.svs",
        SimpleNamespace(
            read_level=0,
            read_tile_size_px=4,
            requested_tile_size_px=2,
            x=np.array([10]),
            y=np.array([20]),
        ),
        backend="openslide",
        use_supertiles=False,
    )

    tile = reader.read_batch(np.array([0], dtype=np.int64))

    expected = np.array([[0, 4], [8, 12]], dtype=np.uint8)
    np.testing.assert_array_equal(tile[0, 0].numpy(), expected)
    assert tuple(tile.shape) == (1, 3, 2, 2)


def test_hierarchical_reader_area_resizes_each_subtile_to_requested(monkeypatch):
    from slide2vec.data import tile_reader

    values = np.arange(64, dtype=np.uint8).reshape(8, 8)
    raw_region = np.repeat(values[:, :, np.newaxis], 3, axis=2)

    class Reader:
        def read_region(self, location, level, size):
            assert size == (8, 8)
            return raw_region

    monkeypatch.setattr(tile_reader, "_open_wsi_backend", lambda *args: Reader())
    collator = tile_reader.OnTheFlyHierarchicalBatchCollator(
        image_path="slide.svs",
        tiling_result=SimpleNamespace(read_level=0, x=np.array([0]), y=np.array([0])),
        region_index=np.array([0, 0, 0, 0]),
        subtile_index_within_region=np.array([0, 1, 2, 3]),
        read_region_size_px=8,
        read_tile_size_px=4,
        requested_tile_size_px=2,
        backend="openslide",
    )

    _, tiles, _ = collator([0, 1, 2, 3])

    assert tuple(tiles.shape) == (4, 3, 2, 2)
    expected_top_left = np.array([[5, 7], [21, 23]], dtype=np.uint8)
    np.testing.assert_array_equal(tiles[0, 0].numpy(), expected_top_left)


def test_gigapath_preset_records_actual_encoder_input_after_shipped_transform(caplog):
    from slide2vec.encoders.models.gigapath import GigaPath
    from slide2vec.encoders.validation import validate_encoder_config
    from slide2vec.runtime.batching import (
        build_batch_preprocessor_for_tile_images,
        run_forward_pass,
    )
    from slide2vec.runtime.types import LoadedModel

    class Encoder:
        def encode_tiles(self, image):
            assert tuple(image.shape[-2:]) == (224, 224)
            return torch.zeros((image.shape[0], 2), dtype=torch.float32)

    loaded = LoadedModel(
        name="gigapath",
        level="tile",
        model=Encoder(),
        transforms=GigaPath.__new__(GigaPath).get_transform(),
        feature_dim=2,
        device=torch.device("cpu"),
    )
    preprocessor = build_batch_preprocessor_for_tile_images(
        loaded,
        requested_tile_size_px=256,
    )
    dataloader = [
        (
            torch.tensor([0]),
            torch.zeros((1, 3, 256, 256), dtype=torch.uint8),
        )
    ]

    with caplog.at_level("WARNING"):
        validate_encoder_config(
            "gigapath",
            requested_spacing_um=0.5,
            requested_tile_size_px=256,
        )
        run_forward_pass(
            dataloader,
            loaded,
            nullcontext(),
            batch_preprocessor=preprocessor,
        )

    assert loaded.encoder_input_size_px == 224
    assert "non-recommended" not in caplog.text.lower()


def test_single_gpu_artifact_records_read_requested_final_geometry_and_spacing(tmp_path):
    from slide2vec.api import ExecutionOptions
    from slide2vec.runtime.embedding import (
        build_tile_embedding_metadata,
        write_tile_embedding_artifact,
    )

    metadata = build_tile_embedding_metadata(
        SimpleNamespace(name="gigapath", level="tile"),
        tiling_result=SimpleNamespace(
            read_tile_size_px=415,
            requested_tile_size_px=256,
            requested_spacing_um=0.5,
        ),
        image_path="slide.svs",
        mask_path=None,
        tile_size_lv0=415,
        backend="openslide",
        encoder_input_size_px=224,
    )

    artifact = write_tile_embedding_artifact(
        "slide",
        np.zeros((1, 2), dtype=np.float32),
        execution=ExecutionOptions(output_dir=tmp_path, output_format="pt"),
        metadata=metadata,
    )
    persisted = artifact.metadata

    assert {
        key: persisted[key]
        for key in (
            "read_tile_size_px",
            "requested_tile_size_px",
            "encoder_input_size_px",
            "requested_spacing_um",
        )
    } == {
        "read_tile_size_px": 415,
        "requested_tile_size_px": 256,
        "encoder_input_size_px": 224,
        "requested_spacing_um": 0.5,
    }
    assert "input_recipe" not in persisted


def test_distributed_shards_preserve_one_factual_encoder_input_size():
    from slide2vec.runtime.distributed import resolve_shard_encoder_input_size

    payloads = [
        {"encoder_input_size_px": 224},
        {"encoder_input_size_px": 224},
    ]

    assert resolve_shard_encoder_input_size(payloads) == 224


def test_tar_reader_preserves_requested_geometry_before_preprocessing(tmp_path):
    from slide2vec.data.dataset import BatchTileCollator

    tar_path = tmp_path / "slide.tiles.tar"
    buffer = io.BytesIO()
    Image.fromarray(np.full((2, 2, 3), 37, dtype=np.uint8)).save(buffer, format="PNG")
    payload = buffer.getvalue()
    with tarfile.open(tar_path, "w") as archive:
        member = tarfile.TarInfo("000000.png")
        member.size = len(payload)
        archive.addfile(member, io.BytesIO(payload))
    collator = BatchTileCollator(
        tar_path=tar_path,
        tiling_result=SimpleNamespace(
            read_tile_size_px=4,
            requested_tile_size_px=2,
        ),
    )

    _, tiles, _ = collator([0])

    assert tuple(tiles.shape) == (1, 3, 2, 2)


def test_zero_tile_metadata_always_records_null_encoder_input_size():
    from slide2vec.runtime.embedding import build_tile_embedding_metadata

    metadata = build_tile_embedding_metadata(
        SimpleNamespace(name="gigapath", level="tile"),
        tiling_result=SimpleNamespace(
            read_tile_size_px=415,
            requested_tile_size_px=256,
            requested_spacing_um=0.5,
        ),
        image_path="empty.svs",
        mask_path=None,
        tile_size_lv0=415,
        backend="openslide",
    )

    assert "encoder_input_size_px" in metadata
    assert metadata["encoder_input_size_px"] is None


def test_hierarchical_metadata_records_pooled_geometry():
    from slide2vec.api import PreprocessingConfig
    from slide2vec.runtime.embedding import build_hierarchical_embedding_metadata

    preprocessing = PreprocessingConfig(
        requested_spacing_um=0.5,
        requested_tile_size_px=256,
        requested_region_size_px=512,
        region_tile_multiple=2,
    )
    metadata = build_hierarchical_embedding_metadata(
        SimpleNamespace(name="gigapath", level="tile"),
        tiling_result=SimpleNamespace(
            read_tile_size_px=415,
            read_spacing_um=0.31,
            tile_size_lv0=415,
            base_spacing_um=0.31,
            level_downsamples=[1.0],
        ),
        image_path="slide.svs",
        mask_path=None,
        backend="openslide",
        preprocessing=preprocessing,
        encoder_input_size_px=224,
    )

    assert metadata["read_tile_size_px"] == 413
    assert metadata["requested_tile_size_px"] == 256
    assert metadata["encoder_input_size_px"] == 224
    assert metadata["requested_spacing_um"] == 0.5


def test_slide_and_patient_models_inherit_tile_dependency_geometry():
    from slide2vec.encoders.registry import resolve_preprocessing_requirements

    assert resolve_preprocessing_requirements("gigapath-slide") == {
        "tile_size_px": 256,
        "spacing_um": 0.5,
        "source_encoder": "gigapath",
    }
    assert resolve_preprocessing_requirements("moozy") == {
        "tile_size_px": 224,
        "spacing_um": 0.5,
        "source_encoder": "lunit",
    }
