from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

import slide2vec.inference as inference
from slide2vec.api import ExecutionOptions, PreprocessingConfig
from slide2vec.encoders.base import PatientEncoder, SlideEncoder
from slide2vec.encoders.models.gigapath import GigaPathSlideEncoder
from slide2vec.runtime import (
    distributed_stage,
    embedding,
    embedding_pipeline,
    patient_pipeline,
    tiling,
)


class CapturingGigaPathEncoder(GigaPathSlideEncoder):
    def __init__(self) -> None:
        self.coordinates: torch.Tensor | None = None

    def encode_slide(
        self,
        tile_features: torch.Tensor,
        coordinates: torch.Tensor | None = None,
        *,
        tile_size_lv0: int | None = None,
    ) -> torch.Tensor:
        self.coordinates = coordinates
        return torch.zeros(4)


class CapturingPatientEncoder(PatientEncoder):
    def __init__(self) -> None:
        self.coordinates: torch.Tensor | None = None

    @property
    def encode_dim(self) -> int:
        return 4

    @property
    def device(self) -> torch.device:
        return torch.device("cpu")

    def to(self, device: torch.device | str) -> "CapturingPatientEncoder":
        return self

    def encode_slide(
        self,
        tile_features: torch.Tensor,
        coordinates: torch.Tensor | None = None,
        *,
        tile_size_lv0: int | None = None,
    ) -> torch.Tensor:
        self.coordinates = coordinates
        return torch.zeros(4)

    def encode_patient(self, slide_embeddings: torch.Tensor) -> torch.Tensor:
        return slide_embeddings.mean(dim=0)


class CapturingIdentitySlideEncoder(SlideEncoder):
    def __init__(self) -> None:
        self.coordinates: torch.Tensor | None = None

    @property
    def encode_dim(self) -> int:
        return 4

    @property
    def device(self) -> torch.device:
        return torch.device("cpu")

    def to(self, device: torch.device | str) -> "CapturingIdentitySlideEncoder":
        return self

    def encode_slide(
        self,
        tile_features: torch.Tensor,
        coordinates: torch.Tensor | None = None,
        *,
        tile_size_lv0: int | None = None,
    ) -> torch.Tensor:
        self.coordinates = coordinates
        return torch.zeros(4)


def test_direct_slide_aggregation_prepares_gigapath_coordinates() -> None:
    encoder = CapturingGigaPathEncoder()
    loaded = SimpleNamespace(device=torch.device("cpu"), model=encoder)
    model = SimpleNamespace(level="slide", name="gigapath-slide")
    tiling_result = SimpleNamespace(
        x=np.array([0, 512], dtype=np.int64),
        y=np.array([0, 1024], dtype=np.int64),
        tile_size_lv0=512,
        base_spacing_um=0.25,
        requested_spacing_um=0.5,
    )

    embedding_pipeline.aggregate_tile_embeddings_for_slide(
        loaded,
        model,
        SimpleNamespace(sample_id="slide-a", image_path=Path("slide-a.svs")),
        tiling_result,
        torch.ones((2, 4)),
        preprocessing=PreprocessingConfig(
            requested_spacing_um=0.5,
            requested_tile_size_px=512,
        ),
        execution=ExecutionOptions(precision="fp32"),
    )

    assert torch.equal(
        encoder.coordinates,
        torch.tensor([[0, 0], [256, 512]], dtype=torch.long),
    )


def test_patient_pipeline_uses_shared_coordinate_preparation(
    monkeypatch,
    tmp_path: Path,
) -> None:
    encoder = CapturingPatientEncoder()
    loaded = SimpleNamespace(device=torch.device("cpu"), model=encoder)
    model = SimpleNamespace(
        level="patient",
        name="identity-patient",
        _declare_encoder_input=lambda *_args, **_kwargs: None,
        _load_backend=lambda: loaded,
    )
    tiling_result = SimpleNamespace(
        x=np.array([0, 512], dtype=np.int64),
        y=np.array([0, 1024], dtype=np.int64),
        tile_size_lv0=512,
        base_spacing_um=0.25,
        requested_spacing_um=0.5,
    )
    monkeypatch.setattr(
        patient_pipeline,
        "compute_tile_embeddings_for_slide",
        lambda *_args, **_kwargs: torch.ones((2, 4)),
    )
    monkeypatch.setattr(
        patient_pipeline,
        "write_patient_embeddings",
        lambda patient_id, *_args, **_kwargs: SimpleNamespace(patient_id=patient_id),
    )

    patient_pipeline.run_patient_pipeline(
        model,
        embeddable_slides=[
            SimpleNamespace(
                sample_id="slide-a",
                image_path=tmp_path / "slide-a.svs",
                mask_path=None,
            )
        ],
        embeddable_tiling_results=[tiling_result],
        patient_id_map={"slide-a": "patient-a"},
        preprocessing=PreprocessingConfig(
            requested_spacing_um=0.5,
            requested_tile_size_px=512,
        ),
        execution=ExecutionOptions(output_dir=tmp_path, precision="fp32"),
        output_dir=tmp_path,
    )

    assert torch.equal(
        encoder.coordinates,
        torch.tensor([[0, 0], [512, 1024]], dtype=torch.long),
    )


def test_distributed_slide_aggregation_uses_shared_coordinate_preparation(
    monkeypatch,
    tmp_path: Path,
) -> None:
    encoder = CapturingGigaPathEncoder()
    loaded = SimpleNamespace(device=torch.device("cpu"), model=encoder)
    model = SimpleNamespace(
        level="slide",
        name="gigapath-slide",
        _load_backend_without_transform=lambda: loaded,
    )
    slide = SimpleNamespace(
        sample_id="slide-a",
        image_path=tmp_path / "slide-a.svs",
        mask_path=None,
    )
    tiling_result = SimpleNamespace(
        x=np.array([0, 512], dtype=np.int64),
        y=np.array([0, 1024], dtype=np.int64),
        tile_size_lv0=512,
        base_spacing_um=0.25,
        requested_spacing_um=0.5,
    )

    @contextmanager
    def fake_coordination_dir(_work_dir: Path):
        yield tmp_path / "coordination"

    monkeypatch.setattr(distributed_stage, "distributed_coordination_dir", fake_coordination_dir)
    monkeypatch.setattr(
        distributed_stage,
        "run_distributed_direct_embedding_stage",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        distributed_stage,
        "load_tile_embedding_shards",
        lambda *_args, **_kwargs: [
            {
                "tile_index": np.array([0, 1], dtype=np.int64),
                "tile_embeddings": np.ones((2, 4), dtype=np.float32),
            }
        ],
    )

    distributed_stage.embed_single_slide_distributed(
        model,
        slide=slide,
        tiling_result=tiling_result,
        preprocessing=PreprocessingConfig(
            requested_spacing_um=0.5,
            requested_tile_size_px=512,
        ),
        execution=ExecutionOptions(num_gpus=2, precision="fp32"),
        work_dir=tmp_path,
    )

    assert torch.equal(
        encoder.coordinates,
        torch.tensor([[0, 0], [256, 512]], dtype=torch.long),
    )


def test_non_gigapath_slide_aggregation_retains_coordinates_with_base_identity_hook() -> None:
    encoder = CapturingIdentitySlideEncoder()
    loaded = SimpleNamespace(device=torch.device("cpu"), model=encoder)
    model = SimpleNamespace(level="slide", name="identity-slide")
    tiling_result = SimpleNamespace(
        x=np.array([0, 512], dtype=np.int64),
        y=np.array([0, 1024], dtype=np.int64),
        tile_size_lv0=512,
        base_spacing_um=0.25,
        requested_spacing_um=0.5,
    )

    embedding_pipeline.aggregate_tile_embeddings_for_slide(
        loaded,
        model,
        SimpleNamespace(sample_id="slide-a", image_path=Path("slide-a.svs")),
        tiling_result,
        torch.ones((2, 4)),
        preprocessing=PreprocessingConfig(
            requested_spacing_um=0.5,
            requested_tile_size_px=512,
        ),
        execution=ExecutionOptions(precision="fp32"),
    )

    assert torch.equal(
        encoder.coordinates,
        torch.tensor([[0, 0], [512, 1024]], dtype=torch.long),
    )


def test_persisted_tile_aggregation_prepares_gigapath_coordinates(
    monkeypatch,
    tmp_path: Path,
) -> None:
    encoder = CapturingGigaPathEncoder()
    loaded = SimpleNamespace(device=torch.device("cpu"), model=encoder)
    model = SimpleNamespace(
        level="slide",
        name="gigapath-slide",
        _load_backend_without_transform=lambda: loaded,
    )
    tiling_result = SimpleNamespace(
        x=np.array([0, 512], dtype=np.int64),
        y=np.array([0, 1024], dtype=np.int64),
        tile_size_lv0=512,
        base_spacing_um=0.25,
        requested_spacing_um=0.5,
    )
    monkeypatch.setattr(tiling, "load_tiling_result_from_paths", lambda *_args: tiling_result)
    monkeypatch.setattr(inference, "load_array", lambda _path: torch.ones((2, 4)))
    monkeypatch.setattr(
        embedding,
        "write_slide_embedding_artifact",
        lambda sample_id, *_args, **_kwargs: SimpleNamespace(sample_id=sample_id),
    )
    artifact = SimpleNamespace(
        sample_id="slide-a",
        path=tmp_path / "slide-a.tiles.h5",
        metadata={
            "coordinates_npz_path": str(tmp_path / "slide-a.coordinates.npz"),
            "coordinates_meta_path": str(tmp_path / "slide-a.coordinates.meta.json"),
            "image_path": str(tmp_path / "slide-a.svs"),
        },
    )

    inference.aggregate_tiles(
        model,
        [artifact],
        execution=ExecutionOptions(output_dir=tmp_path, precision="fp32"),
    )

    assert torch.equal(
        encoder.coordinates,
        torch.tensor([[0, 0], [256, 512]], dtype=torch.long),
    )
