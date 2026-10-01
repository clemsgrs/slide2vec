"""Patient-level models and annotation-aware sampling.

A patient encoder aggregates one embedding per slide. Ways the combination with
annotation sampling can fail:

- per-annotation sampling yields one tile bag per ``(slide, class)``; the patient
  pipeline writes every class of a slide to one path (last one wins) and counts each
  class bag as a slide of the patient;
- the combination is rejected only after tiles or embeddings were computed;
- the rejection also blocks merged output, which keeps one bag per slide.
"""

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from torchvision.transforms import v2

from slide2vec.api import ExecutionOptions, Model, Pipeline, PreprocessingConfig
from slide2vec.artifacts import load_array
from slide2vec.encoders.base import PatientEncoder
from slide2vec.runtime.types import LoadedModel


pytest.importorskip("openslide")

WSI_PATH = Path(__file__).parent / "fixtures" / "input" / "test-wsi.tif"


def _preprocessing(output_mode: str) -> PreprocessingConfig:
    return PreprocessingConfig(
        backend="openslide",
        requested_spacing_um=0.5,
        requested_tile_size_px=224,
        filtering={"a_t": 0, "a_h": 0},
        preview={"save_mask_preview": False, "save_tiling_preview": False},
        masks={
            "pixel_mapping": {"tumor": 2, "stroma": 3},
            "min_coverage": {"tissue": None, "tumor": 0.5, "stroma": 0.5},
            "colors": {"tumor": [255, 0, 0], "stroma": [0, 0, 255]},
            "output_mode": output_mode,
        },
    )


def _annotated_slide(tmp_path: Path, sample_id: str, *, offset: int) -> dict:
    """The fixture WSI with a small annotated block: left half tumor, right half stroma.

    The mask is 1/64 of the slide, so the 28px block covers about 4x4 tiles of 224px at
    0.5 um/px. ``offset`` moves the block so two slides read different pixels.
    """
    mask_path = tmp_path / f"{sample_id}-mask.png"
    labels = np.zeros((200, 216), dtype=np.uint8)
    labels[offset : offset + 28, offset : offset + 14] = 2
    labels[offset : offset + 28, offset + 14 : offset + 28] = 3
    Image.fromarray(labels).save(mask_path)
    return {
        "sample_id": sample_id,
        "image_path": WSI_PATH,
        "mask_path": mask_path,
        "patient_id": "patient-1",
    }


class _MeanPatientEncoder(PatientEncoder):
    """Deterministic stand-in: tiles -> channel means, slides and patients -> means."""

    encode_dim = 3
    device = torch.device("cpu")

    def to(self, device):
        return self

    def encode_tiles(self, batch):
        return batch.float().mean(dim=(2, 3))

    def encode_slide(self, tile_features, coordinates=None, *, tile_size_lv0=None):
        return tile_features.float().mean(dim=0)

    def encode_patient(self, slide_embeddings):
        return slide_embeddings.float().mean(dim=0)


def _patient_model(monkeypatch) -> Model:
    model = Model.from_preset("moozy", device="cpu")
    loaded = LoadedModel(
        name="moozy",
        level="patient",
        model=_MeanPatientEncoder(),
        transforms=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
        feature_dim=3,
        device=torch.device("cpu"),
        tile_feature_dim=3,
    )
    monkeypatch.setattr(model, "_load_backend", lambda: loaded)
    return model


def _execution(output_dir: Path | None) -> ExecutionOptions:
    return ExecutionOptions(
        output_dir=output_dir,
        num_gpus=1,
        precision="fp32",
        num_workers_per_gpu=0,
        num_preprocessing_workers=1,
        save_tile_embeddings=True,
        save_slide_embeddings=True,
    )


def test_pipeline_rejects_patient_model_with_per_annotation_sampling(tmp_path):
    model = Model.from_preset("moozy", device="cpu")
    output_dir = tmp_path / "out"
    pipeline = Pipeline(model, _preprocessing("per_annotation"), execution=_execution(output_dir))

    with pytest.raises(ValueError, match="does not support per-annotation sampling"):
        pipeline.run(slides=[_annotated_slide(tmp_path, "slide-a", offset=60)])

    assert model._backend is None  # nothing was loaded, so nothing was embedded
    assert not (output_dir / "process_list.csv").exists()  # nor tiled


@pytest.mark.parametrize("method", ["embed_patient", "embed_patients"])
def test_model_rejects_patient_embedding_with_per_annotation_sampling(tmp_path, method):
    model = Model.from_preset("moozy", device="cpu")

    with pytest.raises(ValueError, match="does not support per-annotation sampling"):
        getattr(model, method)(
            [_annotated_slide(tmp_path, "slide-a", offset=60)],
            preprocessing=_preprocessing("per_annotation"),
            execution=_execution(None),
        )

    assert model._backend is None


def test_pipeline_embeds_patient_from_merged_annotation_bags(tmp_path, monkeypatch):
    output_dir = tmp_path / "out"
    slides = [
        _annotated_slide(tmp_path, "slide-a", offset=60),
        _annotated_slide(tmp_path, "slide-b", offset=120),
    ]

    result = Pipeline(
        _patient_model(monkeypatch), _preprocessing("merged"), execution=_execution(output_dir)
    ).run(slides=slides)

    patient, = result.patient_artifacts
    assert patient.patient_id == "patient-1"
    assert patient.num_slides == 2
    # One flat bag per slide, holding the union of the tumor and stroma tiles.
    assert [(a.sample_id, a.annotation) for a in result.tile_artifacts] == [
        ("slide-a", None),
        ("slide-b", None),
    ]
    assert [a.path for a in result.tile_artifacts] == [
        (output_dir / "tile_embeddings" / "slide-a.pt").resolve(),
        (output_dir / "tile_embeddings" / "slide-b.pt").resolve(),
    ]
    assert all(a.num_tiles > 0 for a in result.tile_artifacts)
    assert [a.path for a in result.slide_artifacts] == [
        (output_dir / "slide_embeddings" / "slide-a.pt").resolve(),
        (output_dir / "slide_embeddings" / "slide-b.pt").resolve(),
    ]
    slide_a, slide_b = (load_array(a.path) for a in result.slide_artifacts)
    assert not torch.equal(slide_a, slide_b)
    torch.testing.assert_close(load_array(patient.path), (slide_a + slide_b) / 2)


def test_model_embeds_patient_from_merged_annotation_bags(tmp_path, monkeypatch):
    slides = [
        _annotated_slide(tmp_path, "slide-a", offset=60),
        _annotated_slide(tmp_path, "slide-b", offset=120),
    ]

    patient = _patient_model(monkeypatch).embed_patient(
        slides, preprocessing=_preprocessing("merged"), execution=_execution(None)
    )

    assert patient.patient_id == "patient-1"
    assert sorted(patient.slide_embeddings) == ["slide-a", "slide-b"]
    torch.testing.assert_close(
        patient.patient_embedding,
        (patient.slide_embeddings["slide-a"] + patient.slide_embeddings["slide-b"]) / 2,
    )
