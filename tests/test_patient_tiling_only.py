"""Patient-level models and slides that carry no patient id.

Tiling does not use patient ids, so a tiling-only run must not need them. An
embedding run groups slides by patient, so it still needs one per slide. Ways this
can fail:

- a tiling-only run rejects slides without a patient id (for example hs2p
  ``SlideSpec`` objects, which have no ``patient_id`` field);
- a full run accepts such slides and groups or names patients wrongly;
- a full run rejects them only after the slides were tiled or the model was loaded.
"""

from pathlib import Path

import pandas as pd
import pytest
from hs2p import SlideSpec

from slide2vec.api import ExecutionOptions, Model, Pipeline, PreprocessingConfig
from tests.output_consistency_config import (
    TILING_FILTER_PARAMS,
    TILING_MASKS,
    TILING_PARAMS,
    TILING_SEG_PARAMS,
)


pytest.importorskip("openslide")

FIXTURES = Path(__file__).parent / "fixtures" / "input"
SAMPLE_IDS = ["slide-a", "slide-b"]


def _slides() -> list[SlideSpec]:
    return [
        SlideSpec(
            sample_id=sample_id,
            image_path=FIXTURES / "test-wsi.tif",
            mask_path=FIXTURES / "test-mask.tif",
        )
        for sample_id in SAMPLE_IDS
    ]


def _model_and_pipeline(output_dir: Path) -> tuple[Model, Pipeline]:
    model = Model.from_preset("moozy", device="cpu")
    pipeline = Pipeline(
        model,
        PreprocessingConfig(
            backend="openslide",
            mask_backend="openslide",
            masks=TILING_MASKS,
            segmentation=TILING_SEG_PARAMS,
            filtering=TILING_FILTER_PARAMS,
            preview={"save_mask_preview": False, "save_tiling_preview": False},
            **TILING_PARAMS,
        ),
        execution=ExecutionOptions(output_dir=output_dir, num_gpus=1, num_preprocessing_workers=1),
    )
    return model, pipeline


def test_tiling_only_patient_run_tiles_slides_without_patient_ids(tmp_path):
    model, pipeline = _model_and_pipeline(tmp_path / "out")

    result = pipeline.run(slides=_slides(), tiling_only=True)

    rows = pd.read_csv(result.process_list_path, dtype=str, keep_default_na=False)
    assert rows["sample_id"].tolist() == SAMPLE_IDS
    assert rows["tiling_status"].tolist() == ["success"] * len(SAMPLE_IDS)
    assert all(Path(path).is_file() for path in rows["coordinates_npz_path"])
    assert result.tile_artifacts == []
    assert result.slide_artifacts == []
    assert result.patient_artifacts == []
    assert model._backend is None  # tiling only: the model was never loaded


def test_full_patient_run_rejects_slides_without_patient_ids_before_tiling(tmp_path):
    output_dir = tmp_path / "out"
    model, pipeline = _model_and_pipeline(output_dir)

    with pytest.raises(ValueError, match="Patient-level models require a 'patient_id' for every slide"):
        pipeline.run(slides=_slides())

    assert model._backend is None  # nothing was loaded, so nothing was embedded
    assert not output_dir.exists()  # nor tiled
