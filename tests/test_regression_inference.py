import ast
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import types

import numpy as np
import pandas as pd
import pytest

from slide2vec.api import (
    EmbeddedSlide,
    ExecutionOptions,
    Model,
    Pipeline,
    PreprocessingConfig as BasePreprocessingConfig,
)
from slide2vec.artifacts import (
    load_array,
    load_metadata,
    write_hierarchical_embeddings,
    write_slide_embeddings,
    write_tile_embedding_metadata,
    write_tile_embeddings,
)
from slide2vec.configs.resources import config_resource, load_config

ROOT = Path(__file__).resolve().parents[1]


def PreprocessingConfig(*args, **kwargs):
    kwargs.setdefault("requested_spacing_um", 0.5)
    kwargs.setdefault("requested_tile_size_px", 224)
    return BasePreprocessingConfig(*args, **kwargs)


DEFAULT_PREPROCESSING = PreprocessingConfig()


def make_slide(
    sample_id: str,
    *,
    image_path: Path | None = None,
    mask_path: Path | None = None,
    spacing_at_level_0: float | None = None,
):
    return SimpleNamespace(
        sample_id=sample_id,
        image_path=image_path or Path(f"/tmp/{sample_id}.svs"),
        mask_path=mask_path,
        spacing_at_level_0=spacing_at_level_0,
    )


def test_pipeline_run_uses_distributed_embedding_path_when_num_gpus_is_greater_than_one(
    monkeypatch,
    tmp_path: Path,
):
    import slide2vec.inference as inference

    model = Model.from_preset("virchow2")
    slide = make_slide("slide-a")
    tiling_result = SimpleNamespace(
        x=np.array([0, 1]),
        y=np.array([2, 3]),
        tile_size_lv0=224,
    )
    captured = {}

    monkeypatch.setattr(
        inference,
        "_prepare_tiled_slides",
        lambda *args, **kwargs: ([slide], [tiling_result], tmp_path / "process_list.csv"),
    )
    monkeypatch.setattr(
        inference,
        "_run_distributed_embedding_stage",
        lambda *args, **kwargs: captured.update({"args": args, "kwargs": kwargs}),
    )
    monkeypatch.setattr(inference, "_validate_multi_gpu_execution", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        inference,
        "_collect_pipeline_artifacts",
        lambda *args, **kwargs: (["tile-artifact"], [], ["slide-artifact"]),
    )
    monkeypatch.setattr(inference, "_update_process_list_after_embedding", lambda *args, **kwargs: None)

    result = inference.run_pipeline(
        model,
        slides=[slide],
        preprocessing=DEFAULT_PREPROCESSING,
        execution=ExecutionOptions(output_dir=tmp_path, num_gpus=2),
    )

    assert captured["kwargs"]["output_dir"] == tmp_path
    assert captured["kwargs"]["execution"].num_gpus == 2
    assert result.tile_artifacts == ["tile-artifact"]
    assert result.slide_artifacts == ["slide-artifact"]

def test_run_pipeline_distributed_branch_delegates_to_distributed_collection_helper(monkeypatch, tmp_path: Path):
    import slide2vec.inference as inference

    slide = make_slide("slide-a")
    tiling_result = SimpleNamespace(x=np.array([0]), y=np.array([1]), tile_size_lv0=224)

    monkeypatch.setattr(
        inference,
        "_prepare_tiled_slides",
        lambda *args, **kwargs: ([slide], [tiling_result], tmp_path / "process_list.csv"),
    )
    monkeypatch.setattr(inference, "_validate_multi_gpu_execution", lambda *args, **kwargs: None)

    captured = {}

    def fake_collect(*, model, successful_slides, process_list_path, preprocessing, execution, output_dir):
        captured["model"] = model
        captured["successful_slides"] = successful_slides
        captured["process_list_path"] = process_list_path
        captured["preprocessing"] = preprocessing
        captured["execution"] = execution
        captured["output_dir"] = output_dir
        return ["tile-artifact"], [], ["slide-artifact"]

    monkeypatch.setattr(inference, "_collect_distributed_pipeline_artifacts", fake_collect)

    result = inference.run_pipeline(
        Model.from_preset("virchow2"),
        slides=[slide],
        preprocessing=DEFAULT_PREPROCESSING,
        execution=ExecutionOptions(output_dir=tmp_path, num_gpus=2),
    )

    assert captured["successful_slides"] == [slide]
    assert captured["process_list_path"] == tmp_path / "process_list.csv"
    assert isinstance(captured["preprocessing"], BasePreprocessingConfig)
    assert captured["output_dir"] == tmp_path
    assert captured["execution"].num_gpus == 2
    assert result.tile_artifacts == ["tile-artifact"]
    assert result.slide_artifacts == ["slide-artifact"]

def test_collect_distributed_pipeline_artifacts_runs_stage_collects_and_updates(monkeypatch, tmp_path: Path):
    import slide2vec.inference as inference

    model = Model.from_preset("prism")
    slide = make_slide("slide-a")
    process_list_path = tmp_path / "process_list.csv"
    execution = ExecutionOptions(output_dir=tmp_path, num_gpus=2, output_format="npz", save_tile_embeddings=True)

    captured = {}

    def fake_run_stage(*, model, successful_slides, preprocessing, execution, output_dir):
        captured["run_stage"] = {
            "model": model,
            "successful_slides": successful_slides,
            "preprocessing": preprocessing,
            "execution": execution,
            "output_dir": output_dir,
        }

    def fake_collect(slides, *, output_dir, output_format, include_tile_embeddings, include_hierarchical_embeddings, include_slide_embeddings):
        captured["collect"] = {
            "slides": slides,
            "output_dir": output_dir,
            "output_format": output_format,
            "include_tile_embeddings": include_tile_embeddings,
            "include_hierarchical_embeddings": include_hierarchical_embeddings,
            "include_slide_embeddings": include_slide_embeddings,
        }
        return ["tile-artifact"], [], ["slide-artifact"]

    def fake_update(
        process_list_path_arg,
        *,
        successful_slides,
        persist_tile_embeddings,
        persist_hierarchical_embeddings,
        include_slide_embeddings,
        encoder_name,
        output_variant,
        tile_artifacts,
        hierarchical_artifacts,
        slide_artifacts,
    ):
        captured["update"] = {
            "process_list_path": process_list_path_arg,
            "successful_slides": successful_slides,
            "persist_tile_embeddings": persist_tile_embeddings,
            "persist_hierarchical_embeddings": persist_hierarchical_embeddings,
            "include_slide_embeddings": include_slide_embeddings,
            "encoder_name": encoder_name,
            "output_variant": output_variant,
            "tile_artifacts": tile_artifacts,
            "hierarchical_artifacts": hierarchical_artifacts,
            "slide_artifacts": slide_artifacts,
        }

    monkeypatch.setattr(inference, "_run_distributed_embedding_stage", fake_run_stage)
    monkeypatch.setattr(inference, "_collect_pipeline_artifacts", fake_collect)
    monkeypatch.setattr(inference, "_update_process_list_after_embedding", fake_update)

    tile_artifacts, hierarchical_artifacts, slide_artifacts = inference._collect_distributed_pipeline_artifacts(
        model=model,
        successful_slides=[slide],
        process_list_path=process_list_path,
        preprocessing=DEFAULT_PREPROCESSING,
        execution=execution,
        output_dir=tmp_path,
    )

    assert captured["run_stage"]["model"] is model
    assert captured["run_stage"]["successful_slides"] == [slide]
    assert captured["run_stage"]["output_dir"] == tmp_path

    assert captured["collect"]["slides"] == [slide]
    assert captured["collect"]["output_dir"] == tmp_path
    assert captured["collect"]["output_format"] == "npz"
    assert captured["collect"]["include_tile_embeddings"] is True
    assert captured["collect"]["include_hierarchical_embeddings"] is False
    assert captured["collect"]["include_slide_embeddings"] is True

    assert captured["update"]["process_list_path"] == process_list_path
    assert captured["update"]["successful_slides"] == [slide]
    assert captured["update"]["persist_tile_embeddings"] is True
    assert captured["update"]["persist_hierarchical_embeddings"] is False
    assert captured["update"]["include_slide_embeddings"] is True
    assert captured["update"]["encoder_name"] == "prism"
    assert captured["update"]["output_variant"] == "default"
    assert captured["update"]["tile_artifacts"] == ["tile-artifact"]
    assert captured["update"]["hierarchical_artifacts"] == []
    assert captured["update"]["slide_artifacts"] == ["slide-artifact"]

    assert tile_artifacts == ["tile-artifact"]
    assert hierarchical_artifacts == []
    assert slide_artifacts == ["slide-artifact"]


def test_collect_distributed_pipeline_artifacts_uses_hierarchical_artifacts_for_hierarchical_preprocessing(
    monkeypatch,
    tmp_path: Path,
):
    import slide2vec.inference as inference

    slide = make_slide("slide-a")
    write_hierarchical_embeddings(
        "slide-a",
        np.zeros((1, 2, 4), dtype=np.float32),
        output_dir=tmp_path,
        output_format="pt",
        metadata={"image_path": "/tmp/slide-a.svs"},
    )
    preprocessing = replace(
        DEFAULT_PREPROCESSING,
        requested_region_size_px=448,
        region_tile_multiple=2,
    )
    execution = ExecutionOptions(output_dir=tmp_path, num_gpus=2, output_format="pt")
    model = SimpleNamespace(name="virchow2", level="tile")

    monkeypatch.setattr(inference, "_run_distributed_embedding_stage", lambda *args, **kwargs: None)
    monkeypatch.setattr(inference, "_update_process_list_after_embedding", lambda *args, **kwargs: None)

    tile_artifacts, hierarchical_artifacts, slide_artifacts = inference._collect_distributed_pipeline_artifacts(
        model=model,
        successful_slides=[slide],
        process_list_path=tmp_path / "process_list.csv",
        preprocessing=preprocessing,
        execution=execution,
        output_dir=tmp_path,
    )

    assert tile_artifacts == []
    assert [artifact.sample_id for artifact in hierarchical_artifacts] == ["slide-a"]
    assert slide_artifacts == []


def test_has_complete_local_embedding_outputs_uses_hierarchical_artifacts_for_hierarchical_preprocessing(
    tmp_path: Path,
):
    import slide2vec.inference as inference

    write_hierarchical_embeddings(
        "slide-a",
        np.zeros((1, 2, 4), dtype=np.float32),
        output_dir=tmp_path,
        output_format="pt",
        metadata={"image_path": "/tmp/slide-a.svs"},
    )

    assert inference._has_complete_local_embedding_outputs(
        "slide-a",
        output_dir=tmp_path,
        output_format="pt",
        persist_tile_embeddings=True,
        persist_hierarchical_embeddings=True,
        include_slide_embeddings=False,
        save_latents=False,
    )


@pytest.mark.parametrize(
    ("persist_hierarchical_embeddings", "include_slide_embeddings", "expected_feature_kind"),
    [
        (False, False, "tile"),
        (True, False, "hierarchical"),
        (False, True, "slide"),
    ],
)
def test_update_process_list_after_embedding_writes_feature_provenance(
    persist_hierarchical_embeddings: bool,
    include_slide_embeddings: bool,
    expected_feature_kind: str,
    tmp_path: Path,
):
    import slide2vec.inference as inference

    slide = make_slide("slide-a")
    process_list_path = tmp_path / "process_list.csv"
    process_list_path.write_text(
        "sample_id,annotation,image_path,mask_path,requested_backend,backend,spacing_at_level_0,tiling_status,num_tiles,coordinates_npz_path,coordinates_meta_path,feature_status,error,traceback\n"
        "slide-a,tissue,/tmp/slide-a.svs,,asap,asap,,success,1,/tmp/slide-a.coordinates.npz,/tmp/slide-a.coordinates.meta.json,tbp,,\n",
        encoding="utf-8",
    )
    slide_artifacts = []
    if include_slide_embeddings:
        artifact = write_slide_embeddings(
            "slide-a",
            np.zeros((4,), dtype=np.float32),
            output_dir=tmp_path,
            output_format="pt",
            metadata={"image_path": "/tmp/slide-a.svs"},
        )
        tile_artifacts = []
        hierarchical_artifacts = []
        slide_artifacts = [artifact]
    elif persist_hierarchical_embeddings:
        artifact = write_hierarchical_embeddings(
            "slide-a",
            np.zeros((1, 2, 4), dtype=np.float32),
            output_dir=tmp_path,
            output_format="pt",
            metadata={"image_path": "/tmp/slide-a.svs"},
        )
        tile_artifacts = []
        hierarchical_artifacts = [artifact]
    else:
        artifact = write_tile_embeddings(
            "slide-a",
            np.zeros((1, 4), dtype=np.float32),
            output_dir=tmp_path,
            output_format="pt",
            metadata={"image_path": "/tmp/slide-a.svs"},
        )
        tile_artifacts = [artifact]
        hierarchical_artifacts = []

    inference._update_process_list_after_embedding(
        process_list_path,
        successful_slides=[slide],
        persist_tile_embeddings=not persist_hierarchical_embeddings,
        persist_hierarchical_embeddings=persist_hierarchical_embeddings,
        include_slide_embeddings=include_slide_embeddings,
        encoder_name="virchow2" if not include_slide_embeddings else "prism",
        output_variant="cls" if not include_slide_embeddings else "default",
        tile_artifacts=tile_artifacts,
        hierarchical_artifacts=hierarchical_artifacts,
        slide_artifacts=slide_artifacts,
    )

    recorded = pd.read_csv(process_list_path).set_index("sample_id")
    assert recorded.loc["slide-a", "feature_status"] == "success"
    assert recorded.loc["slide-a", "feature_path"] == str(artifact.path)
    assert recorded.loc["slide-a", "encoder_name"] == ("virchow2" if not include_slide_embeddings else "prism")
    assert recorded.loc["slide-a", "output_variant"] == ("cls" if not include_slide_embeddings else "default")
    assert recorded.loc["slide-a", "feature_kind"] == expected_feature_kind


def test_model_embed_slide_updates_process_list_feature_status_and_path_in_distributed_path(
    monkeypatch,
    tmp_path: Path,
):
    import slide2vec.inference as inference

    monkeypatch.chdir(tmp_path)
    output_dir = Path("relative-output")
    slide_path = tmp_path / "slide-a.svs"
    process_list_path = output_dir / "process_list.csv"
    process_list_path.parent.mkdir(parents=True, exist_ok=True)
    process_list_path.write_text(
        "sample_id,annotation,image_path,mask_path,spacing_at_level_0,tiling_status,num_tiles,coordinates_npz_path,coordinates_meta_path,error,traceback\n"
        "slide-a,tissue,/tmp/slide-a.svs,,,success,1,/tmp/slide-a.coordinates.npz,/tmp/slide-a.coordinates.meta.json,,\n",
        encoding="utf-8",
    )
    slide_record = make_slide("slide-a", image_path=slide_path)
    tiling_result = SimpleNamespace(
        x=np.array([0], dtype=np.int64),
        y=np.array([1], dtype=np.int64),
        tile_size_lv0=224,
        coordinates_npz_path=Path("/tmp/slide-a.coordinates.npz"),
        coordinates_meta_path=Path("/tmp/slide-a.coordinates.meta.json"),
    )
    embedded = EmbeddedSlide(
        sample_id="slide-a",
        tile_embeddings=np.zeros((1, 4), dtype=np.float32),
        slide_embedding=np.zeros((8,), dtype=np.float32),
        x=np.array([0], dtype=np.int64),
        y=np.array([1], dtype=np.int64),
        tile_size_lv0=224,
        image_path=slide_path,
        mask_path=None,
        num_tiles=1,
    )

    monkeypatch.setattr(
        inference,
        "_prepare_tiled_slides",
        lambda *args, **kwargs: ([slide_record], [tiling_result], process_list_path),
    )
    monkeypatch.setattr(inference, "_validate_multi_gpu_execution", lambda *args, **kwargs: None)
    monkeypatch.setattr(inference, "_select_embedding_path", lambda **kwargs: [embedded])

    def fake_persist_embedded_slide(model, embedded_slide, tiling_result, *, preprocessing, execution):
        run_dir = Path(execution.output_dir)
        tile_artifact = write_tile_embeddings(
            embedded_slide.sample_id,
            embedded_slide.tile_embeddings,
            output_dir=run_dir,
            output_format=execution.output_format,
            metadata={"image_path": str(embedded_slide.image_path)},
        )
        slide_artifact = write_slide_embeddings(
            embedded_slide.sample_id,
            embedded_slide.slide_embedding,
            output_dir=run_dir,
            output_format=execution.output_format,
            metadata={"image_path": str(embedded_slide.image_path)},
        )
        return tile_artifact, slide_artifact

    monkeypatch.setattr(inference, "_persist_embedded_slide", fake_persist_embedded_slide)

    model = Model.from_preset("prism")
    result = model.embed_slide(
        slide_path,
        preprocessing=DEFAULT_PREPROCESSING,
        execution=ExecutionOptions(output_dir=output_dir, num_gpus=2),
    )

    assert result.sample_id == "slide-a"
    recorded = pd.read_csv(process_list_path).set_index("sample_id")
    assert recorded.loc["slide-a", "feature_status"] == "success"
    assert recorded.loc["slide-a", "feature_path"] == str((tmp_path / "relative-output" / "slide_embeddings" / "slide-a.pt").resolve())


def test_run_pipeline_skips_zero_tile_slides_and_counts_only_embeddable_slides(monkeypatch, tmp_path: Path):
    import slide2vec.inference as inference
    import slide2vec.progress as progress

    class RecordingReporter:
        def __init__(self):
            self.events = []

        def emit(self, event):
            self.events.append(event)

        def close(self):
            return None

        def write_log(self, message, *, stream=None):
            return None

    reporter = RecordingReporter()
    slide_zero = make_slide("slide-zero")
    slide_full = make_slide("slide-full")
    zero_tiling = SimpleNamespace(
        x=np.array([], dtype=np.int64),
        y=np.array([], dtype=np.int64),
        tile_size_lv0=224,
        backend="asap",
        coordinates_npz_path=Path("/tmp/slide-zero.coordinates.npz"),
        coordinates_meta_path=Path("/tmp/slide-zero.coordinates.meta.json"),
    )
    full_tiling = SimpleNamespace(
        x=np.array([0, 1], dtype=np.int64),
        y=np.array([2, 3], dtype=np.int64),
        tile_size_lv0=224,
        backend="asap",
        coordinates_npz_path=Path("/tmp/slide-full.coordinates.npz"),
        coordinates_meta_path=Path("/tmp/slide-full.coordinates.meta.json"),
    )
    process_list_path = tmp_path / "process_list.csv"
    process_list_path.write_text(
        "sample_id,annotation,image_path,mask_path,spacing_at_level_0,tiling_status,num_tiles,coordinates_npz_path,coordinates_meta_path,error,traceback\n"
        "slide-zero,tissue,/tmp/slide-zero.svs,,,success,0,/tmp/slide-zero.coordinates.npz,/tmp/slide-zero.coordinates.meta.json,,\n"
        "slide-full,tissue,/tmp/slide-full.svs,,,success,2,/tmp/slide-full.coordinates.npz,/tmp/slide-full.coordinates.meta.json,,\n",
        encoding="utf-8",
    )

    embedded_full = EmbeddedSlide(
        sample_id="slide-full",
        tile_embeddings=np.array([[1.0, 2.0]], dtype=np.float32),
        slide_embedding=np.array([9.0, 10.0], dtype=np.float32),
        x=np.array([0], dtype=np.int64),
        y=np.array([1], dtype=np.int64),
        tile_size_lv0=224,
        image_path=Path("/tmp/slide-full.svs"),
        mask_path=None,
        num_tiles=1,
    )
    captured = {}

    monkeypatch.setattr(
        inference,
        "_prepare_tiled_slides",
        lambda *args, **kwargs: ([slide_zero, slide_full], [zero_tiling, full_tiling], process_list_path),
    )

    def fake_compute_embedded_slides(model, slide_records, tiling_results, *, preprocessing, execution, on_embedded_slide=None):
        captured["slide_records"] = [slide.sample_id for slide in slide_records]
        captured["tiling_results"] = [result.x.shape[0] for result in tiling_results]
        if on_embedded_slide is not None:
            on_embedded_slide(slide_full, full_tiling, embedded_full)
        return [embedded_full]

    monkeypatch.setattr(inference, "_compute_embedded_slides", fake_compute_embedded_slides)
    monkeypatch.setattr(inference, "_collect_pipeline_artifacts", lambda *args, **kwargs: (["tile-artifact"], [], ["slide-artifact"]))
    monkeypatch.setattr(inference, "_update_process_list_after_embedding", lambda *args, **kwargs: None)

    model = SimpleNamespace(
        name="prism",
        level="slide",
        _requested_device="cpu",
        _load_backend=lambda: SimpleNamespace(),
    )

    with progress.activate_progress_reporter(reporter):
        result = inference.run_pipeline(
            model,
            slides=[slide_zero, slide_full],
            preprocessing=DEFAULT_PREPROCESSING,
            execution=ExecutionOptions(output_dir=tmp_path, save_tile_embeddings=True),
        )

    zero_meta = load_metadata(tmp_path / "tile_embeddings" / "slide-zero.meta.json")
    assert not (tmp_path / "tile_embeddings" / "slide-zero.pt").exists()
    assert zero_meta["num_tiles"] == 0
    assert zero_meta["feature_dim"] is None
    assert captured["slide_records"] == ["slide-full"]
    assert captured["tiling_results"] == [2]
    assert result.tile_artifacts == ["tile-artifact"]
    assert result.slide_artifacts == ["slide-artifact"]

    embedding_finished = [event for event in reporter.events if event.kind == "embedding.finished"]
    assert embedding_finished
    assert embedding_finished[-1].payload["slide_count"] == 1
    assert embedding_finished[-1].payload["slides_completed"] == 1


def test_write_tile_embedding_metadata_creates_sidecar_without_tensor(tmp_path: Path):
    metadata_path = write_tile_embedding_metadata(
        "slide-zero",
        output_dir=tmp_path,
        output_format="pt",
        feature_dim=None,
        num_tiles=0,
        metadata={"image_path": "/tmp/slide-zero.svs"},
    )

    assert metadata_path.is_file()
    assert not (tmp_path / "tile_embeddings" / "slide-zero.pt").exists()
    metadata = load_metadata(metadata_path)
    assert metadata["sample_id"] == "slide-zero"
    assert metadata["num_tiles"] == 0
    assert metadata["feature_dim"] is None
    assert metadata["image_path"] == "/tmp/slide-zero.svs"


def test_collect_local_pipeline_artifacts_filters_none_artifacts(monkeypatch):
    import slide2vec.inference as inference

    embedded_slides = [
        EmbeddedSlide(
            sample_id="slide-a",
            tile_embeddings=np.zeros((1, 2), dtype=np.float32),
            slide_embedding=np.zeros((2,), dtype=np.float32),
            x=np.array([0], dtype=np.int64),
            y=np.array([0], dtype=np.int64),
            tile_size_lv0=224,
            image_path=Path("/tmp/slide-a.svs"),
            mask_path=None,
        ),
        EmbeddedSlide(
            sample_id="slide-b",
            tile_embeddings=np.zeros((1, 2), dtype=np.float32),
            slide_embedding=None,
            x=np.array([1], dtype=np.int64),
            y=np.array([1], dtype=np.int64),
            tile_size_lv0=224,
            image_path=Path("/tmp/slide-b.svs"),
            mask_path=None,
        ),
    ]
    tiling_results = [SimpleNamespace(), SimpleNamespace()]

    responses = [
        ("tile-a", "slide-a"),
        (None, "slide-b"),
    ]
    monkeypatch.setattr(inference, "_persist_embedded_slide", lambda *args, **kwargs: responses.pop(0))

    tile_artifacts, hierarchical_artifacts, slide_artifacts = inference._collect_local_pipeline_artifacts(
        model=SimpleNamespace(),
        embedded_slides=embedded_slides,
        tiling_results=tiling_results,
        preprocessing=DEFAULT_PREPROCESSING,
        execution=ExecutionOptions(output_dir=Path("/tmp")),
    )

    assert tile_artifacts == ["tile-a"]
    assert hierarchical_artifacts == []
    assert slide_artifacts == ["slide-a", "slide-b"]


def test_make_embedded_slide_carries_tiling_artifact_fields():
    import slide2vec.inference as inference

    slide = make_slide("slide-a")
    tiling_result = SimpleNamespace(
        x=np.array([0], dtype=np.int64),
        y=np.array([1], dtype=np.int64),
        tile_size_lv0=224,
        num_tiles=7,
        mask_preview_path=Path("/tmp/slide-a-mask-preview.png"),
        tiling_preview_path=Path("/tmp/slide-a-tiling-preview.png"),
    )

    embedded = inference._make_embedded_slide(
        slide=slide,
        tiling_result=tiling_result,
        tile_embeddings=np.zeros((1, 2), dtype=np.float32),
        slide_embedding=None,
    )

    assert embedded.num_tiles == 7
    assert embedded.mask_preview_path == Path("/tmp/slide-a-mask-preview.png")
    assert embedded.tiling_preview_path == Path("/tmp/slide-a-tiling-preview.png")


def test_run_pipeline_local_branch_uses_incremental_persist_callback(monkeypatch, tmp_path: Path):
    import slide2vec.inference as inference

    slide_record = make_slide("slide-a")
    tiling_result = SimpleNamespace(
        x=np.array([0]),
        y=np.array([1]),
        tile_size_lv0=224,
    )
    embedded = EmbeddedSlide(
        sample_id="slide-a",
        tile_embeddings=np.zeros((1, 2), dtype=np.float32),
        slide_embedding=None,
        x=np.array([0], dtype=np.int64),
        y=np.array([1], dtype=np.int64),
        tile_size_lv0=224,
        image_path=Path("/tmp/slide-a.svs"),
        mask_path=None,
    )

    monkeypatch.setattr(
        inference,
        "_prepare_tiled_slides",
        lambda *args, **kwargs: ([slide_record], [tiling_result], tmp_path / "process_list.csv"),
    )
    monkeypatch.setattr(
        inference,
        "_compute_embedded_slides",
        lambda *args, **kwargs: [embedded],
    )

    captured = {}

    def fake_build_callback(*, model, preprocessing, execution, process_list_path):
        captured["model"] = model
        captured["preprocessing"] = preprocessing
        captured["execution"] = execution
        captured["process_list_path"] = process_list_path
        return None, [], []

    monkeypatch.setattr(inference, "_build_incremental_persist_callback", fake_build_callback)
    monkeypatch.setattr(
        inference,
        "_collect_pipeline_artifacts",
        lambda *args, **kwargs: (["tile-artifact"], [], ["slide-artifact"]),
    )
    monkeypatch.setattr(inference, "_update_process_list_after_embedding", lambda *args, **kwargs: None)

    result = inference.run_pipeline(
        Model.from_preset("virchow2"),
        slides=[slide_record],
        preprocessing=DEFAULT_PREPROCESSING,
        execution=ExecutionOptions(output_dir=tmp_path, num_gpus=1),
    )

    assert captured["process_list_path"] == tmp_path / "process_list.csv"
    assert result.tile_artifacts == ["tile-artifact"]
    assert result.slide_artifacts == ["slide-artifact"]


def test_run_pipeline_local_branch_persists_completed_slides_before_later_failure(monkeypatch, tmp_path: Path):
    import slide2vec.inference as inference

    slides = [make_slide("slide-a"), make_slide("slide-b")]
    tiling_results = [
        SimpleNamespace(x=np.array([0]), y=np.array([1]), tile_size_lv0=224),
        SimpleNamespace(x=np.array([2]), y=np.array([3]), tile_size_lv0=224),
    ]
    process_list_path = tmp_path / "process_list.csv"
    process_list_path.write_text(
        "sample_id,annotation,image_path,mask_path,requested_backend,backend,spacing_at_level_0,tiling_status,num_tiles,coordinates_npz_path,coordinates_meta_path,feature_status,error,traceback\n"
        "slide-a,tissue,/tmp/slide-a.svs,,asap,asap,,success,1,/tmp/slide-a.coordinates.npz,/tmp/slide-a.coordinates.meta.json,tbp,,\n"
        "slide-b,tissue,/tmp/slide-b.svs,,asap,asap,,success,1,/tmp/slide-b.coordinates.npz,/tmp/slide-b.coordinates.meta.json,tbp,,\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        inference,
        "_prepare_tiled_slides",
        lambda *args, **kwargs: (slides, tiling_results, process_list_path),
    )
    monkeypatch.setattr(
        inference,
        "_compute_tile_embeddings_for_slide",
        lambda _loaded, _model, slide, _tiling_result, **_kwargs: (
            np.array([[1.0, 2.0]], dtype=np.float32)
            if slide.sample_id == "slide-a"
            else (_ for _ in ()).throw(RuntimeError("boom"))
        ),
    )

    model = SimpleNamespace(
        name="virchow2",
        level="tile",
        _requested_device="cpu",
        _load_backend=lambda: SimpleNamespace(feature_dim=2, device="cpu", model=SimpleNamespace()),
    )

    with pytest.raises(RuntimeError, match="boom"):
        inference.run_pipeline(
            model,
            slides=slides,
            preprocessing=DEFAULT_PREPROCESSING,
            execution=ExecutionOptions(output_dir=tmp_path, output_format="npz", num_gpus=1),
        )

    persisted = load_array(tmp_path / "tile_embeddings" / "slide-a.npz")
    np.testing.assert_array_equal(persisted, np.array([[1.0, 2.0]], dtype=np.float32))
    assert not (tmp_path / "tile_embeddings" / "slide-b.npz").exists()


def test_run_pipeline_resume_skips_successful_local_embeddings(monkeypatch, tmp_path: Path):
    import slide2vec.inference as inference

    slides = [make_slide("slide-a"), make_slide("slide-b")]
    tiling_results = [
        SimpleNamespace(x=np.array([0]), y=np.array([1]), tile_size_lv0=224),
        SimpleNamespace(x=np.array([2]), y=np.array([3]), tile_size_lv0=224),
    ]
    process_list_path = tmp_path / "process_list.csv"
    process_list_path.write_text(
        "sample_id,annotation,image_path,mask_path,requested_backend,backend,spacing_at_level_0,tiling_status,num_tiles,coordinates_npz_path,coordinates_meta_path,feature_status,error,traceback\n"
        "slide-a,tissue,/tmp/slide-a.svs,,auto,asap,,success,1,/tmp/slide-a.coordinates.npz,/tmp/slide-a.coordinates.meta.json,success,,\n"
        "slide-b,tissue,/tmp/slide-b.svs,,auto,asap,,success,1,/tmp/slide-b.coordinates.npz,/tmp/slide-b.coordinates.meta.json,tbp,,\n",
        encoding="utf-8",
    )
    write_tile_embeddings(
        "slide-a",
        np.array([[9.0, 9.0]], dtype=np.float32),
        output_dir=tmp_path,
        output_format="npz",
        metadata={"image_path": "/tmp/slide-a.svs"},
        tile_index=np.array([0], dtype=np.int64),
    )

    monkeypatch.setattr(
        inference,
        "_prepare_tiled_slides",
        lambda *args, **kwargs: (slides, tiling_results, process_list_path),
    )

    computed_sample_ids = []

    def fake_compute_tile_embeddings(_loaded, _model, slide, _tiling_result, **_kwargs):
        computed_sample_ids.append(slide.sample_id)
        return np.array([[1.0, 2.0]], dtype=np.float32)

    monkeypatch.setattr(inference, "_compute_tile_embeddings_for_slide", fake_compute_tile_embeddings)

    model = SimpleNamespace(
        name="virchow2",
        level="tile",
        _requested_device="cpu",
        _load_backend=lambda: SimpleNamespace(feature_dim=2, device="cpu", model=SimpleNamespace()),
    )

    result = inference.run_pipeline(
        model,
        slides=slides,
        preprocessing=replace(DEFAULT_PREPROCESSING, resume=True),
        execution=ExecutionOptions(output_dir=tmp_path, output_format="npz", num_gpus=1),
    )

    assert computed_sample_ids == ["slide-b"]
    assert [artifact.sample_id for artifact in result.tile_artifacts] == ["slide-a", "slide-b"]
    assert result.slide_artifacts == []


def test_run_pipeline_local_persists_completed_embeddings_before_later_slide_failure(monkeypatch, tmp_path: Path):
    pytest.importorskip("torch")
    import slide2vec.inference as inference

    slides = [make_slide("slide-a"), make_slide("slide-b")]
    tiling_results = [
        SimpleNamespace(
            x=np.array([0, 1]),
            y=np.array([2, 3]),
            tile_size_lv0=224,
            coordinates_npz_path=Path("/tmp/slide-a.coordinates.npz"),
            coordinates_meta_path=Path("/tmp/slide-a.coordinates.meta.json"),
        ),
        SimpleNamespace(
            x=np.array([4, 5]),
            y=np.array([6, 7]),
            tile_size_lv0=224,
            coordinates_npz_path=Path("/tmp/slide-b.coordinates.npz"),
            coordinates_meta_path=Path("/tmp/slide-b.coordinates.meta.json"),
        ),
    ]
    process_list_path = tmp_path / "process_list.csv"
    process_list_path.write_text(
        "sample_id,annotation,image_path,mask_path,spacing_at_level_0,tiling_status,num_tiles,coordinates_npz_path,coordinates_meta_path,error,traceback\n"
        "slide-a,tissue,/tmp/slide-a.svs,,,"  # spacing_at_level_0
        "success,2,/tmp/slide-a.coordinates.npz,/tmp/slide-a.coordinates.meta.json,,\n"
        "slide-b,tissue,/tmp/slide-b.svs,,,"
        "success,2,/tmp/slide-b.coordinates.npz,/tmp/slide-b.coordinates.meta.json,,\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        inference,
        "_prepare_tiled_slides",
        lambda *args, **kwargs: (slides, tiling_results, process_list_path),
    )

    def fake_compute_tile_embeddings(loaded, model, slide, tiling_result, *, preprocessing, execution, tile_indices=None):
        if slide.sample_id == "slide-b":
            raise RuntimeError("embedding boom")
        return np.zeros((2, 4), dtype=np.float32)

    monkeypatch.setattr(inference, "_compute_tile_embeddings_for_slide", fake_compute_tile_embeddings)
    monkeypatch.setattr(
        inference,
        "_aggregate_tile_embeddings_for_slide",
        lambda *args, **kwargs: (np.zeros((8,), dtype=np.float32), None),
    )

    model = SimpleNamespace(
        name="prism",
        level="slide",
        _requested_device="cpu",
        _load_backend=lambda: SimpleNamespace(),
    )

    with pytest.raises(RuntimeError, match="embedding boom"):
        inference.run_pipeline(
            model,
            slides=slides,
            preprocessing=DEFAULT_PREPROCESSING,
            execution=ExecutionOptions(output_dir=tmp_path, save_tile_embeddings=True),
        )

    assert (tmp_path / "tile_embeddings" / "slide-a.pt").is_file()
    assert (tmp_path / "tile_embeddings" / "slide-a.meta.json").is_file()
    assert (tmp_path / "slide_embeddings" / "slide-a.pt").is_file()
    assert (tmp_path / "slide_embeddings" / "slide-a.meta.json").is_file()
    assert not (tmp_path / "slide_embeddings" / "slide-b.pt").exists()

    process_df = pd.read_csv(process_list_path)
    recorded = process_df.set_index("sample_id")
    assert recorded.loc["slide-a", "feature_status"] == "success"
    assert recorded.loc["slide-a", "aggregation_status"] == "success"
    assert recorded.loc["slide-b", "feature_status"] == "tbp"
    assert recorded.loc["slide-b", "aggregation_status"] == "tbp"


def test_tile_slides_forwards_spacing_at_level_0_to_hs2p(monkeypatch, tmp_path: Path):
    import slide2vec.inference as inference

    captured = {}

    def fake_tile_slides(slides, **kwargs):
        captured["slides"] = list(slides)
        captured["kwargs"] = kwargs

    monkeypatch.setattr(inference, "tile_slides", fake_tile_slides)
    monkeypatch.setattr(
        inference.runtime_tiling,
        "build_hs2p_configs",
        lambda preprocessing: (
            SimpleNamespace(requested_backend="cucim"),
            "segmentation",
            "filtering",
            "preview",
            None,
            False,
        ),
    )

    slide = inference._coerce_slide_spec(
        {
            "sample_id": "slide-a",
            "image_path": "/tmp/slide-a.svs",
            "spacing_at_level_0": 0.25,
        }
    )

    inference._tile_slides(
        [slide],
        replace(DEFAULT_PREPROCESSING, on_the_fly=False),
        output_dir=tmp_path,
        num_workers=0,
    )

    assert captured["slides"][0].spacing_at_level_0 == pytest.approx(0.25)
    assert captured["kwargs"]["preview"] == "preview"
    assert captured["kwargs"]["save_tiles"] is True


def test_tile_slides_skips_saving_tiles_when_external_store_is_configured(monkeypatch, tmp_path: Path):
    import slide2vec.inference as inference

    captured = {}

    def fake_tile_slides(slides, **kwargs):
        captured["slides"] = list(slides)
        captured["kwargs"] = kwargs

    monkeypatch.setattr(inference, "tile_slides", fake_tile_slides)
    monkeypatch.setattr(
        inference.runtime_tiling,
        "build_hs2p_configs",
        lambda preprocessing: (
            SimpleNamespace(requested_backend="auto"),
            "segmentation",
            "filtering",
            "preview",
            None,
            False,
        ),
    )

    inference._tile_slides(
        [make_slide("slide-a")],
        replace(DEFAULT_PREPROCESSING, read_tiles_from=Path("/tmp/existing-tiles")),
        output_dir=tmp_path,
        num_workers=0,
    )

    assert captured["kwargs"]["save_tiles"] is False


def test_tile_slides_does_not_pre_resolve_backend_auto(monkeypatch, tmp_path: Path):
    import slide2vec.inference as inference
    import slide2vec.progress as progress
    from hs2p import progress as hs2p_progress

    class Reporter:
        def __init__(self):
            self.events = []

        def emit(self, event):
            self.events.append(event)

        def close(self):
            return None

    reporter = Reporter()
    captured = {}

    def fake_tile_slides(slides, **kwargs):
        captured["slides"] = list(slides)
        captured["kwargs"] = kwargs
        hs2p_progress.emit_progress("tissue.started", total=1)
        hs2p_progress.emit_progress(
            "tissue.progress",
            total=1,
            completed=1,
            failed=0,
            pending=0,
        )
        hs2p_progress.emit_progress(
            "tissue.finished",
            total=1,
            completed=1,
            failed=0,
            pending=0,
        )
        hs2p_progress.emit_progress(
            "backend.selected",
            sample_id="slide-a",
            backend="asap",
            reason="selected asap for auto backend",
        )
        hs2p_progress.emit_progress("tiling.started", total=1)
        hs2p_progress.emit_progress(
            "tiling.progress",
            total=1,
            completed=1,
            failed=0,
            pending=0,
            discovered_tiles=1,
        )
        hs2p_progress.emit_progress(
            "tiling.finished",
            total=1,
            completed=1,
            failed=0,
            pending=0,
            discovered_tiles=1,
            output_dir=str(tmp_path),
            process_list_path=str(tmp_path / "process_list.csv"),
            zero_tile_successes=0,
        )
        hs2p_progress.emit_progress("preview.started", total=1)
        hs2p_progress.emit_progress(
            "preview.progress",
            total=1,
            completed=1,
            failed=0,
            pending=0,
        )
        hs2p_progress.emit_progress(
            "preview.finished",
            total=1,
            completed=1,
            failed=0,
            pending=0,
        )

    assert not hasattr(inference, "resolve_backend")
    monkeypatch.setattr(inference, "tile_slides", fake_tile_slides)
    monkeypatch.setattr(
        inference.runtime_tiling,
        "build_hs2p_configs",
        lambda preprocessing: (
            SimpleNamespace(requested_backend="auto"),
            "segmentation",
            "filtering",
            "preview",
            None,
            False,
        ),
    )

    with progress.activate_progress_reporter(reporter):
        inference._tile_slides(
            [make_slide("slide-a")],
            replace(DEFAULT_PREPROCESSING, backend="auto", on_the_fly=False),
            output_dir=tmp_path,
            num_workers=0,
        )

    assert captured["slides"][0].sample_id == "slide-a"
    assert captured["kwargs"]["preview"] == "preview"
    assert [event.kind for event in reporter.events] == [
        "tissue.started",
        "tissue.progress",
        "tissue.finished",
        "backend.selected",
        "tiling.progress",
        "tiling.finished",
        "preview.started",
        "preview.progress",
        "preview.finished",
    ]


def test_build_hs2p_configs_constructs_preview_config():
    import slide2vec.runtime.tiling as runtime_tiling

    preprocessing = PreprocessingConfig(
        backend="asap",
        requested_spacing_um=0.5,
        requested_tile_size_px=224,
        tolerance=0.05,
        overlap=0.0,
        tissue_threshold=0.1,
        segmentation={"downsample": 64},
        filtering={"ref_tile_size": 224},
        preview={
            "save_mask_preview": True,
            "save_tiling_preview": False,
            "downsample": 32,
            "tissue_contour_color": (157, 219, 129),
            "mask_overlay_alpha": 0.5,
        },
    )

    tiling_cfg, segmentation_cfg, filtering_cfg, preview_cfg, read_coordinates_from, resume = (
        runtime_tiling.build_hs2p_configs(preprocessing)
    )

    assert tiling_cfg.backend == "asap"
    assert segmentation_cfg.downsample == 64
    assert filtering_cfg.ref_tile_size == 224
    assert preview_cfg.save_mask_preview is True
    assert preview_cfg.save_tiling_preview is False
    assert preview_cfg.downsample == 32
    assert preview_cfg.mask_overlay_color == (157, 219, 129)
    assert preview_cfg.mask_overlay_alpha == pytest.approx(0.5)
    assert read_coordinates_from is None
    assert resume is False


def test_num_tiles_accepts_x_y_tiling_result():
    import slide2vec.inference as inference

    tiling_result = SimpleNamespace(x=np.array([0, 2, 4], dtype=np.int64), y=np.array([1, 3, 5], dtype=np.int64))

    assert inference._num_tiles(tiling_result) == 3


def test_prepare_tiled_slides_records_spacing_at_level_0_in_process_list(monkeypatch, tmp_path: Path):
    import slide2vec.inference as inference

    process_list_path = tmp_path / "process_list.csv"
    process_list_path.write_text(
        "sample_id,annotation,image_path,mask_path,requested_backend,backend,tiling_status,num_tiles,coordinates_npz_path,coordinates_meta_path,error,traceback\n"
        "slide-a,tissue,/tmp/slide-a.svs,,asap,asap,success,1,/tmp/slide-a.coordinates.npz,/tmp/slide-a.coordinates.meta.json,,\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(inference, "_tile_slides", lambda *args, **kwargs: None)
    monkeypatch.setattr(inference, "load_tiling_result_from_row", lambda row: SimpleNamespace())

    slide = make_slide("slide-a", spacing_at_level_0=0.25)

    inference._prepare_tiled_slides(
        [slide],
        DEFAULT_PREPROCESSING,
        output_dir=tmp_path,
        num_workers=0,
    )

    recorded = pd.read_csv(process_list_path)
    assert recorded.loc[0, "spacing_at_level_0"] == pytest.approx(0.25)


def test_prepare_tiled_slides_records_preview_paths_in_process_list(monkeypatch, tmp_path: Path):
    import slide2vec.inference as inference

    process_list_path = tmp_path / "process_list.csv"
    process_list_path.write_text(
        "sample_id,annotation,image_path,mask_path,requested_backend,backend,tiling_status,num_tiles,coordinates_npz_path,coordinates_meta_path,error,traceback\n"
        "slide-a,tissue,/tmp/slide-a.svs,,asap,asap,success,1,/tmp/slide-a.coordinates.npz,/tmp/slide-a.coordinates.meta.json,,\n",
        encoding="utf-8",
    )

    tiling_artifacts = [
        SimpleNamespace(
            sample_id="slide-a",
            mask_preview_path=Path("/tmp/preview/mask/slide-a.png"),
            tiling_preview_path=Path("/tmp/preview/tiling/slide-a.png"),
        )
    ]

    monkeypatch.setattr(inference, "_tile_slides", lambda *args, **kwargs: tiling_artifacts)
    monkeypatch.setattr(inference, "load_tiling_result_from_row", lambda row: SimpleNamespace())

    slide = make_slide("slide-a")

    inference._prepare_tiled_slides(
        [slide],
        DEFAULT_PREPROCESSING,
        output_dir=tmp_path,
        num_workers=0,
    )

    recorded = pd.read_csv(process_list_path)
    assert Path(recorded.loc[0, "mask_preview_path"]) == Path("/tmp/preview/mask/slide-a.png").resolve()
    assert Path(recorded.loc[0, "tiling_preview_path"]) == Path("/tmp/preview/tiling/slide-a.png").resolve()


def test_record_slide_metadata_in_process_list_adds_backend_columns(monkeypatch, tmp_path: Path):
    import slide2vec.inference as inference

    process_list_path = tmp_path / "process_list.csv"
    process_list_path.write_text(
        "sample_id,annotation,image_path,mask_path,requested_backend,backend,tiling_status,num_tiles,coordinates_npz_path,coordinates_meta_path,error,traceback\n"
        "slide-a,tissue,/tmp/slide-a.svs,,auto,,success,1,/tmp/slide-a.coordinates.npz,/tmp/slide-a.coordinates.meta.json,,\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        inference,
        "load_tiling_result_from_row",
        lambda row: SimpleNamespace(backend="asap"),
    )

    inference._record_slide_metadata_in_process_list(
        process_list_path,
        [make_slide("slide-a")],
        preprocessing=DEFAULT_PREPROCESSING,
        tiling_artifacts=[],
    )

    recorded = pd.read_csv(process_list_path)
    assert "requested_backend" in recorded.columns
    assert "backend" in recorded.columns
    assert recorded.loc[0, "requested_backend"] == DEFAULT_PREPROCESSING.backend
    assert recorded.loc[0, "backend"] == "asap"


def test_resolve_slide_backend_uses_tiling_result_backend_for_auto():
    import slide2vec.runtime.tiling as runtime_tiling

    assert runtime_tiling.resolve_slide_backend(replace(DEFAULT_PREPROCESSING, backend="auto"), SimpleNamespace(backend="cucim")) == "cucim"
    assert runtime_tiling.resolve_slide_backend(replace(DEFAULT_PREPROCESSING, backend="auto"), SimpleNamespace(backend="asap")) == "asap"
    assert runtime_tiling.resolve_slide_backend(replace(DEFAULT_PREPROCESSING, backend="auto"), SimpleNamespace()) == "asap"
    assert runtime_tiling.resolve_slide_backend(replace(DEFAULT_PREPROCESSING, backend="cucim"), SimpleNamespace(backend="asap")) == "cucim"


def test_preload_asap_wholeslidedata_suppresses_noisy_import(monkeypatch, capfd):
    import os

    import slide2vec.inference as inference

    calls: list[str] = []

    def fake_import_module(name: str):
        calls.append(name)
        if name == "wholeslidedata":
            os.write(2, b"cuFile initialization failed\n")
            return SimpleNamespace()
        raise AssertionError(f"Unexpected import: {name}")

    monkeypatch.setattr(inference.importlib, "import_module", fake_import_module)

    inference._preload_asap_wholeslidedata(replace(DEFAULT_PREPROCESSING, backend="asap"))

    captured = capfd.readouterr()
    assert captured.err == ""
    assert calls == ["wholeslidedata"]


def test_configure_cucim_worker_stderr_wraps_existing_worker_init(monkeypatch):
    import slide2vec.inference as inference

    calls: list[tuple[str, int] | str] = []

    monkeypatch.setattr(inference, "_redirect_worker_output", lambda: calls.append("redirected"))

    def _existing(worker_id: int):
        calls.append(("existing", worker_id))

    loader_kwargs = {"num_workers": 3, "worker_init_fn": _existing}

    inference._configure_cucim_worker_stderr(loader_kwargs, backend="cucim")

    worker_init = loader_kwargs["worker_init_fn"]
    worker_init(5)

    assert calls == ["redirected", ("existing", 5)]


def test_configure_cucim_worker_stderr_skips_non_cucim_or_single_process_loader():
    import slide2vec.inference as inference

    loader_kwargs = {"num_workers": 0}
    inference._configure_cucim_worker_stderr(loader_kwargs, backend="cucim")
    assert "worker_init_fn" not in loader_kwargs

    loader_kwargs = {"num_workers": 4}
    inference._configure_cucim_worker_stderr(loader_kwargs, backend="asap")
    assert "worker_init_fn" not in loader_kwargs


def test_should_suppress_cucim_dataloader_stderr_only_for_multi_worker_cucim_collators():
    import slide2vec.inference as inference

    dataloader = SimpleNamespace(
        num_workers=4,
        collate_fn=SimpleNamespace(_reader=SimpleNamespace(_backend="cucim")),
    )
    assert inference._should_suppress_cucim_dataloader_stderr(dataloader) is True

    dataloader = SimpleNamespace(
        num_workers=0,
        collate_fn=SimpleNamespace(_reader=SimpleNamespace(_backend="cucim")),
    )
    assert inference._should_suppress_cucim_dataloader_stderr(dataloader) is False


    dataloader = SimpleNamespace(
        num_workers=4,
        collate_fn=SimpleNamespace(_reader=SimpleNamespace(_backend="asap")),
    )
    assert inference._should_suppress_cucim_dataloader_stderr(dataloader) is False


def test_load_successful_tiled_slides_preserves_spacing_at_level_0(monkeypatch, tmp_path: Path):
    import slide2vec.inference as inference

    process_list_path = tmp_path / "process_list.csv"
    process_list_path.write_text(
        "sample_id,annotation,image_path,mask_path,requested_backend,backend,spacing_at_level_0,tiling_status,num_tiles,coordinates_npz_path,coordinates_meta_path,error,traceback\n"
        "slide-a,tissue,/tmp/slide-a.svs,,auto,,0.25,success,1,/tmp/slide-a.coordinates.npz,/tmp/slide-a.coordinates.meta.json,,\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(inference, "load_tiling_result_from_row", lambda row: SimpleNamespace())

    slide_records, tiling_results = inference.load_successful_tiled_slides(tmp_path)

    assert len(slide_records) == 1
    assert slide_records[0].spacing_at_level_0 == pytest.approx(0.25)
    assert len(tiling_results) == 1

def test_embed_single_slide_distributed_uses_shared_slide_aggregation_helper(monkeypatch, tmp_path: Path):
    from contextlib import contextmanager

    import slide2vec.inference as inference

    slide = make_slide("slide-a")
    tiling_result = SimpleNamespace(
        x=np.array([0, 1]),
        y=np.array([2, 3]),
        tile_size_lv0=224,
        requested_spacing_um=0.5,
    )

    @contextmanager
    def fake_coordination_dir(work_dir: Path):
        yield work_dir / "coord"

    monkeypatch.setattr(inference, "_distributed_coordination_dir", fake_coordination_dir)
    monkeypatch.setattr(inference, "_run_distributed_direct_embedding_stage", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        inference.runtime_distributed,
        "load_tile_embedding_shards",
        lambda *_args, **_kwargs: [
            {
                "tile_index": np.array([0, 1], dtype=np.int64),
                "tile_embeddings": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
            }
        ],
    )

    loaded = SimpleNamespace(device="cpu", model=SimpleNamespace())
    model = SimpleNamespace(level="slide", _load_backend=lambda: loaded)
    captured = {}

    def fake_aggregate(loaded_arg, model_arg, slide_arg, tiling_result_arg, tile_embeddings_arg, *, preprocessing, execution):
        captured["loaded"] = loaded_arg
        captured["model"] = model_arg
        captured["slide"] = slide_arg
        captured["tiling_result"] = tiling_result_arg
        captured["tile_embeddings_shape"] = tile_embeddings_arg.shape
        captured["execution_num_gpus"] = execution.num_gpus
        return np.array([9.0, 8.0], dtype=np.float32), np.array([[1.0, 1.0]], dtype=np.float32)

    monkeypatch.setattr(inference, "_aggregate_tile_embeddings_for_slide", fake_aggregate)

    embedded = inference._embed_single_slide_distributed(
        model,
        slide=slide,
        tiling_result=tiling_result,
        preprocessing=DEFAULT_PREPROCESSING,
        execution=ExecutionOptions(num_gpus=2),
        work_dir=tmp_path,
    )

    assert captured["loaded"] is loaded
    assert captured["model"] is model
    assert captured["slide"] is slide
    assert captured["tiling_result"] is tiling_result
    assert captured["tile_embeddings_shape"] == (2, 2)
    assert captured["execution_num_gpus"] == 2
    np.testing.assert_array_equal(embedded.slide_embedding, np.array([9.0, 8.0], dtype=np.float32))
    np.testing.assert_array_equal(embedded.x, np.array([0, 1], dtype=np.int64))
    np.testing.assert_array_equal(embedded.y, np.array([2, 3], dtype=np.int64))

def test_embed_single_slide_distributed_skips_parent_backend_load_for_tile_models(monkeypatch, tmp_path: Path):
    from contextlib import contextmanager

    import slide2vec.inference as inference

    slide = make_slide("slide-a")
    tiling_result = SimpleNamespace(
        x=np.array([0, 1]),
        y=np.array([2, 3]),
        tile_size_lv0=224,
        requested_spacing_um=0.5,
    )

    @contextmanager
    def fake_coordination_dir(work_dir: Path):
        yield work_dir / "coord"

    monkeypatch.setattr(inference, "_distributed_coordination_dir", fake_coordination_dir)
    monkeypatch.setattr(inference, "_run_distributed_direct_embedding_stage", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        inference.runtime_distributed,
        "load_tile_embedding_shards",
        lambda *_args, **_kwargs: [
            {
                "tile_index": np.array([0, 1], dtype=np.int64),
                "tile_embeddings": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
            }
        ],
    )
    monkeypatch.setattr(
        inference.runtime_distributed,
        "merge_tile_embedding_shards",
        lambda shard_payloads: shard_payloads[0]["tile_embeddings"],
    )

    def fail_if_called(*args, **kwargs):
        raise AssertionError("tile encoders should not load the parent backend or aggregate slide features")

    model = SimpleNamespace(
        name="h0-mini",
        level="tile",
        _load_backend=fail_if_called,
    )
    monkeypatch.setattr(inference, "_aggregate_tile_embeddings_for_slide", fail_if_called)

    embedded = inference._embed_single_slide_distributed(
        model,
        slide=slide,
        tiling_result=tiling_result,
        preprocessing=DEFAULT_PREPROCESSING,
        execution=ExecutionOptions(num_gpus=2),
        work_dir=tmp_path,
    )

    np.testing.assert_array_equal(embedded.tile_embeddings, np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
    assert embedded.slide_embedding is None
    assert embedded.latents is None
    np.testing.assert_array_equal(embedded.x, np.array([0, 1], dtype=np.int64))
    np.testing.assert_array_equal(embedded.y, np.array([2, 3], dtype=np.int64))

def test_select_embedding_path_uses_local_compute_when_single_gpu(monkeypatch):
    import slide2vec.inference as inference

    slide = make_slide("slide-a")
    tiling_result = SimpleNamespace(x=np.array([0]), y=np.array([1]), tile_size_lv0=224)
    expected = [
        EmbeddedSlide(
            sample_id="slide-a",
            tile_embeddings=np.zeros((1, 2), dtype=np.float32),
            slide_embedding=None,
            x=np.array([0], dtype=np.int64),
        y=np.array([1], dtype=np.int64),
            tile_size_lv0=224,
            image_path=Path("/tmp/slide-a.svs"),
            mask_path=None,
        )
    ]

    monkeypatch.setattr(inference, "_compute_embedded_slides", lambda *args, **kwargs: expected)
    monkeypatch.setattr(
        inference,
        "_embed_single_slide_distributed",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("single-slide distributed path should not be used")),
    )
    monkeypatch.setattr(
        inference,
        "_embed_multi_slides_distributed",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("multi-slide distributed path should not be used")),
    )

    result = inference._select_embedding_path(
        model=Model.from_preset("virchow2"),
        slide_records=[slide],
        tiling_results=[tiling_result],
        preprocessing=DEFAULT_PREPROCESSING,
        execution=ExecutionOptions(output_dir=Path("/tmp"), num_gpus=1),
        work_dir=Path("/tmp"),
    )

    assert result == expected

def test_select_embedding_path_uses_single_slide_distributed_when_one_slide(monkeypatch, tmp_path: Path):
    import slide2vec.inference as inference

    slide = make_slide("slide-a")
    tiling_result = SimpleNamespace(x=np.array([0]), y=np.array([1]), tile_size_lv0=224)
    expected = EmbeddedSlide(
        sample_id="slide-a",
        tile_embeddings=np.zeros((1, 2), dtype=np.float32),
        slide_embedding=None,
        x=np.array([0], dtype=np.int64),
        y=np.array([1], dtype=np.int64),
        tile_size_lv0=224,
        image_path=Path("/tmp/slide-a.svs"),
        mask_path=None,
    )

    monkeypatch.setattr(inference, "_embed_single_slide_distributed", lambda *args, **kwargs: expected)
    monkeypatch.setattr(
        inference,
        "_embed_multi_slides_distributed",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("multi-slide distributed path should not be used")),
    )
    monkeypatch.setattr(
        inference,
        "_compute_embedded_slides",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("local path should not be used")),
    )

    result = inference._select_embedding_path(
        model=Model.from_preset("virchow2"),
        slide_records=[slide],
        tiling_results=[tiling_result],
        preprocessing=DEFAULT_PREPROCESSING,
        execution=ExecutionOptions(output_dir=tmp_path, num_gpus=2),
        work_dir=tmp_path,
    )

    assert result == [expected]

def test_select_embedding_path_uses_multi_slide_distributed_when_multiple_slides(monkeypatch, tmp_path: Path):
    import slide2vec.inference as inference

    slides = [
        make_slide("slide-a"),
        make_slide("slide-b"),
    ]
    tiling_results = [
        SimpleNamespace(x=np.array([0]), y=np.array([1]), tile_size_lv0=224),
        SimpleNamespace(x=np.array([2]), y=np.array([3]), tile_size_lv0=224),
    ]
    expected = ["a", "b"]

    monkeypatch.setattr(inference, "_embed_multi_slides_distributed", lambda *args, **kwargs: expected)
    monkeypatch.setattr(
        inference,
        "_embed_single_slide_distributed",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("single-slide distributed path should not be used")),
    )
    monkeypatch.setattr(
        inference,
        "_compute_embedded_slides",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("local path should not be used")),
    )

    result = inference._select_embedding_path(
        model=Model.from_preset("virchow2"),
        slide_records=slides,
        tiling_results=tiling_results,
        preprocessing=DEFAULT_PREPROCESSING,
        execution=ExecutionOptions(output_dir=tmp_path, num_gpus=2),
        work_dir=tmp_path,
    )

    assert result == expected

def test_inference_embed_tiles_requires_output_dir_before_loading_runtime(monkeypatch):
    import slide2vec.inference as inference

    model = SimpleNamespace(_load_backend=lambda: (_ for _ in ()).throw(AssertionError("should fail before loading model")))

    with pytest.raises(ValueError, match="ExecutionOptions.output_dir is required to persist tile embeddings"):
        inference.embed_tiles(
            model,
            slides=[{"sample_id": "slide-a", "image_path": "/tmp/slide-a.svs"}],
            tiling_results=[SimpleNamespace(x=np.array([0]), y=np.array([0]), tile_size_lv0=224)],
            execution=ExecutionOptions(),
        )

def test_make_embedded_slide_validates_coordinates_and_supports_tile_and_slide_outputs():
    import slide2vec.inference as inference

    slide = make_slide("slide-a")
    tiling_result = SimpleNamespace(
        x=np.array([0, 1]),
        y=np.array([2, 3]),
        tile_size_lv0=224,
    )
    tile_only = inference._make_embedded_slide(
        slide=slide,
        tiling_result=tiling_result,
        tile_embeddings=np.zeros((2, 4), dtype=np.float32),
    )
    assert tile_only.slide_embedding is None
    np.testing.assert_array_equal(tile_only.x, np.array([0, 1], dtype=np.int64))
    np.testing.assert_array_equal(tile_only.y, np.array([2, 3], dtype=np.int64))

    slide_level = inference._make_embedded_slide(
        slide=slide,
        tiling_result=tiling_result,
        tile_embeddings=np.zeros((2, 4), dtype=np.float32),
        slide_embedding=np.zeros((8,), dtype=np.float32),
        latents=np.zeros((3, 8), dtype=np.float32),
    )
    assert slide_level.slide_embedding.shape == (8,)
    assert slide_level.latents.shape == (3, 8)

    with pytest.raises(ValueError, match="Tile embedding count"):
        inference._make_embedded_slide(
            slide=slide,
            tiling_result=tiling_result,
            tile_embeddings=np.zeros((1, 4), dtype=np.float32),
        )

def test_distributed_enable_keeps_explicit_device_binding_without_unconditional_barrier():
    source = (ROOT / "slide2vec" / "distributed" / "__init__.py").read_text(encoding="utf-8")
    tree = ast.parse(source)

    init_process_group_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "dist"
        and node.func.attr == "init_process_group"
    ]
    assert init_process_group_calls

    device_keyword = next(
        (
            keyword
            for keyword in init_process_group_calls[0].keywords
            if keyword.arg == "device_id"
        ),
        None,
    )
    assert device_keyword is not None
    assert isinstance(device_keyword.value, ast.Call)
    assert isinstance(device_keyword.value.func, ast.Attribute)
    assert isinstance(device_keyword.value.func.value, ast.Name)
    assert device_keyword.value.func.value.id == "torch"
    assert device_keyword.value.func.attr == "device"

    barrier_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "dist"
        and node.func.attr == "barrier"
    ]
    assert not barrier_calls

def test_setup_distributed_helper_has_been_removed():
    source = (ROOT / "slide2vec" / "utils" / "config.py").read_text(encoding="utf-8")

    assert "def setup_distributed" not in source

def test_direct_embed_slides_allows_no_output_dir_and_optional_persistence(monkeypatch, tmp_path: Path):
    import slide2vec.inference as inference

    slide_record = make_slide("slide-a")
    tiling_result = SimpleNamespace(
        x=np.array([0, 1]),
        y=np.array([2, 3]),
        tile_size_lv0=224,
        coordinates_npz_path=Path("/tmp/slide-a.coordinates.npz"),
        coordinates_meta_path=Path("/tmp/slide-a.coordinates.meta.json"),
    )
    embedded = EmbeddedSlide(
        sample_id="slide-a",
        tile_embeddings=np.zeros((2, 4), dtype=np.float32),
        slide_embedding=np.zeros((8,), dtype=np.float32),
        x=np.array([0, 1], dtype=np.int64),
        y=np.array([2, 3], dtype=np.int64),
        tile_size_lv0=224,
        image_path=Path("/tmp/slide-a.svs"),
        mask_path=None,
        latents=np.zeros((3, 8), dtype=np.float32),
    )

    monkeypatch.setattr(
        inference,
        "_prepare_tiled_slides",
        lambda slide_records, preprocessing, output_dir, num_workers: ([slide_record], [tiling_result], Path(output_dir) / "process_list.csv"),
    )
    def fake_compute_embedded_slides(*args, **kwargs):
        on_embedded_slide = kwargs.get("on_embedded_slide")
        if on_embedded_slide is not None:
            on_embedded_slide(slide_record, tiling_result, embedded)
        return [embedded]

    monkeypatch.setattr(inference, "_compute_embedded_slides", fake_compute_embedded_slides)
    monkeypatch.setattr(inference, "_update_process_list_after_embedding", lambda *args, **kwargs: None)

    model = SimpleNamespace(
        name="prism",
        level="slide",
        _requested_device="cpu",
        _load_backend=lambda: SimpleNamespace(),
    )
    in_memory = inference.embed_slides(
        model,
        [slide_record],
        preprocessing=DEFAULT_PREPROCESSING,
        execution=ExecutionOptions(num_gpus=1),
    )
    assert in_memory == [embedded]

    persisted = inference.embed_slides(
        model,
        [slide_record],
        preprocessing=DEFAULT_PREPROCESSING,
        execution=ExecutionOptions(
            output_dir=tmp_path,
            output_format="npz",
            save_latents=True,
            save_tile_embeddings=True,
            num_gpus=1,
        ),
    )
    assert persisted == [embedded]
    assert (tmp_path / "tile_embeddings" / "slide-a.npz").is_file()
    assert (tmp_path / "slide_embeddings" / "slide-a.npz").is_file()


def test_direct_embed_slides_persists_completed_embeddings_before_later_slide_failure(monkeypatch, tmp_path: Path):
    pytest.importorskip("torch")
    import slide2vec.inference as inference

    slides = [make_slide("slide-a"), make_slide("slide-b")]
    tiling_results = [
        SimpleNamespace(
            x=np.array([0, 1]),
            y=np.array([2, 3]),
            tile_size_lv0=224,
            coordinates_npz_path=Path("/tmp/slide-a.coordinates.npz"),
            coordinates_meta_path=Path("/tmp/slide-a.coordinates.meta.json"),
        ),
        SimpleNamespace(
            x=np.array([4, 5]),
            y=np.array([6, 7]),
            tile_size_lv0=224,
            coordinates_npz_path=Path("/tmp/slide-b.coordinates.npz"),
            coordinates_meta_path=Path("/tmp/slide-b.coordinates.meta.json"),
        ),
    ]
    process_list_path = tmp_path / "process_list.csv"
    process_list_path.write_text(
        "sample_id,annotation,image_path,mask_path,spacing_at_level_0,tiling_status,num_tiles,coordinates_npz_path,coordinates_meta_path,error,traceback\n"
        "slide-a,tissue,/tmp/slide-a.svs,,,"  # spacing_at_level_0
        "success,2,/tmp/slide-a.coordinates.npz,/tmp/slide-a.coordinates.meta.json,,\n"
        "slide-b,tissue,/tmp/slide-b.svs,,,"
        "success,2,/tmp/slide-b.coordinates.npz,/tmp/slide-b.coordinates.meta.json,,\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        inference,
        "_prepare_tiled_slides",
        lambda slide_records, preprocessing, output_dir, num_workers: (slides, tiling_results, process_list_path),
    )

    def fake_compute_tile_embeddings(loaded, model, slide, tiling_result, *, preprocessing, execution, tile_indices=None):
        if slide.sample_id == "slide-b":
            raise RuntimeError("embedding boom")
        return np.zeros((2, 4), dtype=np.float32)

    monkeypatch.setattr(inference, "_compute_tile_embeddings_for_slide", fake_compute_tile_embeddings)
    monkeypatch.setattr(
        inference,
        "_aggregate_tile_embeddings_for_slide",
        lambda *args, **kwargs: (np.zeros((8,), dtype=np.float32), None),
    )

    model = SimpleNamespace(
        name="prism",
        level="slide",
        _requested_device="cpu",
        _load_backend=lambda: SimpleNamespace(),
    )

    with pytest.raises(RuntimeError, match="embedding boom"):
        inference.embed_slides(
            model,
            slides,
            preprocessing=DEFAULT_PREPROCESSING,
            execution=ExecutionOptions(output_dir=tmp_path, save_tile_embeddings=True, num_gpus=1),
        )

    assert (tmp_path / "tile_embeddings" / "slide-a.pt").is_file()
    assert (tmp_path / "tile_embeddings" / "slide-a.meta.json").is_file()
    assert (tmp_path / "slide_embeddings" / "slide-a.pt").is_file()
    assert (tmp_path / "slide_embeddings" / "slide-a.meta.json").is_file()
    assert not (tmp_path / "slide_embeddings" / "slide-b.pt").exists()

    process_df = pd.read_csv(process_list_path)
    recorded = process_df.set_index("sample_id")
    assert recorded.loc["slide-a", "feature_status"] == "success"
    assert recorded.loc["slide-a", "aggregation_status"] == "success"
    assert recorded.loc["slide-b", "feature_status"] == "tbp"
    assert recorded.loc["slide-b", "aggregation_status"] == "tbp"

def test_slide_level_pipeline_skips_tile_artifacts_when_save_tile_embeddings_is_false(monkeypatch, tmp_path: Path):
    import slide2vec.inference as inference

    slide_record = make_slide("slide-a")
    tiling_result = SimpleNamespace(
        x=np.array([0, 1]),
        y=np.array([2, 3]),
        tile_size_lv0=224,
        coordinates_npz_path=Path("/tmp/slide-a.coordinates.npz"),
        coordinates_meta_path=Path("/tmp/slide-a.coordinates.meta.json"),
    )
    embedded = EmbeddedSlide(
        sample_id="slide-a",
        tile_embeddings=np.zeros((2, 4), dtype=np.float32),
        slide_embedding=np.zeros((8,), dtype=np.float32),
        x=np.array([0, 1], dtype=np.int64),
        y=np.array([2, 3], dtype=np.int64),
        tile_size_lv0=224,
        image_path=Path("/tmp/slide-a.svs"),
        mask_path=None,
        latents=None,
    )

    monkeypatch.setattr(
        inference,
        "_prepare_tiled_slides",
        lambda slide_records, preprocessing, output_dir, num_workers: ([slide_record], [tiling_result], Path(output_dir) / "process_list.csv"),
    )
    def fake_compute_embedded_slides(*args, **kwargs):
        on_embedded_slide = kwargs.get("on_embedded_slide")
        if on_embedded_slide is not None:
            on_embedded_slide(slide_record, tiling_result, embedded)
        return [embedded]

    monkeypatch.setattr(inference, "_compute_embedded_slides", fake_compute_embedded_slides)
    monkeypatch.setattr(inference, "_update_process_list_after_embedding", lambda *args, **kwargs: None)

    result = inference.run_pipeline(
        Model.from_preset("prism"),
        slides=[slide_record],
        preprocessing=DEFAULT_PREPROCESSING,
        execution=ExecutionOptions(
            output_dir=tmp_path,
            output_format="npz",
            save_tile_embeddings=False,
            num_gpus=1,
        ),
    )

    assert result.tile_artifacts == []
    assert len(result.slide_artifacts) == 1
    assert not (tmp_path / "tile_embeddings" / "slide-a.npz").exists()
    assert (tmp_path / "slide_embeddings" / "slide-a.npz").is_file()

def test_direct_embed_slides_uses_tile_sharding_for_single_slide(monkeypatch, tmp_path: Path):
    import slide2vec.inference as inference

    slide_record = make_slide("slide-a")
    tiling_result = SimpleNamespace(
        x=np.array([0, 1]),
        y=np.array([2, 3]),
        tile_size_lv0=224,
    )
    embedded = EmbeddedSlide(
        sample_id="slide-a",
        tile_embeddings=np.zeros((2, 4), dtype=np.float32),
        slide_embedding=None,
        x=np.array([0, 1], dtype=np.int64),
        y=np.array([2, 3], dtype=np.int64),
        tile_size_lv0=224,
        image_path=Path("/tmp/slide-a.svs"),
        mask_path=None,
    )
    captured = {}

    monkeypatch.setattr(
        inference,
        "_prepare_tiled_slides",
        lambda slide_records, preprocessing, output_dir, num_workers: ([slide_record], [tiling_result], Path(output_dir) / "process_list.csv"),
    )
    monkeypatch.setattr(inference, "_validate_multi_gpu_execution", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        inference,
        "_embed_single_slide_distributed",
        lambda *args, **kwargs: captured.update({"single": kwargs}) or embedded,
    )
    monkeypatch.setattr(
        inference,
        "_embed_multi_slides_distributed",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("multi-slide path should not be used")),
    )

    result = inference.embed_slides(
        Model.from_preset("virchow2"),
        [slide_record],
        preprocessing=DEFAULT_PREPROCESSING,
        execution=ExecutionOptions(output_dir=tmp_path, output_format="npz", num_gpus=2),
    )

    assert result == [embedded]
    assert captured["single"]["slide"] == slide_record

def test_direct_embed_slides_uses_balanced_slide_sharding_for_multiple_slides(monkeypatch, tmp_path: Path):
    import slide2vec.inference as inference

    slides = [
        make_slide("slide-a"),
        make_slide("slide-b"),
    ]
    tiling_results = [
        SimpleNamespace(x=np.array([0, 1, 2]), y=np.array([0, 1, 2]), tile_size_lv0=224),
        SimpleNamespace(x=np.array([0]), y=np.array([0]), tile_size_lv0=224),
    ]
    expected = [
        EmbeddedSlide(
            sample_id="slide-a",
            tile_embeddings=np.zeros((3, 2), dtype=np.float32),
            slide_embedding=None,
            x=np.array([0, 1, 2], dtype=np.int64),
            y=np.array([0, 1, 2], dtype=np.int64),
            tile_size_lv0=224,
            image_path=Path("/tmp/slide-a.svs"),
            mask_path=None,
        ),
        EmbeddedSlide(
            sample_id="slide-b",
            tile_embeddings=np.zeros((1, 2), dtype=np.float32),
            slide_embedding=None,
            x=np.array([0], dtype=np.int64),
            y=np.array([0], dtype=np.int64),
            tile_size_lv0=224,
            image_path=Path("/tmp/slide-b.svs"),
            mask_path=None,
        ),
    ]
    captured = {}

    monkeypatch.setattr(
        inference,
        "_prepare_tiled_slides",
        lambda slide_records, preprocessing, output_dir, num_workers: (slides, tiling_results, Path(output_dir) / "process_list.csv"),
    )
    monkeypatch.setattr(inference, "_validate_multi_gpu_execution", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        inference,
        "_embed_multi_slides_distributed",
        lambda *args, **kwargs: captured.update({"multi": kwargs}) or expected,
    )
    monkeypatch.setattr(
        inference,
        "_embed_single_slide_distributed",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("single-slide path should not be used")),
    )

    result = inference.embed_slides(
        Model.from_preset("virchow2"),
        slides,
        preprocessing=DEFAULT_PREPROCESSING,
        execution=ExecutionOptions(output_dir=tmp_path, output_format="npz", num_gpus=2),
    )

    assert result == expected
    assert captured["multi"]["slide_records"] == slides

def test_pipeline_worker_assigns_slides_by_tile_count():
    from slide2vec.distributed import pipeline_worker

    slides = [
        make_slide("slide-a"),
        make_slide("slide-b"),
        make_slide("slide-c"),
        make_slide("slide-d"),
    ]
    tiling_results = [
        SimpleNamespace(x=np.arange(9), y=np.arange(9), tile_size_lv0=224),
        SimpleNamespace(x=np.arange(8), y=np.arange(8), tile_size_lv0=224),
        SimpleNamespace(x=np.arange(7), y=np.arange(7), tile_size_lv0=224),
        SimpleNamespace(x=np.arange(6), y=np.arange(6), tile_size_lv0=224),
    ]

    assignments = pipeline_worker.assign_slides_to_ranks(slides, tiling_results, num_gpus=2)

    assert assignments == {
        0: ["slide-a", "slide-d"],
        1: ["slide-b", "slide-c"],
    }

def test_assign_slides_to_ranks_balances_by_tile_count():
    from slide2vec.runtime.distributed import assign_slides_to_ranks

    slides = [
        make_slide("slide-a"),
        make_slide("slide-b"),
        make_slide("slide-c"),
        make_slide("slide-d"),
    ]
    tiling_results = [
        SimpleNamespace(x=np.arange(9), y=np.arange(9), tile_size_lv0=224),
        SimpleNamespace(x=np.arange(8), y=np.arange(8), tile_size_lv0=224),
        SimpleNamespace(x=np.arange(7), y=np.arange(7), tile_size_lv0=224),
        SimpleNamespace(x=np.arange(6), y=np.arange(6), tile_size_lv0=224),
    ]

    assignments = assign_slides_to_ranks(slides, tiling_results, num_gpus=2)

    assert assignments == {
        0: ["slide-a", "slide-d"],
        1: ["slide-b", "slide-c"],
    }


def test_assign_slides_to_ranks_tiebreaks_by_rank_deterministically():
    from slide2vec.runtime.distributed import assign_slides_to_ranks

    slides = [
        make_slide("slide-a"),
        make_slide("slide-b"),
        make_slide("slide-c"),
        make_slide("slide-d"),
        make_slide("slide-e"),
    ]
    tiling_results = [
        SimpleNamespace(x=np.arange(10), y=np.arange(10), tile_size_lv0=224),
        SimpleNamespace(x=np.arange(10), y=np.arange(10), tile_size_lv0=224),
        SimpleNamespace(x=np.arange(10), y=np.arange(10), tile_size_lv0=224),
        SimpleNamespace(x=np.arange(1), y=np.arange(1), tile_size_lv0=224),
        SimpleNamespace(x=np.arange(1), y=np.arange(1), tile_size_lv0=224),
    ]

    assignments = assign_slides_to_ranks(slides, tiling_results, num_gpus=3)

    assert assignments == {
        0: ["slide-a", "slide-d"],
        1: ["slide-b", "slide-e"],
        2: ["slide-c"],
    }


def test_merge_tile_embedding_shards_restores_original_tile_order():
    from slide2vec.runtime.distributed import merge_tile_embedding_shards

    merged = merge_tile_embedding_shards(
        [
            {
                "tile_index": np.array([2, 0], dtype=np.int64),
                "tile_embeddings": np.array([[20.0, 21.0], [0.0, 1.0]], dtype=np.float32),
            },
            {
                "tile_index": np.array([3, 1], dtype=np.int64),
                "tile_embeddings": np.array([[30.0, 31.0], [10.0, 11.0]], dtype=np.float32),
            },
        ]
    )

    np.testing.assert_array_equal(
        merged,
        np.array(
            [
                [0.0, 1.0],
                [10.0, 11.0],
                [20.0, 21.0],
                [30.0, 31.0],
            ],
            dtype=np.float32,
        ),
    )

def test_run_forward_pass_handles_empty_dataloader():
    import slide2vec.inference as inference
    torch = pytest.importorskip("torch")
    from contextlib import nullcontext

    class DummyModel:
        def encode_tiles(self, image):
            return torch.zeros((image.shape[0], 5), dtype=torch.float32)

    dataloader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.empty((0,), dtype=torch.int64),
            torch.empty((0, 3, 2, 2), dtype=torch.float32),
        ),
        batch_size=2,
    )
    loaded = inference.LoadedModel(
        name="virchow2",
        level="tile",
        model=DummyModel(),
        transforms=None,
        feature_dim=5,
        device=torch.device("cpu"),
    )

    result = inference._run_forward_pass(dataloader, loaded, nullcontext())

    assert result.shape == (0, 5)
    assert result.dtype == torch.float32


def test_build_batch_preprocessor_falls_back_for_unsupported_transform_stack(caplog):
    import slide2vec.inference as inference
    torch = pytest.importorskip("torch")

    class UnsupportedTransform:
        pass

    loaded = inference.LoadedModel(
        name="h0-mini",
        level="tile",
        model=SimpleNamespace(),
        transforms=SimpleNamespace(transforms=[UnsupportedTransform()]),
        feature_dim=3,
        device=torch.device("cpu"),
    )
    tiling_result = SimpleNamespace(requested_tile_size_px=224)

    with caplog.at_level("WARNING", logger="slide2vec.inference"):
        preprocess = inference._build_batch_preprocessor(
            loaded,
            tiling_result,
        )

    assert preprocess is None
    assert "falling back to per-item preprocessing" in caplog.text


def test_run_forward_pass_applies_itemwise_transforms_when_batch_preprocessing_is_unavailable():
    import slide2vec.inference as inference
    torch = pytest.importorskip("torch")
    from contextlib import nullcontext

    class UnsupportedTransform:
        pass

    class ItemwiseTransform:
        def __init__(self):
            self.transforms = [UnsupportedTransform()]

        def __call__(self, image):
            return image.float().div(255.0)

    class DummyLoader:
        def __iter__(self):
            yield (
                torch.tensor([0, 1], dtype=torch.long),
                torch.full((2, 3, 4, 4), 255, dtype=torch.uint8),
            )

        def __len__(self):
            return 1

    class DummyModel:
        def encode_tiles(self, image):
            assert image.dtype == torch.float32
            assert torch.allclose(image, torch.ones_like(image))
            return torch.ones((image.shape[0], 3), dtype=torch.float32, device=image.device)

    loaded = inference.LoadedModel(
        name="h0-mini",
        level="tile",
        model=DummyModel(),
        transforms=ItemwiseTransform(),
        feature_dim=3,
        device=torch.device("cpu"),
    )

    result = inference._run_forward_pass(
        DummyLoader(),
        loaded,
        nullcontext(),
        batch_preprocessor=None,
        sample_id="slide-a",
        total_items=2,
    )

    assert result.shape == (2, 3)
    assert torch.allclose(result, torch.ones((2, 3), dtype=torch.float32))


def test_serialize_execution_preserves_loader_optimization_fields():
    import slide2vec.inference as inference
    from slide2vec.runtime.serialization import deserialize_execution

    execution = ExecutionOptions(
        output_dir=Path("/tmp/output"),
        batch_size=64,
        num_workers=8,
        num_gpus=2,
        precision="bf16",
        prefetch_factor=7,
        persistent_workers=False,
        save_tile_embeddings=True,
        save_latents=True,
    )

    payload = inference._serialize_execution(execution)
    restored = deserialize_execution(payload)

    assert payload["prefetch_factor"] == 7
    assert payload["persistent_workers"] is False
    assert payload["precision"] == "bf16"
    assert restored.prefetch_factor == 7
    assert restored.persistent_workers is False
    assert restored.precision == "bf16"


def test_deserialize_execution_defaults_num_workers_to_auto():
    from slide2vec.runtime.serialization import deserialize_execution

    restored = deserialize_execution({"batch_size": 4, "num_gpus": 1})

    assert restored.num_workers is None


def test_deserialize_execution_preserves_auto_num_workers():
    from slide2vec.runtime.serialization import deserialize_execution

    restored = deserialize_execution({"batch_size": 4, "num_workers": None, "num_gpus": 1})

    assert restored.num_workers is None


def test_embedding_dataloader_kwargs_resolve_auto_mode_to_cpu_budget(monkeypatch):
    import slide2vec.api as api
    import slide2vec.inference as inference
    torch = pytest.importorskip("torch")

    monkeypatch.setattr(api, "cpu_worker_limit", lambda: 24)

    loaded = inference.LoadedModel(
        name="test",
        level="tile",
        model=object(),
        transforms=object(),
        feature_dim=3,
        device=torch.device("cpu"),
    )

    kwargs = inference._embedding_dataloader_kwargs(
        loaded,
        ExecutionOptions(num_workers=None, num_gpus=1),
    )

    assert kwargs["num_workers"] == 24
    assert kwargs["persistent_workers"] is True
    assert kwargs["prefetch_factor"] == 4


def test_compute_tile_embeddings_for_slide_uses_cpu_budget_for_auto_workers_on_non_cucim_on_the_fly(monkeypatch):
    import slide2vec.api as api
    import slide2vec.inference as inference
    torch = pytest.importorskip("torch")

    captured = {}

    class DummyLoader:
        def __init__(self, dataset, **kwargs):
            captured["kwargs"] = kwargs

        def __iter__(self):
            yield (
                torch.tensor([0, 1], dtype=torch.long),
                torch.zeros((2, 3, 4, 4), dtype=torch.uint8),
                {"worker_batch_ms": 0.0, "reader_open_ms": 0.0, "reader_read_ms": 0.0},
            )

        def __len__(self):
            return 1

    class DummyEncoder:
        pretrained_cfg = {}

    class DummyModel:
        encoder = DummyEncoder()

        def encode_tiles(self, image):
            return torch.ones((image.shape[0], 3), dtype=torch.float32, device=image.device)

    class DummyCollator:
        ordered_indices = None

        def __init__(self, **kwargs):
            captured["wsd_collator_kwargs"] = kwargs

        def __call__(self, batch_indices):
            tile_indices = torch.as_tensor(batch_indices, dtype=torch.long)
            batch = torch.zeros((len(batch_indices), 3, 4, 4), dtype=torch.uint8)
            return tile_indices, batch, {"worker_batch_ms": 0.0, "reader_open_ms": 0.0, "reader_read_ms": 0.0}

    monkeypatch.setattr(inference, "OnTheFlyBatchTileCollator", DummyCollator)
    monkeypatch.setattr(torch.utils.data, "DataLoader", DummyLoader)
    monkeypatch.setattr(inference, "_build_batch_preprocessor", lambda *args, **kwargs: lambda batch: batch.float())
    monkeypatch.setattr(api, "cpu_worker_limit", lambda: 24)

    loaded = inference.LoadedModel(
        name="prov-gigapath",
        level="tile",
        model=DummyModel(),
        transforms=object(),
        feature_dim=3,
        device=torch.device("cpu"),
    )

    result = inference._compute_tile_embeddings_for_slide(
        loaded,
        SimpleNamespace(level="tile"),
        make_slide("slide-a"),
        SimpleNamespace(
            x=np.array([0, 10]),
            y=np.array([5, 15]),
            backend="asap",
            requested_spacing_um=0.5,
            requested_tile_size_px=4,
            read_spacing_um=0.5,
            read_tile_size_px=4,
            tile_size_lv0=224,
        ),
        preprocessing=replace(DEFAULT_PREPROCESSING, on_the_fly=True, backend="auto", num_cucim_workers=4),
        execution=ExecutionOptions(batch_size=2, num_workers=None, num_gpus=1),
    )

    assert result.shape == (2, 3)
    assert captured["kwargs"]["num_workers"] == 24
    assert captured["kwargs"]["persistent_workers"] is True
    assert captured["kwargs"]["prefetch_factor"] == 4
    assert captured["wsd_collator_kwargs"]["backend"] == "asap"


def test_compute_tile_embeddings_for_slide_uses_batched_loader_knobs(monkeypatch):
    import slide2vec.inference as inference
    torch = pytest.importorskip("torch")

    captured = {}

    class DummyLoader:
        def __init__(self, dataset, **kwargs):
            captured["dataset"] = dataset
            captured["kwargs"] = kwargs

        def __iter__(self):
            yield (
                torch.tensor([0, 1], dtype=torch.long),
                torch.zeros((2, 3, 4, 4), dtype=torch.uint8),
            )

        def __len__(self):
            return 1

    class DummyEncoder:
        pretrained_cfg = {}

    class DummyModel:
        encoder = DummyEncoder()

        def encode_tiles(self, image):
            return torch.ones((image.shape[0], 3), dtype=torch.float32, device=image.device)

    monkeypatch.setattr(inference, "BatchTileCollator", lambda **kwargs: ("collator", kwargs))
    monkeypatch.setattr(inference, "TileIndexDataset", lambda tile_indices: list(tile_indices))
    monkeypatch.setattr(torch.utils.data, "DataLoader", DummyLoader)
    monkeypatch.setattr(inference, "_build_batch_preprocessor", lambda *args, **kwargs: lambda batch: batch.float())

    loaded = inference.LoadedModel(
        name="prov-gigapath",
        level="tile",
        model=DummyModel(),
        transforms=object(),
        feature_dim=3,
        device=torch.device("cpu"),
    )
    slide = make_slide("slide-a")
    tiling_result = SimpleNamespace(
        x=np.array([0, 10]),
        y=np.array([5, 15]),
        requested_spacing_um=0.5,
        requested_tile_size_px=4,
        read_spacing_um=0.5,
        read_tile_size_px=4,
        tile_size_lv0=224,
        tiles_tar_path=Path("/tmp/slide-a.tiles.tar"),
    )
    execution = ExecutionOptions(
        batch_size=2,
        num_workers=3,
        num_gpus=1,
        prefetch_factor=9,
        persistent_workers=True,
    )

    result = inference._compute_tile_embeddings_for_slide(
        loaded,
        SimpleNamespace(level="tile"),
        slide,
        tiling_result,
        preprocessing=replace(DEFAULT_PREPROCESSING, on_the_fly=False),
        execution=execution,
    )

    assert result.shape == (2, 3)
    assert captured["kwargs"]["num_workers"] == 3
    assert captured["kwargs"]["persistent_workers"] is True
    assert captured["kwargs"]["prefetch_factor"] == 9
    assert captured["kwargs"]["collate_fn"] == (
        "collator",
        {
            "tar_path": Path("/tmp/slide-a.tiles.tar"),
            "tiling_result": tiling_result,
        },
    )


def test_compute_tile_embeddings_for_slide_prefers_explicit_tile_store_root(monkeypatch):
    import slide2vec.inference as inference
    torch = pytest.importorskip("torch")

    captured = {}

    class DummyLoader:
        def __init__(self, dataset, **kwargs):
            captured["dataset"] = dataset
            captured["kwargs"] = kwargs

        def __iter__(self):
            yield (
                torch.tensor([0], dtype=torch.long),
                torch.zeros((1, 3, 4, 4), dtype=torch.uint8),
            )

        def __len__(self):
            return 1

    class DummyEncoder:
        pretrained_cfg = {}

    class DummyModel:
        encoder = DummyEncoder()

        def encode_tiles(self, image):
            return torch.ones((image.shape[0], 3), dtype=torch.float32, device=image.device)

    monkeypatch.setattr(inference, "BatchTileCollator", lambda **kwargs: ("collator", kwargs))
    monkeypatch.setattr(inference, "TileIndexDataset", lambda tile_indices: list(tile_indices))
    monkeypatch.setattr(torch.utils.data, "DataLoader", DummyLoader)
    monkeypatch.setattr(inference, "_build_batch_preprocessor", lambda *args, **kwargs: lambda batch: batch.float())

    loaded = inference.LoadedModel(
        name="prov-gigapath",
        level="tile",
        model=DummyModel(),
        transforms=object(),
        feature_dim=3,
        device=torch.device("cpu"),
    )
    slide = make_slide("slide-a")
    tiling_result = SimpleNamespace(
        x=np.array([0]),
        y=np.array([5]),
        requested_spacing_um=0.5,
        requested_tile_size_px=4,
        read_spacing_um=0.5,
        read_tile_size_px=4,
        tile_size_lv0=224,
        tiles_tar_path=Path("/tmp/current-run.tiles.tar"),
    )

    result = inference._compute_tile_embeddings_for_slide(
        loaded,
        SimpleNamespace(level="tile"),
        slide,
        tiling_result,
        preprocessing=replace(DEFAULT_PREPROCESSING, read_tiles_from=Path("/tmp/external-tiles")),
        execution=ExecutionOptions(batch_size=1, num_workers=0, num_gpus=1),
    )

    assert result.shape == (1, 3)
    assert captured["kwargs"]["collate_fn"] == (
        "collator",
        {
            "tar_path": Path("/tmp/external-tiles/slide-a.tiles.tar"),
            "tiling_result": tiling_result,
        },
    )


def test_resolve_on_the_fly_num_workers_caps_to_slurm_allocation(monkeypatch):
    import slide2vec.inference as inference

    monkeypatch.setattr(inference.os, "cpu_count", lambda: 96)
    monkeypatch.setenv("SLURM_JOB_CPUS_PER_NODE", "32")
    monkeypatch.delenv("SLURM_CPUS_PER_TASK", raising=False)
    monkeypatch.delenv("SLURM_CPUS_ON_NODE", raising=False)

    workers, details = inference._resolve_on_the_fly_num_workers(4)

    assert workers == 8
    assert "cpu_count=96" in details
    assert "slurm_cpu_limit=32" in details
    assert "num_cucim_workers=4" in details


def test_compute_tile_embeddings_for_slide_caps_on_the_fly_workers_to_slurm(monkeypatch, caplog):
    import slide2vec.inference as inference
    torch = pytest.importorskip("torch")

    captured = {}

    class DummyLoader:
        def __init__(self, dataset, **kwargs):
            captured["dataset"] = dataset
            captured["kwargs"] = kwargs

        def __iter__(self):
            yield (
                torch.tensor([0, 1], dtype=torch.long),
                torch.zeros((2, 3, 4, 4), dtype=torch.uint8),
                {"worker_batch_ms": 0.0, "reader_open_ms": 0.0, "reader_read_ms": 0.0},
            )

        def __len__(self):
            return 1

    class DummyEncoder:
        pretrained_cfg = {}

    class DummyModel:
        encoder = DummyEncoder()

        def encode_tiles(self, image):
            return torch.ones((image.shape[0], 3), dtype=torch.float32, device=image.device)

    class DummyCollator:
        ordered_indices = None

        def __init__(self, **kwargs):
            captured["collator_kwargs"] = kwargs

        def __call__(self, batch_indices):
            tile_indices = torch.as_tensor(batch_indices, dtype=torch.long)
            batch = torch.zeros((len(batch_indices), 3, 4, 4), dtype=torch.uint8)
            return tile_indices, batch, {"worker_batch_ms": 0.0, "reader_open_ms": 0.0, "reader_read_ms": 0.0}

    monkeypatch.setattr(inference, "OnTheFlyBatchTileCollator", DummyCollator)
    monkeypatch.setattr(torch.utils.data, "DataLoader", DummyLoader)
    monkeypatch.setattr(inference, "_build_batch_preprocessor", lambda *args, **kwargs: lambda batch: batch.float())
    monkeypatch.setattr(inference.os, "cpu_count", lambda: 96)
    monkeypatch.setenv("SLURM_JOB_CPUS_PER_NODE", "32")
    monkeypatch.delenv("SLURM_CPUS_PER_TASK", raising=False)
    monkeypatch.delenv("SLURM_CPUS_ON_NODE", raising=False)

    loaded = inference.LoadedModel(
        name="prov-gigapath",
        level="tile",
        model=DummyModel(),
        transforms=object(),
        feature_dim=3,
        device=torch.device("cpu"),
    )
    slide = make_slide("slide-a")
    tiling_result = SimpleNamespace(
        x=np.array([0, 10]),
        y=np.array([5, 15]),
        requested_spacing_um=0.5,
        requested_tile_size_px=4,
        read_spacing_um=0.5,
        read_tile_size_px=4,
        tile_size_lv0=224,
    )
    execution = ExecutionOptions(
        batch_size=2,
        num_workers=99,
        num_gpus=1,
        prefetch_factor=9,
        persistent_workers=True,
    )

    with caplog.at_level("INFO"):
        result = inference._compute_tile_embeddings_for_slide(
            loaded,
            SimpleNamespace(level="tile"),
            slide,
            tiling_result,
            preprocessing=replace(DEFAULT_PREPROCESSING, on_the_fly=True, backend="cucim", num_cucim_workers=4),
            execution=execution,
        )

    assert result.shape == (2, 3)
    assert captured["kwargs"]["num_workers"] == 8
    assert captured["kwargs"]["persistent_workers"] is True
    assert captured["kwargs"]["prefetch_factor"] == 9
    assert "on-the-fly mode: setting DataLoader num_workers=8" not in caplog.text


def test_run_pipeline_logs_on_the_fly_worker_override_once(monkeypatch, tmp_path: Path, caplog):
    import slide2vec.inference as inference

    slides = [
        make_slide("slide-a"),
        make_slide("slide-b"),
    ]
    tiling_results = [
        SimpleNamespace(
            x=np.array([0, 10]),
            y=np.array([5, 15]),
            tile_size_lv0=224,
            backend="cucim",
        ),
        SimpleNamespace(
            x=np.array([20, 30]),
            y=np.array([25, 35]),
            tile_size_lv0=224,
            backend="cucim",
        ),
    ]

    monkeypatch.setattr(
        inference,
        "_prepare_tiled_slides",
        lambda *args, **kwargs: (slides, tiling_results, tmp_path / "process_list.csv"),
    )
    monkeypatch.setattr(inference, "_emit_tiling_summary", lambda *args, **kwargs: None)
    monkeypatch.setattr(inference, "_write_zero_tile_embedding_sidecars", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        inference,
        "_compute_embedded_slides",
        lambda *args, **kwargs: [SimpleNamespace(slide_embedding=None) for _ in slides],
    )
    monkeypatch.setattr(
        inference,
        "_collect_pipeline_artifacts",
        lambda *args, **kwargs: ([], [], []),
    )
    monkeypatch.setattr(inference, "_update_process_list_after_embedding", lambda *args, **kwargs: None)

    model = SimpleNamespace(name="prov-gigapath", level="tile")
    execution = ExecutionOptions(output_dir=tmp_path, num_gpus=1)

    with caplog.at_level("INFO"):
        inference.run_pipeline(
            model,
            slides=slides,
            preprocessing=replace(DEFAULT_PREPROCESSING, on_the_fly=True, backend="cucim", num_cucim_workers=4),
            execution=execution,
        )

    assert caplog.text.count("on-the-fly mode: setting DataLoader num_workers=") == 1


def test_compute_tile_embeddings_for_slide_filters_on_the_fly_cucim_stderr_without_changing_workers(monkeypatch):
    import slide2vec.inference as inference
    torch = pytest.importorskip("torch")

    captured = {}

    class DummyLoader:
        def __init__(self, dataset, **kwargs):
            captured["kwargs"] = kwargs

        def __iter__(self):
            yield (
                torch.tensor([0, 1], dtype=torch.long),
                torch.zeros((2, 3, 4, 4), dtype=torch.uint8),
                {"worker_batch_ms": 0.0, "reader_open_ms": 0.0, "reader_read_ms": 0.0},
            )

        def __len__(self):
            return 1

    class DummyEncoder:
        pretrained_cfg = {}

    class DummyModel:
        encoder = DummyEncoder()

        def encode_tiles(self, image):
            return torch.ones((image.shape[0], 3), dtype=torch.float32)

    class DummyCollator:
        ordered_indices = None

        def __init__(self, **kwargs):
            captured["collator_kwargs"] = kwargs

        def __call__(self, batch_indices):
            tile_indices = torch.as_tensor(batch_indices, dtype=torch.long)
            batch = torch.zeros((len(batch_indices), 3, 4, 4), dtype=torch.uint8)
            return tile_indices, batch, {"worker_batch_ms": 0.0, "reader_open_ms": 0.0, "reader_read_ms": 0.0}

    monkeypatch.setattr(inference, "OnTheFlyBatchTileCollator", DummyCollator)
    monkeypatch.setattr(torch.utils.data, "DataLoader", DummyLoader)
    monkeypatch.setattr(inference, "_build_batch_preprocessor", lambda *args, **kwargs: lambda batch: batch.float())
    monkeypatch.setattr(inference, "cpu_worker_limit", lambda: 32)

    def _fake_run_with_filtered_stderr(func, **kwargs):
        del kwargs
        captured["filtered_calls"] = captured.get("filtered_calls", 0) + 1
        return func()

    monkeypatch.setattr(inference, "run_with_filtered_stderr", _fake_run_with_filtered_stderr)

    loaded = inference.LoadedModel(
        name="prov-gigapath",
        level="tile",
        model=DummyModel(),
        transforms=object(),
        feature_dim=3,
        device=torch.device("cuda"),
    )
    slide = make_slide("slide-a")
    tiling_result = SimpleNamespace(
        x=np.array([0, 10]),
        y=np.array([5, 15]),
        requested_spacing_um=0.5,
        requested_tile_size_px=4,
        read_spacing_um=0.5,
        read_tile_size_px=4,
        tile_size_lv0=224,
    )
    execution = ExecutionOptions(
        batch_size=2,
        num_workers=99,
        num_gpus=1,
        prefetch_factor=9,
        persistent_workers=True,
    )

    result = inference._compute_tile_embeddings_for_slide(
        loaded,
        SimpleNamespace(level="tile"),
        slide,
        tiling_result,
        preprocessing=replace(DEFAULT_PREPROCESSING, on_the_fly=True, backend="cucim", num_cucim_workers=4),
        execution=execution,
    )

    assert result.shape == (2, 3)
    assert captured["kwargs"]["num_workers"] == 8
    assert captured["kwargs"]["persistent_workers"] is True
    assert captured["kwargs"]["prefetch_factor"] == 9
    assert captured["filtered_calls"] == 1


def test_compute_tile_embeddings_for_slide_uses_resolved_cucim_backend_when_auto(monkeypatch):
    import slide2vec.inference as inference
    torch = pytest.importorskip("torch")

    captured = {}

    class DummyLoader:
        def __init__(self, dataset, **kwargs):
            captured["kwargs"] = kwargs

        def __iter__(self):
            yield (
                torch.tensor([0, 1], dtype=torch.long),
                torch.zeros((2, 3, 4, 4), dtype=torch.uint8),
                {"worker_batch_ms": 0.0, "reader_open_ms": 0.0, "reader_read_ms": 0.0},
            )

        def __len__(self):
            return 1

    class DummyEncoder:
        pretrained_cfg = {}

    class DummyModel:
        encoder = DummyEncoder()

        def encode_tiles(self, image):
            return torch.ones((image.shape[0], 3), dtype=torch.float32, device=image.device)

    class DummyCucimCollator:
        ordered_indices = None

        def __init__(self, **kwargs):
            captured["cucim_collator_kwargs"] = kwargs

        def __call__(self, batch_indices):
            tile_indices = torch.as_tensor(batch_indices, dtype=torch.long)
            batch = torch.zeros((len(batch_indices), 3, 4, 4), dtype=torch.uint8)
            return tile_indices, batch, {"worker_batch_ms": 0.0, "reader_open_ms": 0.0, "reader_read_ms": 0.0}

    monkeypatch.setattr(inference, "BatchTileCollator", lambda **kwargs: ("collator", kwargs))
    monkeypatch.setattr(inference, "TileIndexDataset", lambda tile_indices: list(tile_indices))
    monkeypatch.setattr(inference, "OnTheFlyBatchTileCollator", DummyCucimCollator)
    monkeypatch.setattr(torch.utils.data, "DataLoader", DummyLoader)
    monkeypatch.setattr(inference, "_build_batch_preprocessor", lambda *args, **kwargs: lambda batch: batch.float())
    monkeypatch.setattr(inference.os, "cpu_count", lambda: 32)

    loaded = inference.LoadedModel(
        name="prov-gigapath",
        level="tile",
        model=DummyModel(),
        transforms=object(),
        feature_dim=3,
        device=torch.device("cpu"),
    )

    result = inference._compute_tile_embeddings_for_slide(
        loaded,
        SimpleNamespace(level="tile"),
        make_slide("slide-a"),
        SimpleNamespace(
            x=np.array([0, 10]),
            y=np.array([5, 15]),
            backend="cucim",
            requested_spacing_um=0.5,
            requested_tile_size_px=4,
            read_spacing_um=0.5,
            read_tile_size_px=4,
            tile_size_lv0=224,
        ),
        preprocessing=replace(DEFAULT_PREPROCESSING, on_the_fly=True, backend="auto", num_cucim_workers=4),
        execution=ExecutionOptions(batch_size=2, num_workers=8, num_gpus=1),
    )

    assert result.shape == (2, 3)
    assert captured["cucim_collator_kwargs"]["num_cucim_workers"] == 4
    assert captured["cucim_collator_kwargs"]["gpu_decode"] is False


def test_compute_tile_embeddings_for_slide_uses_resolved_wsd_backend_when_auto(monkeypatch):
    import slide2vec.inference as inference
    torch = pytest.importorskip("torch")

    captured = {}

    class DummyLoader:
        def __init__(self, dataset, **kwargs):
            captured["kwargs"] = kwargs

        def __iter__(self):
            yield (
                torch.tensor([0, 1], dtype=torch.long),
                torch.zeros((2, 3, 4, 4), dtype=torch.uint8),
                {"worker_batch_ms": 0.0, "reader_open_ms": 0.0, "reader_read_ms": 0.0},
            )

        def __len__(self):
            return 1

    class DummyEncoder:
        pretrained_cfg = {}

    class DummyModel:
        encoder = DummyEncoder()

        def encode_tiles(self, image):
            return torch.ones((image.shape[0], 3), dtype=torch.float32, device=image.device)

    class DummyCollator:
        ordered_indices = None

        def __init__(self, **kwargs):
            captured["wsd_collator_kwargs"] = kwargs

        def __call__(self, batch_indices):
            tile_indices = torch.as_tensor(batch_indices, dtype=torch.long)
            batch = torch.zeros((len(batch_indices), 3, 4, 4), dtype=torch.uint8)
            return tile_indices, batch, {"worker_batch_ms": 0.0, "reader_open_ms": 0.0, "reader_read_ms": 0.0}

    monkeypatch.setattr(inference, "OnTheFlyBatchTileCollator", DummyCollator)
    monkeypatch.setattr(torch.utils.data, "DataLoader", DummyLoader)
    monkeypatch.setattr(inference, "_build_batch_preprocessor", lambda *args, **kwargs: lambda batch: batch.float())
    monkeypatch.setattr(inference.os, "cpu_count", lambda: 32)

    loaded = inference.LoadedModel(
        name="prov-gigapath",
        level="tile",
        model=DummyModel(),
        transforms=object(),
        feature_dim=3,
        device=torch.device("cpu"),
    )

    result = inference._compute_tile_embeddings_for_slide(
        loaded,
        SimpleNamespace(level="tile"),
        make_slide("slide-a"),
        SimpleNamespace(
            x=np.array([0, 10]),
            y=np.array([5, 15]),
            backend="asap",
            requested_spacing_um=0.5,
            requested_tile_size_px=4,
            read_spacing_um=0.5,
            read_tile_size_px=4,
            tile_size_lv0=224,
        ),
        preprocessing=replace(DEFAULT_PREPROCESSING, on_the_fly=True, backend="auto", num_cucim_workers=4),
        execution=ExecutionOptions(batch_size=2, num_workers=8, num_gpus=1),
    )

    assert result.shape == (2, 3)
    assert captured["kwargs"]["num_workers"] == 8
    assert captured["kwargs"]["persistent_workers"] is True
    assert captured["kwargs"]["prefetch_factor"] == 4
    assert captured["wsd_collator_kwargs"]["backend"] == "asap"


def test_persist_embedded_slide_records_resolved_backend_when_auto(monkeypatch, tmp_path: Path):
    import slide2vec.inference as inference

    embedded = EmbeddedSlide(
        sample_id="slide-a",
        tile_embeddings=np.zeros((2, 4), dtype=np.float32),
        slide_embedding=None,
        x=np.array([0, 1], dtype=np.int64),
        y=np.array([2, 3], dtype=np.int64),
        tile_size_lv0=224,
        image_path=Path("/tmp/slide-a.svs"),
        mask_path=None,
    )
    captured = {}

    monkeypatch.setattr(
        inference.runtime_embedding,
        "write_tile_embedding_artifact",
        lambda sample_id, features, *, execution, metadata: captured.setdefault("metadata", metadata) or SimpleNamespace(),
    )

    inference._persist_embedded_slide(
        SimpleNamespace(name="prov-gigapath", level="tile"),
        embedded,
        SimpleNamespace(
            backend="cucim",
            coordinates_npz_path=Path("/tmp/slide-a.coordinates.npz"),
            coordinates_meta_path=Path("/tmp/slide-a.coordinates.meta.json"),
            tiles_tar_path=Path("/tmp/slide-a.tiles.tar"),
        ),
        preprocessing=replace(DEFAULT_PREPROCESSING, backend="auto"),
        execution=ExecutionOptions(output_dir=tmp_path),
    )

    assert captured["metadata"]["backend"] == "cucim"


def test_compute_tile_embeddings_for_slide_requires_current_run_tile_store_without_explicit_override():
    import slide2vec.inference as inference
    torch = pytest.importorskip("torch")

    loaded = inference.LoadedModel(
        name="prov-gigapath",
        level="tile",
        model=object(),
        transforms=object(),
        feature_dim=3,
        device=torch.device("cpu"),
    )

    with pytest.raises(ValueError, match="missing tiles_tar_path"):
        inference._compute_tile_embeddings_for_slide(
            loaded,
            SimpleNamespace(level="tile"),
            make_slide("slide-a"),
            SimpleNamespace(
                x=np.array([0]),
                y=np.array([1]),
                requested_spacing_um=0.5,
                requested_tile_size_px=4,
                read_spacing_um=0.5,
                read_tile_size_px=4,
                tile_size_lv0=224,
                tiles_tar_path=None,
            ),
            preprocessing=replace(DEFAULT_PREPROCESSING, on_the_fly=False),
            execution=ExecutionOptions(batch_size=1, num_workers=0, num_gpus=1),
        )


def test_build_hierarchical_index_is_region_major_and_row_major_within_region():
    import slide2vec.inference as inference

    tiling_result = SimpleNamespace(
        x=np.array([100, 1000], dtype=np.int64),
        y=np.array([200, 1200], dtype=np.int64),
        tile_size_lv0=672,
        read_region_size_px=672,
        requested_region_size_px=672,
        read_tile_size_px=224,
        requested_tile_size_px=224,
    )

    index = inference._build_hierarchical_index(
        tiling_result,
        region_tile_multiple=3,
    )

    np.testing.assert_array_equal(index.flat_index, np.arange(18, dtype=np.int64))
    np.testing.assert_array_equal(
        index.region_index,
        np.array([0] * 9 + [1] * 9, dtype=np.int32),
    )
    np.testing.assert_array_equal(
        index.subtile_index_within_region,
        np.array(list(range(9)) * 2, dtype=np.int32),
    )
    np.testing.assert_array_equal(
        index.subtile_x[:9],
        np.array([100, 324, 548, 100, 324, 548, 100, 324, 548], dtype=np.int64),
    )
    np.testing.assert_array_equal(
        index.subtile_y[:9],
        np.array([200, 200, 200, 424, 424, 424, 648, 648, 648], dtype=np.int64),
    )


def test_resolve_hierarchical_geometry_scales_tile_first_under_spacing_mismatch():
    import slide2vec.inference as inference

    preprocessing = PreprocessingConfig(
        requested_spacing_um=0.5,
        requested_tile_size_px=224,
        requested_region_size_px=1792,
        region_tile_multiple=8,
    )
    tiling_result = SimpleNamespace(
        read_tile_size_px=3319,
        read_spacing_um=0.27,
        tile_size_lv0=3319,
        base_spacing_um=0.27,
    )

    geometry = inference._resolve_hierarchical_geometry(preprocessing, tiling_result)

    assert geometry["read_tile_size_px"] == 415
    assert geometry["read_region_size_px"] == 3320
    assert geometry["tile_size_lv0"] == 415
    assert geometry["tiles_per_region"] == 64


def test_resolve_hierarchical_geometry_keeps_level0_footprint_when_spacing_matches_base():
    import slide2vec.inference as inference

    preprocessing = PreprocessingConfig(
        requested_spacing_um=0.5,
        requested_tile_size_px=224,
        requested_region_size_px=448,
        region_tile_multiple=2,
    )
    tiling_result = SimpleNamespace(
        read_tile_size_px=224,
        read_spacing_um=0.486187607049942,
        tile_size_lv0=224,
        base_spacing_um=0.486187607049942,
    )

    geometry = inference._resolve_hierarchical_geometry(preprocessing, tiling_result)

    assert geometry["read_tile_size_px"] == 224
    assert geometry["tile_size_lv0"] == 224


def test_build_hierarchical_index_uses_tile_first_level0_offsets_under_spacing_mismatch():
    import slide2vec.inference as inference

    tiling_result = SimpleNamespace(
        x=np.array([100], dtype=np.int64),
        y=np.array([200], dtype=np.int64),
    )

    index = inference._build_hierarchical_index(
        tiling_result,
        region_tile_multiple=8,
        tile_size_lv0=415,
    )

    assert index.tiles_per_region == 64
    np.testing.assert_array_equal(index.subtile_x[:8], np.array([100, 515, 930, 1345, 1760, 2175, 2590, 3005], dtype=np.int64))
    np.testing.assert_array_equal(index.subtile_y[::8], np.array([200, 615, 1030, 1445, 1860, 2275, 2690, 3105], dtype=np.int64))


def test_merge_hierarchical_embedding_shards_restores_original_region_shape():
    from slide2vec.runtime.distributed import merge_hierarchical_embedding_shards

    merged = merge_hierarchical_embedding_shards(
        [
            {
                "flat_index": np.array([2, 0, 7], dtype=np.int64),
                "tile_embeddings": np.array([[20.0, 21.0], [0.0, 1.0], [70.0, 71.0]], dtype=np.float32),
            },
            {
                "flat_index": np.array([6, 3, 1, 5, 4], dtype=np.int64),
                "tile_embeddings": np.array(
                    [[60.0, 61.0], [30.0, 31.0], [10.0, 11.0], [50.0, 51.0], [40.0, 41.0]],
                    dtype=np.float32,
                ),
            },
        ],
        num_regions=2,
        tiles_per_region=4,
    )

    np.testing.assert_array_equal(
        merged,
        np.array(
            [
                [[0.0, 1.0], [10.0, 11.0], [20.0, 21.0], [30.0, 31.0]],
                [[40.0, 41.0], [50.0, 51.0], [60.0, 61.0], [70.0, 71.0]],
            ],
            dtype=np.float32,
        ),
    )


def test_compute_hierarchical_embeddings_for_slide_encodes_flat_tile_batches_and_reshapes(monkeypatch):
    import slide2vec.inference as inference
    torch = pytest.importorskip("torch")

    captured = {}

    class DummyDataset:
        def __init__(self, flat_indices):
            self._flat_indices = np.asarray(flat_indices, dtype=np.int64)

        def __len__(self):
            return int(self._flat_indices.shape[0])

        def __getitem__(self, idx):
            return int(self._flat_indices[idx])

    class DummyLoader:
        def __init__(self, dataset, **kwargs):
            captured["loader_kwargs"] = kwargs
            self._batches = [
                (
                    torch.tensor([0, 3, 4, 7], dtype=torch.long),
                    torch.tensor(
                        [
                            [[[0, 0], [0, 0]]] * 3,
                            [[[3, 3], [3, 3]]] * 3,
                            [[[4, 4], [4, 4]]] * 3,
                            [[[7, 7], [7, 7]]] * 3,
                        ],
                        dtype=torch.uint8,
                    ),
                    {"worker_batch_ms": 0.0, "reader_open_ms": 0.0, "reader_read_ms": 0.0},
                ),
                (
                    torch.tensor([1, 2, 5, 6], dtype=torch.long),
                    torch.tensor(
                        [
                            [[[1, 1], [1, 1]]] * 3,
                            [[[2, 2], [2, 2]]] * 3,
                            [[[5, 5], [5, 5]]] * 3,
                            [[[6, 6], [6, 6]]] * 3,
                        ],
                        dtype=torch.uint8,
                    ),
                    {"worker_batch_ms": 0.0, "reader_open_ms": 0.0, "reader_read_ms": 0.0},
                ),
            ]

        def __iter__(self):
            return iter(self._batches)

        def __len__(self):
            return len(self._batches)

    class DummyTileModel:
        def encode_tiles(self, image):
            assert image.ndim == 4
            values = image[:, 0, 0, 0].to(torch.float32)
            return torch.stack((values, values + 100.0), dim=1)

    class DummyCollator:
        def __init__(self, **kwargs):
            captured["collator_kwargs"] = kwargs

        def build_batch_sampler(self, *, batch_size, dataset_indices):
            return None

    monkeypatch.setattr(inference, "TileIndexDataset", DummyDataset)
    monkeypatch.setattr(inference, "OnTheFlyHierarchicalBatchCollator", DummyCollator)
    monkeypatch.setattr(torch.utils.data, "DataLoader", DummyLoader)

    loaded = inference.LoadedModel(
        name="uni",
        level="tile",
        model=DummyTileModel(),
        transforms=SimpleNamespace(transforms=[]),
        feature_dim=2,
        device=torch.device("cpu"),
    )
    slide = make_slide("slide-h")
    tiling_result = SimpleNamespace(
        x=np.array([0, 100], dtype=np.int64),
        y=np.array([0, 100], dtype=np.int64),
        requested_tile_size_px=224,
        read_tile_size_px=224,
        requested_region_size_px=448,
        read_region_size_px=448,
        tile_size_lv0=448,
        requested_spacing_um=0.5,
        read_spacing_um=0.5,
        base_spacing_um=0.5,
        read_level=0,
    )

    result = inference._compute_hierarchical_embeddings_for_slide(
        loaded,
        slide,
        tiling_result,
        preprocessing=replace(DEFAULT_PREPROCESSING, region_tile_multiple=2, requested_region_size_px=448),
        execution=ExecutionOptions(batch_size=4, num_workers=0, num_gpus=1),
    )

    assert result.shape == (2, 4, 2)
    np.testing.assert_array_equal(
        result.numpy(),
        np.array(
            [
                [[0.0 / 255.0, 100.0], [1.0 / 255.0, 100.0 + 1.0 / 255.0], [2.0 / 255.0, 100.0 + 2.0 / 255.0], [3.0 / 255.0, 100.0 + 3.0 / 255.0]],
                [[4.0 / 255.0, 100.0 + 4.0 / 255.0], [5.0 / 255.0, 100.0 + 5.0 / 255.0], [6.0 / 255.0, 100.0 + 6.0 / 255.0], [7.0 / 255.0, 100.0 + 7.0 / 255.0]],
            ],
            dtype=np.float32,
        ),
    )
    assert "collator_kwargs" in captured


def test_load_model_auto_prefers_cuda_when_available(monkeypatch):
    import torch
    import slide2vec.inference as inference
    import slide2vec.encoders.base as base

    class FakeModel:
        def eval(self):
            return self

        def to(self, device):
            self.device = torch.device(device)
            return self

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(base.timm, "create_model", lambda *args, **kwargs: FakeModel())
    monkeypatch.delenv("HF_TOKEN", raising=False)

    loaded = inference.load_model(name="h0-mini", device="auto")

    assert loaded.device == torch.device("cuda")


def test_load_model_accepts_allow_non_recommended_settings_without_forwarding(monkeypatch):
    import slide2vec.inference as inference

    captured = {}

    class DummyEncoder:
        def __init__(self, *, output_variant=None):
            captured["output_variant"] = output_variant
            self.device = "cpu"
            self.encode_dim = 8

        def get_transform(self):
            return SimpleNamespace()

        def to(self, device):
            self.device = device
            return self

    monkeypatch.setattr(inference, "canonicalize_model_name", lambda name: name)
    monkeypatch.setattr(
        inference.encoder_registry,
        "info",
        lambda name: {"level": "tile", "precision": "fp32"},
    )
    monkeypatch.setattr(inference.encoder_registry, "require", lambda name: DummyEncoder)
    monkeypatch.delenv("HF_TOKEN", raising=False)

    loaded = inference.load_model(
        name="dummy-model",
        allow_non_recommended_settings=True,
    )

    assert loaded.name == "dummy-model"
    assert captured["output_variant"] is None


def test_scale_coordinates_scales_down():
    from slide2vec.runtime.tiling import scale_coordinates

    coords = np.array([[10, 20], [30, 40]])
    # base=0.25, target=0.5 → scale=0.5 → coordinates halved
    result = scale_coordinates(coords, base_spacing_um=0.25, spacing=0.5)
    np.testing.assert_array_equal(result, [[5, 10], [15, 20]])


def test_scale_coordinates_identity_when_spacings_equal():
    from slide2vec.runtime.tiling import scale_coordinates

    coords = np.array([[10, 20], [30, 40]])
    result = scale_coordinates(coords, base_spacing_um=0.5, spacing=0.5)
    np.testing.assert_array_equal(result, [[10, 20], [30, 40]])
