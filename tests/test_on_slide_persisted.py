"""Per-slide completion hook (``on_slide_persisted``) on the tile-embedding entry points."""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import slide2vec.inference as inference
from slide2vec.api import (
    ExecutionOptions,
    Model,
    Pipeline,
    PreprocessingConfig as BasePreprocessingConfig,
)
from slide2vec.artifacts import (
    HierarchicalEmbeddingArtifact,
    TileEmbeddingArtifact,
    write_hierarchical_embeddings,
    write_tile_embeddings,
)
from slide2vec.runtime import artifacts_collect, distributed_stage, embedding_pipeline, manifest

PREPROCESSING = BasePreprocessingConfig(requested_spacing_um=0.5, requested_tile_size_px=224)

PROCESS_LIST_HEADER = (
    "sample_id,annotation,image_path,mask_path,requested_backend,backend,spacing_at_level_0,"
    "tiling_status,num_tiles,coordinates_npz_path,coordinates_meta_path,feature_status,error,traceback\n"
)


def make_slide(sample_id: str):
    return SimpleNamespace(
        sample_id=sample_id,
        image_path=Path(f"/tmp/{sample_id}.svs"),
        mask_path=None,
        spacing_at_level_0=None,
    )


def make_tiling_result(num_tiles: int = 1):
    return SimpleNamespace(
        x=np.arange(num_tiles),
        y=np.arange(num_tiles),
        tile_size_lv0=224,
    )


def make_tile_model():
    return SimpleNamespace(
        name="virchow2",
        level="tile",
        _requested_device="cpu",
        _declare_encoder_input=lambda *args, **kwargs: None,
        _load_backend=lambda: SimpleNamespace(feature_dim=2, device="cpu", model=SimpleNamespace()),
    )


def write_process_list(path: Path, rows: list[tuple[str, str, int, str]]) -> None:
    """Write ``(sample_id, annotation, num_tiles, feature_status)`` rows in the tiled-slide layout."""
    lines = [PROCESS_LIST_HEADER]
    for sample_id, annotation, num_tiles, feature_status in rows:
        lines.append(
            f"{sample_id},{annotation},/tmp/{sample_id}.svs,,asap,asap,,success,{num_tiles},"
            f"/tmp/{sample_id}.coordinates.npz,/tmp/{sample_id}.coordinates.meta.json,{feature_status},,\n"
        )
    path.write_text("".join(lines), encoding="utf-8")


class HookRecorder:
    """Records each hook call together with whether the artifact file existed at call time."""

    def __init__(self) -> None:
        self.artifacts: list = []
        self.existed_on_disk: list[bool] = []

    def __call__(self, artifact) -> None:
        self.artifacts.append(artifact)
        self.existed_on_disk.append(Path(artifact.path).is_file())

    @property
    def sample_ids(self) -> list[str]:
        return [artifact.sample_id for artifact in self.artifacts]


def fake_tile_embeddings(_loaded, _model, slide, _tiling_result, **_kwargs):
    return np.array([[1.0, 2.0]], dtype=np.float32)


# --- Model.embed_tiles ------------------------------------------------------------------------


def test_embed_tiles_calls_hook_once_per_artifact_in_slide_order_before_returning(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(embedding_pipeline, "compute_tile_embeddings_for_slide", fake_tile_embeddings)
    hook = HookRecorder()

    artifacts = inference.embed_tiles(
        make_tile_model(),
        [make_slide("slide-a"), make_slide("slide-b")],
        [make_tiling_result(), make_tiling_result()],
        execution=ExecutionOptions(output_dir=tmp_path, output_format="npz"),
        preprocessing=PREPROCESSING,
        on_slide_persisted=hook,
    )

    assert hook.sample_ids == ["slide-a", "slide-b"]
    assert all(isinstance(artifact, TileEmbeddingArtifact) for artifact in hook.artifacts)
    assert hook.existed_on_disk == [True, True]
    assert hook.artifacts == artifacts


def test_embed_tiles_hook_exception_propagates(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(embedding_pipeline, "compute_tile_embeddings_for_slide", fake_tile_embeddings)

    def failing_hook(artifact):
        raise RuntimeError("hook boom")

    with pytest.raises(RuntimeError, match="hook boom"):
        inference.embed_tiles(
            make_tile_model(),
            [make_slide("slide-a")],
            [make_tiling_result()],
            execution=ExecutionOptions(output_dir=tmp_path, output_format="npz"),
            preprocessing=PREPROCESSING,
            on_slide_persisted=failing_hook,
        )


def test_model_embed_tiles_forwards_hook(monkeypatch, tmp_path: Path):
    captured = {}

    def fake_embed_tiles(model_arg, slides, tiling_results, *, execution, preprocessing=None, on_slide_persisted=None):
        captured["on_slide_persisted"] = on_slide_persisted
        return []

    monkeypatch.setattr("slide2vec.inference.embed_tiles", fake_embed_tiles)
    hook = HookRecorder()

    Model.from_preset("virchow2").embed_tiles(
        slides=[{"sample_id": "slide-a", "image_path": "/tmp/slide-a.svs"}],
        tiling_results=[make_tiling_result()],
        preprocessing=PREPROCESSING,
        execution=ExecutionOptions(output_dir=tmp_path),
        on_slide_persisted=hook,
    )

    assert captured["on_slide_persisted"] is hook


# --- Pipeline.run_with_coordinates, num_gpus == 1 ---------------------------------------------


def _single_gpu_run(monkeypatch, tmp_path: Path, *, hook, compute=fake_tile_embeddings, slides=None, tiling_results=None):
    coordinates_dir = tmp_path / "coords"
    coordinates_dir.mkdir()
    output_dir = tmp_path / "out"
    slides = slides or [make_slide("slide-a"), make_slide("slide-b")]
    tiling_results = tiling_results or [make_tiling_result() for _ in slides]
    write_process_list(
        coordinates_dir / "process_list.csv",
        [(slide.sample_id, "tissue", int(tiling_result.x.size), "tbp") for slide, tiling_result in zip(slides, tiling_results)],
    )
    monkeypatch.setattr(manifest, "load_successful_tiled_slides", lambda path: (slides, tiling_results))
    monkeypatch.setattr(embedding_pipeline, "compute_tile_embeddings_for_slide", compute)
    return inference.run_pipeline_with_coordinates(
        make_tile_model(),
        coordinates_dir=coordinates_dir,
        preprocessing=PREPROCESSING,
        execution=ExecutionOptions(output_dir=output_dir, output_format="npz", num_gpus=1),
        on_slide_persisted=hook,
    )


def test_run_with_coordinates_single_gpu_calls_hook_per_slide_and_keeps_result(monkeypatch, tmp_path: Path):
    hook = HookRecorder()

    result = _single_gpu_run(monkeypatch, tmp_path, hook=hook)

    assert hook.sample_ids == ["slide-a", "slide-b"]
    assert all(isinstance(artifact, TileEmbeddingArtifact) for artifact in hook.artifacts)
    assert hook.existed_on_disk == [True, True]
    assert [artifact.sample_id for artifact in result.tile_artifacts] == ["slide-a", "slide-b"]
    assert [artifact.path for artifact in result.tile_artifacts] == [artifact.path for artifact in hook.artifacts]


def test_run_with_coordinates_single_gpu_skips_zero_tile_slides(monkeypatch, tmp_path: Path):
    hook = HookRecorder()
    slides = [make_slide("slide-a"), make_slide("slide-empty")]
    tiling_results = [make_tiling_result(), make_tiling_result(num_tiles=0)]

    result = _single_gpu_run(monkeypatch, tmp_path, hook=hook, slides=slides, tiling_results=tiling_results)

    assert hook.sample_ids == ["slide-a"]
    assert [artifact.sample_id for artifact in result.tile_artifacts] == ["slide-a"]


def test_run_with_coordinates_single_gpu_failed_slide_does_not_fire_hook(monkeypatch, tmp_path: Path):
    hook = HookRecorder()

    def compute(_loaded, _model, slide, _tiling_result, **_kwargs):
        if slide.sample_id == "slide-b":
            raise RuntimeError("embedding boom")
        return np.array([[1.0, 2.0]], dtype=np.float32)

    with pytest.raises(RuntimeError, match="embedding boom"):
        _single_gpu_run(monkeypatch, tmp_path, hook=hook, compute=compute)

    assert hook.sample_ids == ["slide-a"]


def test_run_with_coordinates_single_gpu_hook_exception_propagates(monkeypatch, tmp_path: Path):
    def failing_hook(artifact):
        raise RuntimeError("hook boom")

    with pytest.raises(RuntimeError, match="hook boom"):
        _single_gpu_run(monkeypatch, tmp_path, hook=failing_hook)


def test_pipeline_run_with_coordinates_forwards_hook(monkeypatch, tmp_path: Path):
    captured = {}

    def fake_run(model, *, coordinates_dir, slides, preprocessing, execution, on_slide_persisted=None):
        captured["on_slide_persisted"] = on_slide_persisted
        return "result"

    monkeypatch.setattr("slide2vec.inference.run_pipeline_with_coordinates", fake_run)
    hook = HookRecorder()
    pipeline = Pipeline(
        model=Model.from_preset("virchow2"),
        preprocessing=PREPROCESSING,
        execution=ExecutionOptions(output_dir=tmp_path),
    )

    assert pipeline.run_with_coordinates(tmp_path, on_slide_persisted=hook) == "result"
    assert captured["on_slide_persisted"] is hook


# --- Pipeline.run_with_coordinates, num_gpus > 1 ----------------------------------------------


def test_run_with_coordinates_multi_gpu_forwards_hook_to_distributed_collector(monkeypatch, tmp_path: Path):
    coordinates_dir = tmp_path / "coords"
    slide = make_slide("slide-a")
    monkeypatch.setattr(manifest, "load_successful_tiled_slides", lambda path: ([slide], [make_tiling_result()]))
    monkeypatch.setattr(distributed_stage, "validate_multi_gpu_execution", lambda *args, **kwargs: None)
    captured = {}

    def fake_collect(*, on_slide_persisted=None, **kwargs):
        captured["on_slide_persisted"] = on_slide_persisted
        return [], [], []

    monkeypatch.setattr(artifacts_collect, "collect_distributed_pipeline_artifacts", fake_collect)
    hook = HookRecorder()

    inference.run_pipeline_with_coordinates(
        Model.from_preset("virchow2"),
        coordinates_dir=coordinates_dir,
        preprocessing=PREPROCESSING,
        execution=ExecutionOptions(output_dir=tmp_path / "out", num_gpus=2),
        on_slide_persisted=hook,
    )

    assert captured["on_slide_persisted"] is hook


def _distributed_collect(monkeypatch, tmp_path: Path, *, fake_run_stage, hook, slides, preprocessing=PREPROCESSING):
    monkeypatch.setattr(artifacts_collect, "run_distributed_embedding_stage", fake_run_stage)
    return artifacts_collect.collect_distributed_pipeline_artifacts(
        model=SimpleNamespace(name="virchow2", level="tile"),
        successful_slides=slides,
        process_list_path=tmp_path / "process_list.csv",
        preprocessing=preprocessing,
        execution=ExecutionOptions(output_dir=tmp_path, num_gpus=2, output_format="npz"),
        output_dir=tmp_path,
        on_slide_persisted=hook,
    )


def _finished_event(sample_id: str):
    return SimpleNamespace(kind="embedding.slide.finished", payload={"sample_id": sample_id})


def test_distributed_hook_fires_as_each_finished_event_is_consumed(monkeypatch, tmp_path: Path):
    write_process_list(tmp_path / "process_list.csv", [("slide-a", "tissue", 1, "tbp"), ("slide-b", "tissue", 1, "tbp")])
    hook = HookRecorder()
    seen_inside_stage = {}

    def fake_run_stage(*, on_progress_event=None, **kwargs):
        write_tile_embeddings("slide-a", np.zeros((1, 2), dtype=np.float32), output_dir=tmp_path, output_format="npz")
        on_progress_event(_finished_event("slide-a"))
        seen_inside_stage["after_a"] = list(hook.sample_ids)
        write_tile_embeddings("slide-b", np.zeros((1, 2), dtype=np.float32), output_dir=tmp_path, output_format="npz")
        on_progress_event(_finished_event("slide-b"))
        seen_inside_stage["after_b"] = list(hook.sample_ids)

    tile_artifacts, _, _ = _distributed_collect(
        monkeypatch, tmp_path, fake_run_stage=fake_run_stage, hook=hook,
        slides=[make_slide("slide-a"), make_slide("slide-b")],
    )

    # Fired from the parent as each event was consumed, not after the stage returned.
    assert seen_inside_stage["after_a"] == ["slide-a"]
    assert seen_inside_stage["after_b"] == ["slide-a", "slide-b"]
    # The end-of-run reconcile does not fire it again.
    assert hook.sample_ids == ["slide-a", "slide-b"]
    assert all(isinstance(artifact, TileEmbeddingArtifact) for artifact in hook.artifacts)
    assert hook.existed_on_disk == [True, True]
    assert [artifact.path for artifact in tile_artifacts] == [artifact.path for artifact in hook.artifacts]


def test_distributed_hook_fires_once_per_annotation_artifact_for_multi_class_sample(monkeypatch, tmp_path: Path):
    write_process_list(tmp_path / "process_list.csv", [("slide-a", "tumor", 1, "tbp"), ("slide-a", "stroma", 1, "tbp")])
    hook = HookRecorder()
    seen_inside_stage = {}

    def fake_run_stage(*, on_progress_event=None, **kwargs):
        write_tile_embeddings("slide-a", np.zeros((1, 2), dtype=np.float32), output_dir=tmp_path, output_format="npz", annotation="tumor")
        on_progress_event(_finished_event("slide-a"))
        seen_inside_stage["after_first"] = list(hook.sample_ids)
        write_tile_embeddings("slide-a", np.zeros((1, 2), dtype=np.float32), output_dir=tmp_path, output_format="npz", annotation="stroma")
        on_progress_event(_finished_event("slide-a"))

    slide = make_slide("slide-a")
    _distributed_collect(monkeypatch, tmp_path, fake_run_stage=fake_run_stage, hook=hook, slides=[slide, slide])

    # Deferred while the sibling class was still in flight, then once per annotation artifact.
    assert seen_inside_stage["after_first"] == []
    assert [artifact.annotation for artifact in hook.artifacts] == ["tumor", "stroma"]
    assert hook.sample_ids == ["slide-a", "slide-a"]


def test_distributed_hook_skips_resume_completed_slides(monkeypatch, tmp_path: Path):
    write_process_list(tmp_path / "process_list.csv", [("slide-a", "tissue", 1, "success"), ("slide-b", "tissue", 1, "tbp")])
    write_tile_embeddings("slide-a", np.zeros((1, 2), dtype=np.float32), output_dir=tmp_path, output_format="npz")
    hook = HookRecorder()

    def fake_run_stage(*, successful_slides, on_progress_event=None, **kwargs):
        assert [slide.sample_id for slide in successful_slides] == ["slide-b"]
        write_tile_embeddings("slide-b", np.zeros((1, 2), dtype=np.float32), output_dir=tmp_path, output_format="npz")
        on_progress_event(_finished_event("slide-b"))

    tile_artifacts, _, _ = _distributed_collect(
        monkeypatch, tmp_path, fake_run_stage=fake_run_stage, hook=hook,
        slides=[make_slide("slide-a"), make_slide("slide-b")],
        preprocessing=replace(PREPROCESSING, resume=True),
    )

    assert hook.sample_ids == ["slide-b"]
    assert [artifact.sample_id for artifact in tile_artifacts] == ["slide-a", "slide-b"]


def test_distributed_hook_receives_hierarchical_artifact_under_hierarchical_preprocessing(monkeypatch, tmp_path: Path):
    write_process_list(tmp_path / "process_list.csv", [("slide-a", "tissue", 1, "tbp")])
    hook = HookRecorder()

    def fake_run_stage(*, on_progress_event=None, **kwargs):
        write_hierarchical_embeddings("slide-a", np.zeros((1, 2, 4), dtype=np.float32), output_dir=tmp_path, output_format="npz")
        on_progress_event(_finished_event("slide-a"))

    _distributed_collect(
        monkeypatch, tmp_path, fake_run_stage=fake_run_stage, hook=hook,
        slides=[make_slide("slide-a")],
        preprocessing=replace(PREPROCESSING, requested_region_size_px=448, region_tile_multiple=2),
    )

    assert hook.sample_ids == ["slide-a"]
    assert isinstance(hook.artifacts[0], HierarchicalEmbeddingArtifact)
    assert hook.existed_on_disk == [True]


def test_distributed_hook_exception_propagates(monkeypatch, tmp_path: Path):
    write_process_list(tmp_path / "process_list.csv", [("slide-a", "tissue", 1, "tbp")])

    def failing_hook(artifact):
        raise RuntimeError("hook boom")

    def fake_run_stage(*, on_progress_event=None, **kwargs):
        write_tile_embeddings("slide-a", np.zeros((1, 2), dtype=np.float32), output_dir=tmp_path, output_format="npz")
        on_progress_event(_finished_event("slide-a"))

    with pytest.raises(RuntimeError, match="hook boom"):
        _distributed_collect(monkeypatch, tmp_path, fake_run_stage=fake_run_stage, hook=failing_hook, slides=[make_slide("slide-a")])
