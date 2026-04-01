import ast
import sys
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
    PreprocessingConfig,
)
from slide2vec.artifacts import (
    load_array,
    load_metadata,
    write_slide_embeddings,
    write_tile_embeddings,
)
from slide2vec.resources import config_resource, load_config

ROOT = Path(__file__).resolve().parents[1]


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


def test_load_model_merges_preprocessing_defaults_for_cross_file_interpolations(monkeypatch):
    import slide2vec.inference as inference

    class AttrDict(dict):
        def __getattr__(self, name):
            try:
                return self[name]
            except KeyError as exc:
                raise AttributeError(name) from exc

        def __setattr__(self, name, value):
            self[name] = value

    def convert(value):
        if isinstance(value, dict):
            return AttrDict({key: convert(item) for key, item in value.items()})
        if isinstance(value, list):
            return [convert(item) for item in value]
        return value

    def merge_values(left, right):
        if isinstance(left, dict) and isinstance(right, dict):
            merged = AttrDict({key: convert(value) for key, value in left.items()})
            for key, value in right.items():
                if key in merged:
                    merged[key] = merge_values(merged[key], value)
                else:
                    merged[key] = convert(value)
            return merged
        return convert(right)

    def lookup(root, path):
        current = root
        for segment in path.split("."):
            current = current[segment]
        return current

    def resolve_value(root, value):
        if isinstance(value, dict):
            for key, item in list(value.items()):
                value[key] = resolve_value(root, item)
            return value
        if isinstance(value, list):
            for index, item in enumerate(list(value)):
                value[index] = resolve_value(root, item)
            return value
        if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
            return resolve_value(root, lookup(root, value[2:-1]))
        return value

    class FakeOmegaConf:
        @staticmethod
        def create(value):
            return convert(value)

        @staticmethod
        def merge(*values):
            merged = AttrDict()
            for value in values:
                merged = merge_values(merged, convert(value))
            return merged

        @staticmethod
        def resolve(value):
            resolve_value(value, value)

    captured: dict[str, object] = {}

    class FakeBackend:
        device = "cpu"
        features_dim = 128

        def to(self, _device):
            return self

        def get_transforms(self):
            return "TRANSFORMS"

    class FakeModelFactory:
        def __init__(self, options):
            captured["options"] = options

        def get_model(self):
            return FakeBackend()

    def fake_load_config(*parts):
        if parts == ("preprocessing", "default"):
            return {
                "tiling": {
                    "params": {
                        "target_tile_size_px": 256,
                    }
                }
            }
        if parts == ("models", "default"):
            return {
                "model": {
                    "mode": "cls",
                    "input_size": "${tiling.params.target_tile_size_px}",
                    "patch_size": 256,
                    "token_size": 16,
                    "normalize_embeddings": False,
                }
            }
        if parts == ("models", "h0-mini"):
            return {"model": {}}
        raise AssertionError(parts)

    monkeypatch.setitem(sys.modules, "omegaconf", types.SimpleNamespace(OmegaConf=FakeOmegaConf))
    monkeypatch.setitem(sys.modules, "slide2vec.models", types.SimpleNamespace(ModelFactory=FakeModelFactory))
    monkeypatch.setitem(sys.modules, "slide2vec.resources", types.SimpleNamespace(load_config=fake_load_config))
    monkeypatch.setattr(inference, "_resolve_device", lambda requested, device: device)

    loaded = inference.load_model(name="h0-mini", level="tile")

    assert captured["options"].name == "h0-mini"
    assert captured["options"].level == "tile"
    assert captured["options"].mode == "cls"
    assert captured["options"].input_size == 256
    assert loaded.feature_dim == 128


def test_load_model_uses_conchv15_preset_for_canonicalized_alias(monkeypatch):
    import slide2vec.inference as inference

    class AttrDict(dict):
        def __getattr__(self, name):
            try:
                return self[name]
            except KeyError as exc:
                raise AttributeError(name) from exc

        def __setattr__(self, name, value):
            self[name] = value

    def convert(value):
        if isinstance(value, dict):
            return AttrDict({key: convert(item) for key, item in value.items()})
        if isinstance(value, list):
            return [convert(item) for item in value]
        return value

    def merge_values(left, right):
        if isinstance(left, dict) and isinstance(right, dict):
            merged = AttrDict({key: convert(value) for key, value in left.items()})
            for key, value in right.items():
                if key in merged:
                    merged[key] = merge_values(merged[key], value)
                else:
                    merged[key] = convert(value)
            return merged
        return convert(right)

    def lookup(root, path):
        current = root
        for segment in path.split("."):
            current = current[segment]
        return current

    def resolve_value(root, value):
        if isinstance(value, dict):
            for key, item in list(value.items()):
                value[key] = resolve_value(root, item)
            return value
        if isinstance(value, list):
            for index, item in enumerate(list(value)):
                value[index] = resolve_value(root, item)
            return value
        if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
            return resolve_value(root, lookup(root, value[2:-1]))
        return value

    class FakeOmegaConf:
        @staticmethod
        def create(value):
            return convert(value)

        @staticmethod
        def merge(*values):
            merged = AttrDict()
            for value in values:
                merged = merge_values(merged, convert(value))
            return merged

        @staticmethod
        def resolve(value):
            resolve_value(value, value)

    captured = {}

    class FakeBackend:
        device = "cpu"
        features_dim = 768

        def to(self, _device):
            return self

        def get_transforms(self):
            return "TRANSFORMS"

    class FakeModelFactory:
        def __init__(self, options):
            captured["options"] = options

        def get_model(self):
            return FakeBackend()

    def fake_load_config(*parts):
        if parts == ("preprocessing", "default"):
            return {"tiling": {"params": {"target_tile_size_px": 256}}}
        if parts == ("models", "default"):
            return {"model": {"mode": "cls", "input_size": "${tiling.params.target_tile_size_px}"}}
        if parts == ("models", "conchv15"):
            return {"model": {"name": "conchv15", "input_size": 448}}
        raise AssertionError(parts)

    monkeypatch.setitem(sys.modules, "omegaconf", types.SimpleNamespace(OmegaConf=FakeOmegaConf))
    monkeypatch.setitem(sys.modules, "slide2vec.models", types.SimpleNamespace(ModelFactory=FakeModelFactory))
    monkeypatch.setitem(sys.modules, "slide2vec.resources", types.SimpleNamespace(load_config=fake_load_config))
    monkeypatch.setattr(inference, "_resolve_device", lambda requested, device: device)

    loaded = inference.load_model(name="conchv15", level="tile")

    assert captured["options"].name == "conchv15"
    assert captured["options"].input_size == 448
    assert loaded.feature_dim == 768

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
        lambda *args, **kwargs: (["tile-artifact"], ["slide-artifact"]),
    )
    monkeypatch.setattr(inference, "_update_process_list_after_embedding", lambda *args, **kwargs: None)

    result = inference.run_pipeline(
        model,
        slides=[slide],
        preprocessing=PreprocessingConfig(),
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
        return ["tile-artifact"], ["slide-artifact"]

    monkeypatch.setattr(inference, "_collect_distributed_pipeline_artifacts", fake_collect)

    result = inference.run_pipeline(
        Model.from_preset("virchow2"),
        slides=[slide],
        preprocessing=PreprocessingConfig(),
        execution=ExecutionOptions(output_dir=tmp_path, num_gpus=2),
    )

    assert captured["successful_slides"] == [slide]
    assert captured["process_list_path"] == tmp_path / "process_list.csv"
    assert isinstance(captured["preprocessing"], PreprocessingConfig)
    assert captured["output_dir"] == tmp_path
    assert captured["execution"].num_gpus == 2
    assert result.tile_artifacts == ["tile-artifact"]
    assert result.slide_artifacts == ["slide-artifact"]

def test_collect_distributed_pipeline_artifacts_runs_stage_collects_and_updates(monkeypatch, tmp_path: Path):
    import slide2vec.inference as inference

    model = Model.from_preset("virchow2", level="slide")
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

    def fake_collect(slides, *, output_dir, output_format, include_tile_embeddings, include_slide_embeddings):
        captured["collect"] = {
            "slides": slides,
            "output_dir": output_dir,
            "output_format": output_format,
            "include_tile_embeddings": include_tile_embeddings,
            "include_slide_embeddings": include_slide_embeddings,
        }
        return ["tile-artifact"], ["slide-artifact"]

    def fake_update(
        process_list_path_arg,
        *,
        successful_slides,
        persist_tile_embeddings,
        include_slide_embeddings,
        tile_artifacts,
        slide_artifacts,
    ):
        captured["update"] = {
            "process_list_path": process_list_path_arg,
            "successful_slides": successful_slides,
            "persist_tile_embeddings": persist_tile_embeddings,
            "include_slide_embeddings": include_slide_embeddings,
            "tile_artifacts": tile_artifacts,
            "slide_artifacts": slide_artifacts,
        }

    monkeypatch.setattr(inference, "_run_distributed_embedding_stage", fake_run_stage)
    monkeypatch.setattr(inference, "_collect_pipeline_artifacts", fake_collect)
    monkeypatch.setattr(inference, "_update_process_list_after_embedding", fake_update)

    tile_artifacts, slide_artifacts = inference._collect_distributed_pipeline_artifacts(
        model=model,
        successful_slides=[slide],
        process_list_path=process_list_path,
        preprocessing=PreprocessingConfig(),
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
    assert captured["collect"]["include_slide_embeddings"] is True

    assert captured["update"]["process_list_path"] == process_list_path
    assert captured["update"]["successful_slides"] == [slide]
    assert captured["update"]["persist_tile_embeddings"] is True
    assert captured["update"]["include_slide_embeddings"] is True
    assert captured["update"]["tile_artifacts"] == ["tile-artifact"]
    assert captured["update"]["slide_artifacts"] == ["slide-artifact"]

    assert tile_artifacts == ["tile-artifact"]
    assert slide_artifacts == ["slide-artifact"]

def test_collect_local_pipeline_artifacts_filters_none_artifacts(monkeypatch):
    import slide2vec.inference as inference

    embedded_slides = [
        EmbeddedSlide(
            sample_id="slide-a",
            tile_embeddings=np.zeros((1, 2), dtype=np.float32),
            slide_embedding=np.zeros((2,), dtype=np.float32),
            coordinates=np.array([[0, 0]], dtype=np.int64),
            tile_size_lv0=224,
            image_path=Path("/tmp/slide-a.svs"),
            mask_path=None,
        ),
        EmbeddedSlide(
            sample_id="slide-b",
            tile_embeddings=np.zeros((1, 2), dtype=np.float32),
            slide_embedding=None,
            coordinates=np.array([[1, 1]], dtype=np.int64),
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

    tile_artifacts, slide_artifacts = inference._collect_local_pipeline_artifacts(
        model=SimpleNamespace(),
        embedded_slides=embedded_slides,
        tiling_results=tiling_results,
        preprocessing=PreprocessingConfig(),
        execution=ExecutionOptions(output_dir=Path("/tmp")),
    )

    assert tile_artifacts == ["tile-a"]
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
        coordinates=np.array([[0, 1]], dtype=np.int64),
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
        lambda *args, **kwargs: (["tile-artifact"], ["slide-artifact"]),
    )
    monkeypatch.setattr(inference, "_update_process_list_after_embedding", lambda *args, **kwargs: None)

    result = inference.run_pipeline(
        Model.from_preset("virchow2"),
        slides=[slide_record],
        preprocessing=PreprocessingConfig(),
        execution=ExecutionOptions(output_dir=tmp_path),
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
        "sample_id,image_path,mask_path,spacing_at_level_0,tiling_status,num_tiles,coordinates_npz_path,coordinates_meta_path,feature_status,error,traceback\n"
        "slide-a,/tmp/slide-a.svs,,,success,1,/tmp/slide-a.coordinates.npz,/tmp/slide-a.coordinates.meta.json,tbp,,\n"
        "slide-b,/tmp/slide-b.svs,,,success,1,/tmp/slide-b.coordinates.npz,/tmp/slide-b.coordinates.meta.json,tbp,,\n",
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
            preprocessing=PreprocessingConfig(),
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
        "sample_id,image_path,mask_path,spacing_at_level_0,tiling_status,num_tiles,coordinates_npz_path,coordinates_meta_path,feature_status,error,traceback\n"
        "slide-a,/tmp/slide-a.svs,,,success,1,/tmp/slide-a.coordinates.npz,/tmp/slide-a.coordinates.meta.json,success,,\n"
        "slide-b,/tmp/slide-b.svs,,,success,1,/tmp/slide-b.coordinates.npz,/tmp/slide-b.coordinates.meta.json,tbp,,\n",
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
        preprocessing=PreprocessingConfig(resume=True),
        execution=ExecutionOptions(output_dir=tmp_path, output_format="npz", num_gpus=1),
    )

    assert computed_sample_ids == ["slide-b"]
    assert [artifact.sample_id for artifact in result.tile_artifacts] == ["slide-a", "slide-b"]
    assert result.slide_artifacts == []


def test_run_pipeline_local_persists_completed_embeddings_before_later_slide_failure(monkeypatch, tmp_path: Path):
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
        "sample_id,image_path,mask_path,spacing_at_level_0,tiling_status,num_tiles,coordinates_npz_path,coordinates_meta_path,error,traceback\n"
        "slide-a,/tmp/slide-a.svs,,,"  # spacing_at_level_0
        "success,2,/tmp/slide-a.coordinates.npz,/tmp/slide-a.coordinates.meta.json,,\n"
        "slide-b,/tmp/slide-b.svs,,,"
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
            preprocessing=PreprocessingConfig(),
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

    monkeypatch.setitem(sys.modules, "hs2p", SimpleNamespace(tile_slides=fake_tile_slides))
    monkeypatch.setattr(
        inference,
        "_build_hs2p_configs",
        lambda preprocessing: ("tiling", "segmentation", "filtering", "preview", None, False),
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
        PreprocessingConfig(on_the_fly=False),
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

    monkeypatch.setitem(sys.modules, "hs2p", SimpleNamespace(tile_slides=fake_tile_slides))
    monkeypatch.setattr(
        inference,
        "_build_hs2p_configs",
        lambda preprocessing: ("tiling", "segmentation", "filtering", "preview", None, False),
    )

    inference._tile_slides(
        [make_slide("slide-a")],
        PreprocessingConfig(read_tiles_from=Path("/tmp/existing-tiles")),
        output_dir=tmp_path,
        num_workers=0,
    )

    assert captured["kwargs"]["save_tiles"] is False


def test_build_hs2p_configs_constructs_preview_config(monkeypatch):
    import slide2vec.inference as inference

    class FakeTilingConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeSegmentationConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeFilterConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakePreviewConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setitem(
        sys.modules,
        "hs2p",
        SimpleNamespace(
            TilingConfig=FakeTilingConfig,
            SegmentationConfig=FakeSegmentationConfig,
            FilterConfig=FakeFilterConfig,
            PreviewConfig=FakePreviewConfig,
        ),
    )

    preprocessing = PreprocessingConfig(
        backend="asap",
        target_spacing_um=0.5,
        target_tile_size_px=224,
        tolerance=0.05,
        overlap=0.0,
        tissue_threshold=0.1,
        segmentation={"downsample": 64},
        filtering={"ref_tile_size": 224},
        preview={"save_mask_preview": True, "save_tiling_preview": False, "downsample": 32},
    )

    tiling_cfg, segmentation_cfg, filtering_cfg, preview_cfg, read_coordinates_from, resume = (
        inference._build_hs2p_configs(preprocessing)
    )

    assert tiling_cfg.kwargs["backend"] == "asap"
    assert segmentation_cfg.kwargs == {"downsample": 64}
    assert filtering_cfg.kwargs == {"ref_tile_size": 224}
    assert preview_cfg.kwargs == {
        "save_mask_preview": True,
        "save_tiling_preview": False,
        "downsample": 32,
    }
    assert read_coordinates_from is None
    assert resume is False


def test_prepare_tiled_slides_records_spacing_at_level_0_in_process_list(monkeypatch, tmp_path: Path):
    import slide2vec.inference as inference

    process_list_path = tmp_path / "process_list.csv"
    process_list_path.write_text(
        "sample_id,image_path,mask_path,tiling_status,num_tiles,coordinates_npz_path,coordinates_meta_path,error,traceback\n"
        "slide-a,/tmp/slide-a.svs,,success,1,/tmp/slide-a.coordinates.npz,/tmp/slide-a.coordinates.meta.json,,\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(inference, "_tile_slides", lambda *args, **kwargs: None)
    monkeypatch.setattr(inference, "_load_tiling_result_from_row", lambda row: SimpleNamespace())

    slide = make_slide("slide-a", spacing_at_level_0=0.25)

    inference._prepare_tiled_slides(
        [slide],
        PreprocessingConfig(),
        output_dir=tmp_path,
        num_workers=0,
    )

    recorded = pd.read_csv(process_list_path)
    assert recorded.loc[0, "spacing_at_level_0"] == pytest.approx(0.25)


def test_prepare_tiled_slides_records_preview_paths_in_process_list(monkeypatch, tmp_path: Path):
    import slide2vec.inference as inference

    process_list_path = tmp_path / "process_list.csv"
    process_list_path.write_text(
        "sample_id,image_path,mask_path,tiling_status,num_tiles,coordinates_npz_path,coordinates_meta_path,error,traceback\n"
        "slide-a,/tmp/slide-a.svs,,success,1,/tmp/slide-a.coordinates.npz,/tmp/slide-a.coordinates.meta.json,,\n",
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
    monkeypatch.setattr(inference, "_load_tiling_result_from_row", lambda row: SimpleNamespace())

    slide = make_slide("slide-a")

    inference._prepare_tiled_slides(
        [slide],
        PreprocessingConfig(),
        output_dir=tmp_path,
        num_workers=0,
    )

    recorded = pd.read_csv(process_list_path)
    assert Path(recorded.loc[0, "mask_preview_path"]) == Path("/tmp/preview/mask/slide-a.png")
    assert Path(recorded.loc[0, "tiling_preview_path"]) == Path("/tmp/preview/tiling/slide-a.png")


def test_resolve_slide_backend_uses_tiling_result_backend_for_auto():
    import slide2vec.inference as inference

    assert inference._resolve_slide_backend(PreprocessingConfig(backend="auto"), SimpleNamespace(backend="cucim")) == "cucim"
    assert inference._resolve_slide_backend(PreprocessingConfig(backend="auto"), SimpleNamespace(backend="asap")) == "asap"
    assert inference._resolve_slide_backend(PreprocessingConfig(backend="auto"), SimpleNamespace()) == "asap"
    assert inference._resolve_slide_backend(PreprocessingConfig(backend="cucim"), SimpleNamespace(backend="asap")) == "cucim"


def test_load_successful_tiled_slides_preserves_spacing_at_level_0(monkeypatch, tmp_path: Path):
    import slide2vec.inference as inference

    process_list_path = tmp_path / "process_list.csv"
    process_list_path.write_text(
        "sample_id,image_path,mask_path,spacing_at_level_0,tiling_status,num_tiles,coordinates_npz_path,coordinates_meta_path,error,traceback\n"
        "slide-a,/tmp/slide-a.svs,,0.25,success,1,/tmp/slide-a.coordinates.npz,/tmp/slide-a.coordinates.meta.json,,\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(inference, "_load_tiling_result_from_row", lambda row: SimpleNamespace())

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
        target_spacing_um=0.5,
    )

    @contextmanager
    def fake_coordination_dir(work_dir: Path):
        yield work_dir / "coord"

    monkeypatch.setattr(inference, "_distributed_coordination_dir", fake_coordination_dir)
    monkeypatch.setattr(inference, "_run_distributed_direct_embedding_stage", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        inference,
        "_load_tile_embedding_shards",
        lambda *_args, **_kwargs: [
            {
                "tile_index": np.array([0, 1], dtype=np.int64),
                "tile_embeddings": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
            }
        ],
    )

    loaded = SimpleNamespace(device="cpu", model=SimpleNamespace())
    model = SimpleNamespace(_load_backend=lambda: loaded)
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
        preprocessing=PreprocessingConfig(),
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
    np.testing.assert_array_equal(embedded.coordinates, np.array([[0, 2], [1, 3]], dtype=np.int64))

def test_select_embedding_path_uses_local_compute_when_single_gpu(monkeypatch):
    import slide2vec.inference as inference

    slide = make_slide("slide-a")
    tiling_result = SimpleNamespace(x=np.array([0]), y=np.array([1]), tile_size_lv0=224)
    expected = [
        EmbeddedSlide(
            sample_id="slide-a",
            tile_embeddings=np.zeros((1, 2), dtype=np.float32),
            slide_embedding=None,
            coordinates=np.array([[0, 1]], dtype=np.int64),
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
        preprocessing=PreprocessingConfig(),
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
        coordinates=np.array([[0, 1]], dtype=np.int64),
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
        preprocessing=PreprocessingConfig(),
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
        preprocessing=PreprocessingConfig(),
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
    assert tile_only.coordinates.shape == (2, 2)

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
        coordinates=np.array([[0, 2], [1, 3]], dtype=np.int64),
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
        preprocessing=PreprocessingConfig(),
        execution=ExecutionOptions(),
    )
    assert in_memory == [embedded]

    persisted = inference.embed_slides(
        model,
        [slide_record],
        preprocessing=PreprocessingConfig(),
        execution=ExecutionOptions(
            output_dir=tmp_path,
            output_format="npz",
            save_latents=True,
            save_tile_embeddings=True,
        ),
    )
    assert persisted == [embedded]
    assert (tmp_path / "tile_embeddings" / "slide-a.npz").is_file()
    assert (tmp_path / "slide_embeddings" / "slide-a.npz").is_file()


def test_direct_embed_slides_persists_completed_embeddings_before_later_slide_failure(monkeypatch, tmp_path: Path):
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
        "sample_id,image_path,mask_path,spacing_at_level_0,tiling_status,num_tiles,coordinates_npz_path,coordinates_meta_path,error,traceback\n"
        "slide-a,/tmp/slide-a.svs,,,"  # spacing_at_level_0
        "success,2,/tmp/slide-a.coordinates.npz,/tmp/slide-a.coordinates.meta.json,,\n"
        "slide-b,/tmp/slide-b.svs,,,"
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
            preprocessing=PreprocessingConfig(),
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
        coordinates=np.array([[0, 2], [1, 3]], dtype=np.int64),
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
        Model.from_preset("prism", level="slide"),
        slides=[slide_record],
        preprocessing=PreprocessingConfig(),
        execution=ExecutionOptions(output_dir=tmp_path, output_format="npz", save_tile_embeddings=False),
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
        coordinates=np.array([[0, 2], [1, 3]], dtype=np.int64),
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
        preprocessing=PreprocessingConfig(),
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
            coordinates=np.array([[0, 0], [1, 1], [2, 2]], dtype=np.int64),
            tile_size_lv0=224,
            image_path=Path("/tmp/slide-a.svs"),
            mask_path=None,
        ),
        EmbeddedSlide(
            sample_id="slide-b",
            tile_embeddings=np.zeros((1, 2), dtype=np.float32),
            slide_embedding=None,
            coordinates=np.array([[0, 0]], dtype=np.int64),
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
        preprocessing=PreprocessingConfig(),
        execution=ExecutionOptions(output_dir=tmp_path, output_format="npz", num_gpus=2),
    )

    assert result == expected
    assert captured["multi"]["slide_records"] == slides

def test_assign_slides_to_ranks_balances_by_tile_count():
    import slide2vec.inference as inference

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

    assignments = inference._assign_slides_to_ranks(slides, tiling_results, num_gpus=2)

    assert assignments == {
        0: ["slide-a", "slide-d"],
        1: ["slide-b", "slide-c"],
    }

def test_merge_tile_embedding_shards_restores_original_tile_order():
    import slide2vec.inference as inference

    merged = inference._merge_tile_embedding_shards(
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
        def __call__(self, image):
            return {"embedding": torch.zeros((image.shape[0], 5), dtype=torch.float32)}

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


def test_region_batch_preprocessor_resizes_whole_region_before_unfolding():
    import slide2vec.inference as inference
    torch = pytest.importorskip("torch")

    loaded = inference.LoadedModel(
        name="region-model",
        level="region",
        model=SimpleNamespace(tile_size=2),
        transforms=SimpleNamespace(transforms=[]),
        feature_dim=3,
        device=torch.device("cpu"),
    )
    tiling_result = SimpleNamespace(
        target_tile_size_px=4,
        read_tile_size_px=2,
    )
    execution = ExecutionOptions(gpu_batch_preprocessing=False)

    preprocess = inference._build_batch_preprocessor(
        loaded,
        SimpleNamespace(level="region"),
        tiling_result,
        execution=execution,
    )

    batch = torch.full((1, 3, 2, 2), 255, dtype=torch.uint8)
    processed = preprocess(batch)

    assert processed.shape == (1, 4, 3, 2, 2)
    assert processed.dtype == torch.float32
    assert torch.allclose(processed, torch.ones_like(processed))


def test_region_batch_preprocessor_unfolds_then_applies_tile_transforms():
    import slide2vec.inference as inference
    torch = pytest.importorskip("torch")

    class Resize:
        def __init__(self, size):
            self.size = size

    loaded = inference.LoadedModel(
        name="region-model",
        level="region",
        model=SimpleNamespace(tile_size=2),
        transforms=SimpleNamespace(transforms=[Resize(1)]),
        feature_dim=3,
        device=torch.device("cpu"),
    )
    tiling_result = SimpleNamespace(
        target_tile_size_px=4,
        read_tile_size_px=4,
    )
    execution = ExecutionOptions(gpu_batch_preprocessing=False)

    preprocess = inference._build_batch_preprocessor(
        loaded,
        SimpleNamespace(level="region"),
        tiling_result,
        execution=execution,
    )

    quadrant_values = torch.tensor(
        [
            [
                [0, 0, 85, 85],
                [0, 0, 85, 85],
                [170, 170, 255, 255],
                [170, 170, 255, 255],
            ]
        ],
        dtype=torch.uint8,
    )
    batch = quadrant_values.unsqueeze(0).repeat(1, 3, 1, 1)
    processed = preprocess(batch)

    expected = torch.tensor([0.0, 85.0 / 255.0, 170.0 / 255.0, 1.0], dtype=torch.float32)

    assert processed.shape == (1, 4, 3, 1, 1)
    assert torch.allclose(processed[0, :, 0, 0, 0], expected, atol=1e-5)
    assert torch.allclose(processed[0, :, 1, 0, 0], expected, atol=1e-5)
    assert torch.allclose(processed[0, :, 2, 0, 0], expected, atol=1e-5)


def test_build_batch_transform_spec_supports_nested_region_unfolding_transform():
    import slide2vec.inference as inference

    class Compose:
        def __init__(self, transforms):
            self.transforms = transforms

    class RegionUnfolding:
        def __init__(self, tile_size):
            self.tile_size = tile_size

    class Normalize:
        def __init__(self, mean, std):
            self.mean = mean
            self.std = std

    transforms = Compose(
        [
            Compose(
                [
                    RegionUnfolding(8),
                    Normalize((0.5, 0.4, 0.3), (0.2, 0.3, 0.4)),
                ]
            )
        ]
    )

    spec = inference._build_batch_transform_spec(transforms)

    assert spec is not None
    assert spec.region_unfold_tile_size == 8
    assert spec.mean == (0.5, 0.4, 0.3)
    assert spec.std == (0.2, 0.3, 0.4)


def test_region_batch_preprocessor_uses_region_unfolding_from_transform_stack():
    import slide2vec.inference as inference
    torch = pytest.importorskip("torch")

    class Compose:
        def __init__(self, transforms):
            self.transforms = transforms

    class RegionUnfolding:
        def __init__(self, tile_size):
            self.tile_size = tile_size

    loaded = inference.LoadedModel(
        name="region-model",
        level="region",
        model=SimpleNamespace(tile_size=4),
        transforms=Compose([RegionUnfolding(4)]),
        feature_dim=3,
        device=torch.device("cpu"),
    )
    tiling_result = SimpleNamespace(
        target_tile_size_px=8,
        read_tile_size_px=8,
    )

    preprocess = inference._build_batch_preprocessor(
        loaded,
        SimpleNamespace(level="region"),
        tiling_result,
        execution=ExecutionOptions(gpu_batch_preprocessing=False),
    )

    batch = torch.ones((1, 3, 8, 8), dtype=torch.uint8)
    processed = preprocess(batch)

    assert processed.shape == (1, 4, 3, 4, 4)


def test_region_batch_preprocessor_rejects_mismatched_region_unfolding_tile_size():
    import slide2vec.inference as inference
    torch = pytest.importorskip("torch")

    class Compose:
        def __init__(self, transforms):
            self.transforms = transforms

    class RegionUnfolding:
        def __init__(self, tile_size):
            self.tile_size = tile_size

    loaded = inference.LoadedModel(
        name="region-model",
        level="region",
        model=SimpleNamespace(tile_size=2),
        transforms=Compose([RegionUnfolding(4)]),
        feature_dim=3,
        device=torch.device("cpu"),
    )
    tiling_result = SimpleNamespace(
        target_tile_size_px=8,
        read_tile_size_px=8,
    )

    preprocess = inference._build_batch_preprocessor(
        loaded,
        SimpleNamespace(level="region"),
        tiling_result,
        execution=ExecutionOptions(gpu_batch_preprocessing=False),
    )

    with pytest.raises(ValueError, match="tile_size"):
        preprocess(torch.ones((1, 3, 8, 8), dtype=torch.uint8))


def test_serialize_execution_preserves_loader_optimization_fields():
    import slide2vec.inference as inference

    execution = ExecutionOptions(
        output_dir=Path("/tmp/output"),
        batch_size=64,
        num_workers=8,
        num_gpus=2,
        precision="bf16",
        prefetch_factor=7,
        persistent_workers=False,
        gpu_batch_preprocessing=False,
        save_tile_embeddings=True,
        save_latents=True,
    )

    payload = inference._serialize_execution(execution)
    restored = inference.deserialize_execution(payload)

    assert payload["prefetch_factor"] == 7
    assert payload["persistent_workers"] is False
    assert payload["gpu_batch_preprocessing"] is False
    assert payload["precision"] == "bf16"
    assert restored.prefetch_factor == 7
    assert restored.persistent_workers is False
    assert restored.gpu_batch_preprocessing is False
    assert restored.precision == "bf16"


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

        def __call__(self, image):
            return {"embedding": torch.ones((image.shape[0], 3), dtype=torch.float32, device=image.device)}

    fake_dataset_module = types.SimpleNamespace(
        BatchTileCollator=lambda **kwargs: ("collator", kwargs),
        TileIndexDataset=lambda tile_indices: list(tile_indices),
    )
    fake_data_package = types.ModuleType("slide2vec.data")
    fake_data_package.__path__ = []
    fake_data_package.dataset = fake_dataset_module

    monkeypatch.setitem(sys.modules, "slide2vec.data", fake_data_package)
    monkeypatch.setitem(sys.modules, "slide2vec.data.dataset", fake_dataset_module)

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
        target_spacing_um=0.5,
        target_tile_size_px=4,
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
        gpu_batch_preprocessing=True,
    )

    result = inference._compute_tile_embeddings_for_slide(
        loaded,
        SimpleNamespace(level="tile"),
        slide,
        tiling_result,
        preprocessing=PreprocessingConfig(on_the_fly=False),
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

        def __call__(self, image):
            return {"embedding": torch.ones((image.shape[0], 3), dtype=torch.float32, device=image.device)}

    fake_dataset_module = types.SimpleNamespace(
        BatchTileCollator=lambda **kwargs: ("collator", kwargs),
        TileIndexDataset=lambda tile_indices: list(tile_indices),
    )
    fake_data_package = types.ModuleType("slide2vec.data")
    fake_data_package.__path__ = []
    fake_data_package.dataset = fake_dataset_module

    monkeypatch.setitem(sys.modules, "slide2vec.data", fake_data_package)
    monkeypatch.setitem(sys.modules, "slide2vec.data.dataset", fake_dataset_module)
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
        target_spacing_um=0.5,
        target_tile_size_px=4,
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
        preprocessing=PreprocessingConfig(read_tiles_from=Path("/tmp/external-tiles")),
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

    workers, details = inference._resolve_on_the_fly_num_workers(4)

    assert workers == 8
    assert "cpu_count=96" in details
    assert "slurm_cpu_limit=32" in details
    assert "num_cucim_workers=4" in details


def test_compute_tile_embeddings_for_slide_caps_on_the_fly_workers_to_slurm(monkeypatch):
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

        def __call__(self, image):
            return {"embedding": torch.ones((image.shape[0], 3), dtype=torch.float32, device=image.device)}

    class DummyCollator:
        ordered_indices = None

        def __init__(self, **kwargs):
            captured["collator_kwargs"] = kwargs

        def __call__(self, batch_indices):
            tile_indices = torch.as_tensor(batch_indices, dtype=torch.long)
            batch = torch.zeros((len(batch_indices), 3, 4, 4), dtype=torch.uint8)
            return tile_indices, batch, {"worker_batch_ms": 0.0, "reader_open_ms": 0.0, "reader_read_ms": 0.0}

    monkeypatch.setitem(sys.modules, "slide2vec.data.tile_reader", types.SimpleNamespace(OnTheFlyBatchTileCollator=DummyCollator))
    monkeypatch.setattr(torch.utils.data, "DataLoader", DummyLoader)
    monkeypatch.setattr(inference, "_build_batch_preprocessor", lambda *args, **kwargs: lambda batch: batch.float())
    monkeypatch.setattr(inference.os, "cpu_count", lambda: 96)
    monkeypatch.setenv("SLURM_JOB_CPUS_PER_NODE", "32")
    monkeypatch.delenv("SLURM_CPUS_PER_TASK", raising=False)

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
        target_spacing_um=0.5,
        target_tile_size_px=4,
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
        gpu_batch_preprocessing=True,
    )

    result = inference._compute_tile_embeddings_for_slide(
        loaded,
        SimpleNamespace(level="tile"),
        slide,
        tiling_result,
        preprocessing=PreprocessingConfig(on_the_fly=True, backend="cucim", num_cucim_workers=4),
        execution=execution,
    )

    assert result.shape == (2, 3)
    assert captured["kwargs"]["num_workers"] == 8
    assert captured["kwargs"]["persistent_workers"] is True
    assert captured["kwargs"]["prefetch_factor"] == 9


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

        def __call__(self, image):
            return {"embedding": torch.ones((image.shape[0], 3), dtype=torch.float32, device=image.device)}

    class DummyCucimCollator:
        ordered_indices = None

        def __init__(self, **kwargs):
            captured["cucim_collator_kwargs"] = kwargs

        def __call__(self, batch_indices):
            tile_indices = torch.as_tensor(batch_indices, dtype=torch.long)
            batch = torch.zeros((len(batch_indices), 3, 4, 4), dtype=torch.uint8)
            return tile_indices, batch, {"worker_batch_ms": 0.0, "reader_open_ms": 0.0, "reader_read_ms": 0.0}

    fake_dataset_module = types.SimpleNamespace(
        BatchTileCollator=lambda **kwargs: ("collator", kwargs),
        TileIndexDataset=lambda tile_indices: list(tile_indices),
    )
    monkeypatch.setitem(sys.modules, "slide2vec.data.dataset", fake_dataset_module)
    monkeypatch.setitem(sys.modules, "slide2vec.data.tile_reader", types.SimpleNamespace(OnTheFlyBatchTileCollator=DummyCucimCollator))
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
            target_spacing_um=0.5,
            target_tile_size_px=4,
            read_spacing_um=0.5,
            read_tile_size_px=4,
            tile_size_lv0=224,
        ),
        preprocessing=PreprocessingConfig(on_the_fly=True, backend="auto", num_cucim_workers=4),
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

        def __call__(self, image):
            return {"embedding": torch.ones((image.shape[0], 3), dtype=torch.float32, device=image.device)}

    class DummyCollator:
        ordered_indices = None

        def __init__(self, **kwargs):
            captured["wsd_collator_kwargs"] = kwargs

        def __call__(self, batch_indices):
            tile_indices = torch.as_tensor(batch_indices, dtype=torch.long)
            batch = torch.zeros((len(batch_indices), 3, 4, 4), dtype=torch.uint8)
            return tile_indices, batch, {"worker_batch_ms": 0.0, "reader_open_ms": 0.0, "reader_read_ms": 0.0}

    monkeypatch.setitem(sys.modules, "slide2vec.data.tile_reader", types.SimpleNamespace(OnTheFlyBatchTileCollator=DummyCollator))
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
            target_spacing_um=0.5,
            target_tile_size_px=4,
            read_spacing_um=0.5,
            read_tile_size_px=4,
            tile_size_lv0=224,
        ),
        preprocessing=PreprocessingConfig(on_the_fly=True, backend="auto", num_cucim_workers=4),
        execution=ExecutionOptions(batch_size=2, num_workers=8, num_gpus=1),
    )

    assert result.shape == (2, 3)
    assert captured["wsd_collator_kwargs"]["backend"] == "asap"


def test_persist_embedded_slide_records_resolved_backend_when_auto(monkeypatch, tmp_path: Path):
    import slide2vec.inference as inference

    embedded = EmbeddedSlide(
        sample_id="slide-a",
        tile_embeddings=np.zeros((2, 4), dtype=np.float32),
        slide_embedding=None,
        coordinates=np.array([[0, 2], [1, 3]], dtype=np.int64),
        tile_size_lv0=224,
        image_path=Path("/tmp/slide-a.svs"),
        mask_path=None,
    )
    captured = {}

    monkeypatch.setattr(
        inference,
        "_write_tile_embedding_artifact",
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
        preprocessing=PreprocessingConfig(backend="auto"),
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
                target_spacing_um=0.5,
                target_tile_size_px=4,
                read_spacing_um=0.5,
                read_tile_size_px=4,
                tile_size_lv0=224,
                tiles_tar_path=None,
            ),
            preprocessing=PreprocessingConfig(on_the_fly=False),
            execution=ExecutionOptions(batch_size=1, num_workers=0, num_gpus=1),
        )


def test_compute_tile_embeddings_for_slide_uses_batched_loader_for_region_models(monkeypatch):
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
                torch.full((2, 3, 4, 4), 255, dtype=torch.uint8),
            )

        def __len__(self):
            return 1

    class DummyRegionModel:
        tile_size = 2

        def __call__(self, image):
            assert image.ndim == 5
            assert image.shape[1:] == (4, 3, 2, 2)
            return {"embedding": torch.ones((image.shape[0], image.shape[1], 3), dtype=torch.float32, device=image.device)}

    fake_dataset_module = types.SimpleNamespace(
        BatchTileCollator=lambda **kwargs: ("collator", kwargs),
        TileIndexDataset=lambda tile_indices: list(tile_indices),
    )
    fake_data_package = types.ModuleType("slide2vec.data")
    fake_data_package.__path__ = []
    fake_data_package.dataset = fake_dataset_module

    monkeypatch.setitem(sys.modules, "slide2vec.data", fake_data_package)
    monkeypatch.setitem(sys.modules, "slide2vec.data.dataset", fake_dataset_module)
    monkeypatch.setattr(torch.utils.data, "DataLoader", DummyLoader)

    class Normalize:
        def __init__(self, mean, std):
            self.mean = mean
            self.std = std

    loaded = inference.LoadedModel(
        name="region-model",
        level="region",
        model=DummyRegionModel(),
        transforms=SimpleNamespace(transforms=[Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
        feature_dim=3,
        device=torch.device("cpu"),
    )
    slide = make_slide("slide-a")
    tiling_result = SimpleNamespace(
        x=np.array([0, 10]),
        y=np.array([5, 15]),
        target_spacing_um=0.5,
        target_tile_size_px=4,
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
        gpu_batch_preprocessing=False,
    )

    result = inference._compute_tile_embeddings_for_slide(
        loaded,
        SimpleNamespace(level="region"),
        slide,
        tiling_result,
        preprocessing=PreprocessingConfig(on_the_fly=False),
        execution=execution,
    )

    assert result.shape == (2, 4, 3)
    assert captured["kwargs"]["persistent_workers"] is True
    assert captured["kwargs"]["prefetch_factor"] == 9
    assert captured["kwargs"]["collate_fn"] == (
        "collator",
        {
            "tar_path": Path("/tmp/slide-a.tiles.tar"),
            "tiling_result": tiling_result,
        },
    )


def test_scale_coordinates_scales_down():
    from slide2vec.inference import _scale_coordinates

    coords = np.array([[10, 20], [30, 40]])
    # base=0.25, target=0.5 → scale=0.5 → coordinates halved
    result = _scale_coordinates(coords, base_spacing_um=0.25, spacing=0.5)
    np.testing.assert_array_equal(result, [[5, 10], [15, 20]])


def test_scale_coordinates_identity_when_spacings_equal():
    from slide2vec.inference import _scale_coordinates

    coords = np.array([[10, 20], [30, 40]])
    result = _scale_coordinates(coords, base_spacing_um=0.5, spacing=0.5)
    np.testing.assert_array_equal(result, [[10, 20], [30, 40]])
