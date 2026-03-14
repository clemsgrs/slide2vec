import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
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

def test_pipeline_run_uses_distributed_embedding_path_when_num_gpus_is_greater_than_one(
    monkeypatch,
    tmp_path: Path,
):
    import slide2vec.inference as inference

    model = Model.from_pretrained("virchow2")
    slide = inference.SlideRecord(sample_id="slide-a", image_path=Path("/tmp/slide-a.svs"))
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

    slide = inference.SlideRecord(sample_id="slide-a", image_path=Path("/tmp/slide-a.svs"))
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
        Model.from_pretrained("virchow2"),
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

    model = Model.from_pretrained("virchow2", level="slide")
    slide = inference.SlideRecord(sample_id="slide-a", image_path=Path("/tmp/slide-a.svs"))
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

def test_run_pipeline_local_branch_uses_collect_local_pipeline_artifacts(monkeypatch, tmp_path: Path):
    import slide2vec.inference as inference

    slide_record = inference.SlideRecord(sample_id="slide-a", image_path=Path("/tmp/slide-a.svs"))
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

    def fake_collect(*, model, embedded_slides, tiling_results, preprocessing, execution):
        captured["model"] = model
        captured["embedded_slides"] = embedded_slides
        captured["tiling_results"] = tiling_results
        captured["preprocessing"] = preprocessing
        captured["execution"] = execution
        return ["tile-artifact"], ["slide-artifact"]

    monkeypatch.setattr(inference, "_collect_local_pipeline_artifacts", fake_collect)

    result = inference.run_pipeline(
        Model.from_pretrained("virchow2"),
        slides=[slide_record],
        preprocessing=PreprocessingConfig(),
        execution=ExecutionOptions(output_dir=tmp_path),
    )

    assert captured["embedded_slides"] == [embedded]
    assert captured["tiling_results"] == [tiling_result]
    assert result.tile_artifacts == ["tile-artifact"]
    assert result.slide_artifacts == ["slide-artifact"]

def test_embed_single_slide_distributed_uses_shared_slide_aggregation_helper(monkeypatch, tmp_path: Path):
    from contextlib import contextmanager

    import slide2vec.inference as inference

    slide = inference.SlideRecord(sample_id="slide-a", image_path=Path("/tmp/slide-a.svs"))
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

    slide = inference.SlideRecord(sample_id="slide-a", image_path=Path("/tmp/slide-a.svs"))
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
        model=Model.from_pretrained("virchow2"),
        slide_records=[slide],
        tiling_results=[tiling_result],
        preprocessing=PreprocessingConfig(),
        execution=ExecutionOptions(output_dir=Path("/tmp"), num_gpus=1),
        work_dir=Path("/tmp"),
    )

    assert result == expected

def test_select_embedding_path_uses_single_slide_distributed_when_one_slide(monkeypatch, tmp_path: Path):
    import slide2vec.inference as inference

    slide = inference.SlideRecord(sample_id="slide-a", image_path=Path("/tmp/slide-a.svs"))
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
        model=Model.from_pretrained("virchow2"),
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
        inference.SlideRecord(sample_id="slide-a", image_path=Path("/tmp/slide-a.svs")),
        inference.SlideRecord(sample_id="slide-b", image_path=Path("/tmp/slide-b.svs")),
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
        model=Model.from_pretrained("virchow2"),
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

    slide = inference.SlideRecord(sample_id="slide-a", image_path=Path("/tmp/slide-a.svs"))
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

    slide_record = inference.SlideRecord(sample_id="slide-a", image_path=Path("/tmp/slide-a.svs"))
    tiling_result = SimpleNamespace(
        x=np.array([0, 1]),
        y=np.array([2, 3]),
        tile_size_lv0=224,
        tiles_npz_path=Path("/tmp/slide-a.tiles.npz"),
        tiles_meta_path=Path("/tmp/slide-a.tiles.meta.json"),
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
    monkeypatch.setattr(
        inference,
        "_compute_embedded_slides",
        lambda model, slide_records, tiling_results, preprocessing, execution: [embedded],
    )

    model = Model.from_pretrained("prism", level="slide")
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

def test_slide_level_pipeline_skips_tile_artifacts_when_save_tile_embeddings_is_false(monkeypatch, tmp_path: Path):
    import slide2vec.inference as inference

    slide_record = inference.SlideRecord(sample_id="slide-a", image_path=Path("/tmp/slide-a.svs"))
    tiling_result = SimpleNamespace(
        x=np.array([0, 1]),
        y=np.array([2, 3]),
        tile_size_lv0=224,
        tiles_npz_path=Path("/tmp/slide-a.tiles.npz"),
        tiles_meta_path=Path("/tmp/slide-a.tiles.meta.json"),
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
    monkeypatch.setattr(
        inference,
        "_compute_embedded_slides",
        lambda model, slide_records, tiling_results, preprocessing, execution: [embedded],
    )

    result = inference.run_pipeline(
        Model.from_pretrained("prism", level="slide"),
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

    slide_record = inference.SlideRecord(sample_id="slide-a", image_path=Path("/tmp/slide-a.svs"))
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
        Model.from_pretrained("virchow2"),
        [slide_record],
        preprocessing=PreprocessingConfig(),
        execution=ExecutionOptions(output_dir=tmp_path, output_format="npz", num_gpus=2),
    )

    assert result == [embedded]
    assert captured["single"]["slide"] == slide_record

def test_direct_embed_slides_uses_balanced_slide_sharding_for_multiple_slides(monkeypatch, tmp_path: Path):
    import slide2vec.inference as inference

    slides = [
        inference.SlideRecord(sample_id="slide-a", image_path=Path("/tmp/slide-a.svs")),
        inference.SlideRecord(sample_id="slide-b", image_path=Path("/tmp/slide-b.svs")),
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
        Model.from_pretrained("virchow2"),
        slides,
        preprocessing=PreprocessingConfig(),
        execution=ExecutionOptions(output_dir=tmp_path, output_format="npz", num_gpus=2),
    )

    assert result == expected
    assert captured["multi"]["slide_records"] == slides

def test_assign_slides_to_ranks_balances_by_tile_count():
    import slide2vec.inference as inference

    slides = [
        inference.SlideRecord(sample_id="slide-a", image_path=Path("/tmp/slide-a.svs")),
        inference.SlideRecord(sample_id="slide-b", image_path=Path("/tmp/slide-b.svs")),
        inference.SlideRecord(sample_id="slide-c", image_path=Path("/tmp/slide-c.svs")),
        inference.SlideRecord(sample_id="slide-d", image_path=Path("/tmp/slide-d.svs")),
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
