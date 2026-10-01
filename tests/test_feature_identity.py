"""The feature identity pooled sidecars record under ``compatibility``, and its resume check.

Ways the transform record can go wrong, each covered below:

* torchvision v1 and v2 hold the same size differently (``int``, ``list``, ``tuple``);
* interpolation arrives as a torchvision enum, a PIL enum or a PIL integer;
* a tensor-backed ``Normalize`` (timm) leaks float32 noise into mean/std;
* steps hide inside a nested ``Compose`` or behind a Hugging Face processor;
* a ``repr()`` string leaks into the record and changes with the torchvision release;
* the transform exposes no step to read.

Ways the pooled sidecars and their resume check can go wrong, each covered below:

* a writer omits the identity: tile, slide, hierarchical, zero-tile, ``embed_tiles``,
  ``aggregate_tiles``, the patient pipeline, and the multi-GPU parent that never loads
  the encoder its workers encoded with;
* a resume reuses artifacts from a different encoder, output variant, precision, stored
  dtype, encoder input size or transform;
* a resume raises on, or recomputes, artifacts written before a field existed, or warns
  once per sidecar instead of once per run;
* a resume loads the encoder although no completed sidecar records a transform, or holds
  GPU memory in the multi-GPU parent;
* the local and the distributed path check different things.

The image and dense paths are covered next to their stages (``test_image_stage.py``,
``test_dense_shard.py``, ``test_dense_stage.py``, ``test_dense_image_*.py``).
"""

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torchvision.transforms import v2

from slide2vec.api import EmbeddedSlide, ExecutionOptions, PreprocessingConfig
from slide2vec.artifacts import load_metadata
from slide2vec.encoders import encoder_registry
from slide2vec.runtime.embedding_persist import persist_embedded_slide
from slide2vec.runtime.feature_identity import transform_record

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def test_transform_record_holds_normalize_resize_and_center_crop_as_json_data():
    transform = v2.Compose([
        v2.ToImage(),
        v2.Resize(256, interpolation=v2.InterpolationMode.BICUBIC, antialias=True),
        v2.CenterCrop(224),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    record = transform_record(transform)

    assert record == {
        "normalize": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
        "resize": {"size": [256], "interpolation": "bicubic"},
        "center_crop": {"size": [224, 224]},
    }
    assert json.loads(json.dumps(record)) == record


def test_transform_record_is_the_same_for_torchvision_v1_and_v2():
    """The two APIs store sizes and repr their steps differently; the record must not."""
    from torchvision import transforms

    def recipe(api):
        return api.Compose([
            api.Resize(256, interpolation=api.InterpolationMode.BICUBIC),
            api.CenterCrop(224),
            api.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    assert repr(recipe(transforms)) != repr(recipe(v2))
    assert transform_record(recipe(transforms)) == transform_record(recipe(v2))


def test_transform_record_of_a_timm_transform_matches_the_float_backed_values():
    """timm builds a v1 ``Compose`` whose Normalize holds float32 tensors."""
    from timm.data import create_transform

    transform = create_transform(
        input_size=(3, 224, 224), crop_pct=0.9, interpolation="bicubic",
        mean=IMAGENET_MEAN, std=IMAGENET_STD,
    )

    assert transform_record(transform) == {
        "normalize": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
        "resize": {"size": [248], "interpolation": "bicubic"},
        "center_crop": {"size": [224, 224]},
    }


def test_transform_record_reads_a_hugging_face_image_processor():
    """Covers both size layouts and both resample spellings (PIL enum and integer)."""
    from PIL import Image
    from transformers import BitImageProcessor, ViTImageProcessor

    fixed = ViTImageProcessor(
        size={"height": 224, "width": 224}, resample=Image.Resampling.BILINEAR,
        image_mean=IMAGENET_MEAN, image_std=IMAGENET_STD,
    )
    shortest_edge = BitImageProcessor(
        size={"shortest_edge": 224}, crop_size={"height": 224, "width": 224}, resample=3,
        image_mean=IMAGENET_MEAN, image_std=IMAGENET_STD,
    )

    assert transform_record(fixed) == {
        "normalize": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
        "resize": {"size": [224, 224], "interpolation": "bilinear"},
        "center_crop": None,
    }
    assert transform_record(shortest_edge) == {
        "normalize": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
        "resize": {"size": [224], "interpolation": "bicubic"},
        "center_crop": {"size": [224, 224]},
    }


def test_transform_record_of_an_unsupported_processor_size_layout_holds_no_size():
    from transformers import ViTImageProcessor

    longest_edge = ViTImageProcessor(
        size={"longest_edge": 224}, resample=2, do_normalize=False
    )

    assert transform_record(longest_edge) == {
        "normalize": None,
        "resize": {"size": None, "interpolation": "bilinear"},
        "center_crop": None,
    }


def _offline_tile_encoder(name: str, monkeypatch):
    """A registered tile encoder with its preprocessing config supplied offline (no weights)."""
    from torchvision import transforms
    from transformers import CLIPImageProcessor, ViTImageProcessor

    import slide2vec.encoders.models.isight as isight

    encoder_cls = encoder_registry.require(name)
    encoder = encoder_cls.__new__(encoder_cls)
    # timm-backed encoders read the checkpoint's pretrained_cfg; Waiv reads the model config.
    encoder._model = SimpleNamespace(
        pretrained_cfg=dict(
            input_size=(3, 224, 224), crop_pct=0.9, interpolation="bicubic",
            mean=IMAGENET_MEAN, std=IMAGENET_STD,
        ),
        config=SimpleNamespace(pixel_mean=IMAGENET_MEAN, pixel_std=IMAGENET_STD),
    )
    # CONCH keeps the third-party transform its package returns (a v1 Compose).
    encoder._transform = transforms.Compose([
        transforms.Resize(448, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(448),
        lambda image: image.convert("RGB"),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    # Phikon and iSight preprocess through Hugging Face processors.
    encoder._processor = ViTImageProcessor(
        size={"height": 224, "width": 224}, image_mean=IMAGENET_MEAN, image_std=IMAGENET_STD,
    )
    clip_processor = SimpleNamespace(
        image_processor=CLIPImageProcessor(
            size={"shortest_edge": 336}, crop_size={"height": 336, "width": 336},
            image_mean=IMAGENET_MEAN, image_std=IMAGENET_STD,
        )
    )
    monkeypatch.setattr(
        isight, "AutoProcessor", SimpleNamespace(from_pretrained=lambda _name: clip_processor)
    )
    return encoder


TILE_ENCODER_NAMES = sorted(
    name for name in encoder_registry.names() if encoder_registry.info(name)["level"] == "tile"
)


@pytest.mark.parametrize("name", TILE_ENCODER_NAMES)
def test_transform_record_reads_every_registered_tile_encoder(name, monkeypatch):
    """Neither regime's transform is opaque: the shipped recipe resizes, both normalize."""
    encoder = _offline_tile_encoder(name, monkeypatch)

    shipped = transform_record(encoder.get_transform())
    declared = transform_record(encoder.get_normalization_transform())

    assert shipped["normalize"] is not None
    assert shipped["resize"] is not None
    assert declared["normalize"] is not None
    assert declared["resize"] is None
    assert declared["center_crop"] is None


def test_transform_record_of_an_opaque_callable_holds_no_step():
    assert transform_record(lambda image: image) == {
        "normalize": None,
        "resize": None,
        "center_crop": None,
    }


def test_transform_record_reads_steps_inside_a_nested_compose():
    from torchvision import transforms

    transform = transforms.Compose([
        transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()]),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ])

    assert transform_record(transform) == {
        "normalize": {"mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]},
        "resize": {"size": [224, 224], "interpolation": "bilinear"},
        "center_crop": None,
    }


# --- Pooled sidecars -------------------------------------------------------------------

NORMALIZE_ONLY = {
    "normalize": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
    "resize": None,
    "center_crop": None,
}


def _tiling_result(**overrides):
    fields = dict(
        x=np.array([0], dtype=np.int64),
        y=np.array([1], dtype=np.int64),
        tile_size_lv0=224,
        num_tiles=1,
        backend="asap",
        annotation=None,
        requested_tile_size_px=224,
        read_tile_size_px=224,
        coordinates_npz_path=Path("/tmp/c.npz"),
        coordinates_meta_path=Path("/tmp/c.meta.json"),
        tiles_tar_path=None,
    )
    fields.update(overrides)
    return SimpleNamespace(**fields)


def _embedded_slide(*, tile_embeddings, slide_embedding=None) -> EmbeddedSlide:
    return EmbeddedSlide(
        sample_id="slide-a",
        tile_embeddings=tile_embeddings,
        slide_embedding=slide_embedding,
        x=np.array([0], dtype=np.int64),
        y=np.array([1], dtype=np.int64),
        tile_size_lv0=224,
        image_path=Path("/tmp/slide-a.svs"),
        transform=NORMALIZE_ONLY,
    )


def test_pooled_tile_sidecar_records_the_feature_identity(tmp_path):
    tile_artifact, _ = persist_embedded_slide(
        SimpleNamespace(name="virchow2", level="tile", _output_variant="cls"),
        _embedded_slide(tile_embeddings=np.zeros((1, 4), dtype=np.float32)),
        _tiling_result(),
        preprocessing=PreprocessingConfig(requested_spacing_um=0.5, requested_tile_size_px=224),
        execution=ExecutionOptions(output_dir=tmp_path, precision="bf16"),
    )

    assert load_metadata(tile_artifact.metadata_path)["compatibility"] == {
        "encoder_name": "virchow2",
        "output_variant": "cls",
        "precision": "bf16",
        "feature_dtype": "fp32",
        "requested_tile_size_px": 224,
        "encoder_input_size_px": 224,
        "transform": NORMALIZE_ONLY,
    }


def test_zero_tile_sidecar_records_the_feature_identity_without_a_transform(tmp_path):
    """No tile was encoded and no encoder is loaded, so there is no transform to record."""
    from slide2vec.runtime import process_list

    slide = SimpleNamespace(sample_id="slide-z", image_path=Path("/tmp/slide-z.svs"), mask_path=None)
    process_list.write_zero_tile_embedding_sidecars(
        [(slide, _tiling_result(x=np.array([]), y=np.array([]), num_tiles=0))],
        model=SimpleNamespace(name="virchow2", level="tile", _output_variant=None),
        preprocessing=PreprocessingConfig(requested_spacing_um=0.5, requested_tile_size_px=224),
        execution=ExecutionOptions(output_dir=tmp_path, precision="fp16"),
    )

    sidecar = load_metadata(tmp_path / "tile_embeddings" / "slide-z.meta.json")
    assert sidecar["compatibility"] == {
        "encoder_name": "virchow2",
        "output_variant": "cls_patch_mean",
        "precision": "fp16",
        "feature_dtype": "fp16",
        "requested_tile_size_px": 224,
        "encoder_input_size_px": 224,
    }


def _normalization_transform(mean=IMAGENET_MEAN):
    return v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=mean, std=IMAGENET_STD),
    ])


class _FakeModel:
    """``Model`` stand-in whose backend carries a real transform and counts its loads."""

    def __init__(self, name="virchow2", *, level="tile", output_variant=None, mean=IMAGENET_MEAN):
        self.name = name
        self.level = level
        self._output_variant = output_variant
        self._requested_device = "cpu"
        self.allow_non_recommended_settings = False
        self.loads = 0
        self._loaded = SimpleNamespace(
            feature_dim=2, device="cpu", model=SimpleNamespace(),
            transforms=_normalization_transform(mean),
        )

    def _declare_encoder_input(self, preprocessing, *, emit_run_info):
        return None

    def _load_backend(self):
        self.loads += 1
        return self._loaded


PREPROCESSING = PreprocessingConfig(requested_spacing_um=0.5, requested_tile_size_px=224)


def _run_pipeline(monkeypatch, tmp_path, model, *, preprocessing=PREPROCESSING, **execution):
    """Run the local pipeline over one already-tiled slide; return the samples it embedded."""
    import slide2vec.inference as inference
    from slide2vec.runtime import embedding_pipeline, tiling_pipeline

    slide = SimpleNamespace(
        sample_id="slide-a", image_path=Path("/tmp/slide-a.svs"), mask_path=None,
        spacing_at_level_0=None,
    )
    process_list_path = tmp_path / "process_list.csv"
    if not process_list_path.exists():
        process_list_path.write_text(
            "sample_id,annotation,image_path,mask_path,requested_backend,backend,"
            "spacing_at_level_0,tiling_status,num_tiles,coordinates_npz_path,"
            "coordinates_meta_path,feature_status,error,traceback\n"
            "slide-a,tissue,/tmp/slide-a.svs,,auto,asap,,success,1,/tmp/c.npz,/tmp/c.meta.json,tbp,,\n",
            encoding="utf-8",
        )
    monkeypatch.setattr(
        tiling_pipeline, "prepare_tiled_slides",
        lambda *args, **kwargs: ([slide], [_tiling_result()], process_list_path),
    )
    embedded: list[str] = []

    def fake_compute_tile_embeddings(_loaded, _model, slide, _tiling_result, **_kwargs):
        embedded.append(slide.sample_id)
        return np.array([[1.0, 2.0]], dtype=np.float32)

    monkeypatch.setattr(
        embedding_pipeline, "compute_tile_embeddings_for_slide", fake_compute_tile_embeddings
    )
    execution.setdefault("precision", "fp16")
    inference.run_pipeline(
        model,
        slides=[slide],
        preprocessing=preprocessing,
        execution=ExecutionOptions(output_dir=tmp_path, output_format="npz", num_gpus=1, **execution),
    )
    return embedded


def test_run_pipeline_records_the_transform_the_loaded_encoder_applies(monkeypatch, tmp_path):
    _run_pipeline(monkeypatch, tmp_path, _FakeModel())

    sidecar = load_metadata(tmp_path / "tile_embeddings" / "slide-a.meta.json")
    assert sidecar["compatibility"] == {
        "encoder_name": "virchow2",
        "output_variant": "cls_patch_mean",
        "precision": "fp16",
        "feature_dtype": "fp16",
        "requested_tile_size_px": 224,
        "encoder_input_size_px": 224,
        "transform": NORMALIZE_ONLY,
    }


RESUME = replace(PREPROCESSING, resume=True)


def test_pooled_resume_with_the_same_recipe_skips_the_completed_slide(monkeypatch, tmp_path):
    _run_pipeline(monkeypatch, tmp_path, _FakeModel())

    embedded = _run_pipeline(monkeypatch, tmp_path, _FakeModel(), preprocessing=RESUME)

    assert embedded == []


def test_pooled_resume_refuses_artifacts_from_a_different_encoder(monkeypatch, tmp_path):
    _run_pipeline(monkeypatch, tmp_path, _FakeModel("virchow2"))

    with pytest.raises(ValueError) as error:
        _run_pipeline(monkeypatch, tmp_path, _FakeModel("uni"), preprocessing=RESUME)

    assert str(error.value) == (
        "Cannot resume 'slide-a': the existing tile embeddings at "
        f"{tmp_path / 'tile_embeddings' / 'slide-a.npz'} were computed with a different "
        "feature identity: encoder_name (recorded 'virchow2', requested 'uni'); "
        "output_variant (recorded 'cls_patch_mean', requested 'default'). Re-run into a new "
        "output_dir, delete the stale artifacts, or request the recorded values."
    )


@pytest.mark.parametrize(
    ("first", "second", "differences"),
    [
        pytest.param(
            dict(model=dict(output_variant="cls")),
            dict(model=dict(output_variant="cls_patch_mean")),
            "output_variant (recorded 'cls', requested 'cls_patch_mean')",
            id="output-variant",
        ),
        pytest.param(
            dict(precision="fp32"),
            dict(precision="bf16"),
            "precision (recorded 'fp32', requested 'bf16')",
            id="precision",
        ),
        pytest.param(
            dict(precision="fp16"),
            dict(precision="fp16", output_dtype="fp32"),
            "feature_dtype (recorded 'fp16', requested 'fp32')",
            id="output-dtype",
        ),
        pytest.param(
            dict(),
            dict(tile_size=448),
            "requested_tile_size_px (recorded 224, requested 448); "
            "encoder_input_size_px (recorded 224, requested 448)",
            id="encoder-input-size",
        ),
        pytest.param(
            dict(),
            dict(model=dict(mean=(0.5, 0.5, 0.5))),
            "transform.normalize.mean (recorded [0.485, 0.456, 0.406], requested [0.5, 0.5, 0.5])",
            id="transform",
        ),
    ],
)
def test_pooled_resume_refuses_artifacts_with_a_different_recipe(
    monkeypatch, tmp_path, first, second, differences
):
    def run(*, model=None, tile_size=224, resume, **execution):
        preprocessing = replace(PREPROCESSING, requested_tile_size_px=tile_size, resume=resume)
        return _run_pipeline(
            monkeypatch, tmp_path, _FakeModel(**(model or {})),
            preprocessing=preprocessing, **execution,
        )

    run(resume=False, **first)

    with pytest.raises(ValueError) as error:
        run(resume=True, **second)

    assert str(error.value) == (
        "Cannot resume 'slide-a': the existing tile embeddings at "
        f"{tmp_path / 'tile_embeddings' / 'slide-a.npz'} were computed with a different "
        f"feature identity: {differences}. Re-run into a new output_dir, delete the stale "
        "artifacts, or request the recorded values."
    )


def _write_completed_tile_run(tmp_path, sidecars: dict[str, dict]) -> Path:
    """Completed tile artifacts as an earlier run left them: one sidecar dict per sample."""
    from slide2vec.artifacts import write_tile_embeddings

    process_list_path = tmp_path / "process_list.csv"
    process_list_path.write_text(
        "sample_id,annotation,image_path,mask_path,requested_backend,backend,"
        "spacing_at_level_0,tiling_status,num_tiles,coordinates_npz_path,"
        "coordinates_meta_path,feature_status,error,traceback\n"
        + "".join(
            f"{sample_id},tissue,/tmp/{sample_id}.svs,,auto,asap,,success,1,/tmp/c.npz,"
            "/tmp/c.meta.json,success,,\n"
            for sample_id in sidecars
        ),
        encoding="utf-8",
    )
    for sample_id, metadata in sidecars.items():
        write_tile_embeddings(
            sample_id, np.array([[1.0, 2.0]], dtype=np.float32),
            output_dir=tmp_path, output_format="npz", metadata=metadata,
        )
    return process_list_path


def test_pooled_resume_accepts_sidecars_that_lack_fields_with_one_warning(
    monkeypatch, tmp_path, caplog
):
    """Artifacts written before the identity existed are reused, and no encoder is loaded."""
    _write_completed_tile_run(
        tmp_path, {"slide-a": {"encoder_name": "uni", "requested_tile_size_px": 224}}
    )
    model = _FakeModel("virchow2")

    with caplog.at_level("WARNING", logger="slide2vec"):
        embedded = _run_pipeline(monkeypatch, tmp_path, model, preprocessing=RESUME)

    assert embedded == []
    assert model.loads == 0
    assert [record.getMessage() for record in caplog.records] == [
        "Resuming over 1 completed sidecar(s) that do not record encoder_input_size_px, "
        "encoder_name, feature_dtype, output_variant, precision, transform; cannot verify "
        "those fields against this run."
    ]


def test_pooled_resume_refuses_a_different_recorded_encoder_input_size(monkeypatch, tmp_path):
    """The tile size and the encoder input size are compared independently."""
    _write_completed_tile_run(
        tmp_path, {"slide-a": {"compatibility": {"encoder_input_size_px": 256}}}
    )

    with pytest.raises(ValueError) as error:
        _run_pipeline(monkeypatch, tmp_path, _FakeModel("virchow2"), preprocessing=RESUME)

    assert str(error.value) == (
        "Cannot resume 'slide-a': the existing tile embeddings at "
        f"{tmp_path / 'tile_embeddings' / 'slide-a.npz'} were computed with a different "
        "feature identity: encoder_input_size_px (recorded 256, requested 224). Re-run "
        "into a new output_dir, delete the stale artifacts, or request the recorded values."
    )


def test_pooled_resume_loads_the_encoder_only_to_verify_a_recorded_transform(
    monkeypatch, tmp_path
):
    _write_completed_tile_run(
        tmp_path, {"slide-a": {"compatibility": {"transform": NORMALIZE_ONLY}}}
    )
    model = _FakeModel("virchow2")

    embedded = _run_pipeline(monkeypatch, tmp_path, model, preprocessing=RESUME)

    assert embedded == []
    assert model.loads == 1


def _collect_distributed(monkeypatch, tmp_path, model, *, process_list_path):
    """Run the multi-GPU parent stage with the torchrun launch captured, not executed."""
    from slide2vec.runtime import artifacts_collect

    launched: list[list[str]] = []
    monkeypatch.setattr(
        artifacts_collect, "run_distributed_embedding_stage",
        lambda **kwargs: launched.append([s.sample_id for s in kwargs["successful_slides"]]),
    )
    monkeypatch.setattr(
        artifacts_collect, "update_process_list_after_embedding", lambda *args, **kwargs: None
    )
    slide = SimpleNamespace(sample_id="slide-a", image_path=Path("/tmp/slide-a.svs"), mask_path=None)
    artifacts_collect.collect_distributed_pipeline_artifacts(
        model=model,
        successful_slides=[slide],
        process_list_path=process_list_path,
        preprocessing=RESUME,
        execution=ExecutionOptions(
            output_dir=tmp_path, output_format="npz", num_gpus=2, precision="fp16"
        ),
        output_dir=tmp_path,
    )
    return launched


def test_distributed_pooled_resume_applies_the_same_check(monkeypatch, tmp_path):
    process_list_path = _write_completed_tile_run(
        tmp_path, {"slide-a": {"compatibility": {"encoder_name": "uni"}}}
    )

    with pytest.raises(ValueError) as error:
        _collect_distributed(
            monkeypatch, tmp_path, _FakeModel("virchow2"), process_list_path=process_list_path
        )

    assert str(error.value) == (
        "Cannot resume 'slide-a': the existing tile embeddings at "
        f"{tmp_path / 'tile_embeddings' / 'slide-a.npz'} were computed with a different "
        "feature identity: encoder_name (recorded 'uni', requested 'virchow2'). Re-run into "
        "a new output_dir, delete the stale artifacts, or request the recorded values."
    )


def test_distributed_pooled_resume_verifies_the_transform_on_a_cpu_copy(monkeypatch, tmp_path):
    """The multi-GPU parent never encodes, so it must not hold GPU memory for the check."""
    import slide2vec.inference as inference
    from slide2vec.api import Model

    process_list_path = _write_completed_tile_run(
        tmp_path, {"slide-a": {"compatibility": {"transform": NORMALIZE_ONLY}}}
    )
    load_devices: list[str] = []

    def fake_load_model(*, name, encoder_input, device, output_variant, allow_non_recommended_settings):
        load_devices.append(device)
        return _FakeModel(name)._loaded

    monkeypatch.setattr(inference, "load_model", fake_load_model)
    model = Model.from_preset("virchow2", device="cuda")

    launched = _collect_distributed(monkeypatch, tmp_path, model, process_list_path=process_list_path)

    assert launched == [[]]
    assert load_devices == ["cpu"]
    assert model._backend is None


def test_embed_tiles_sidecar_records_the_feature_identity(monkeypatch, tmp_path):
    import slide2vec.inference as inference
    from slide2vec.runtime import embedding_pipeline

    monkeypatch.setattr(
        embedding_pipeline, "compute_tile_embeddings_for_slide",
        lambda *args, **kwargs: np.array([[1.0, 2.0]], dtype=np.float32),
    )
    slide = SimpleNamespace(
        sample_id="slide-a", image_path=Path("/tmp/slide-a.svs"), mask_path=None,
        spacing_at_level_0=None,
    )

    [artifact] = inference.embed_tiles(
        _FakeModel("virchow2"),
        [slide],
        [_tiling_result()],
        execution=ExecutionOptions(output_dir=tmp_path, precision="fp16"),
        preprocessing=PREPROCESSING,
    )

    assert load_metadata(artifact.metadata_path)["compatibility"] == {
        "encoder_name": "virchow2",
        "output_variant": "cls_patch_mean",
        "precision": "fp16",
        "feature_dtype": "fp16",
        "requested_tile_size_px": 224,
        "encoder_input_size_px": 224,
        "transform": NORMALIZE_ONLY,
    }


def test_aggregate_tiles_slide_sidecar_carries_the_tile_artifact_recipe(monkeypatch, tmp_path):
    """Aggregation encodes no tile: the geometry and transform recorded with the tiles apply."""
    import slide2vec.inference as inference
    from slide2vec.runtime import slide_encode, tiling

    monkeypatch.setattr(tiling, "load_tiling_result_from_paths", lambda *_args: _tiling_result())
    monkeypatch.setattr(inference, "load_array", lambda _path: torch.ones((1, 4)))
    monkeypatch.setattr(
        slide_encode, "encode_slide_from_tiles", lambda *args, **kwargs: torch.ones(4)
    )
    model = SimpleNamespace(
        name="prism", level="slide", _output_variant=None,
        _load_backend_without_transform=lambda: SimpleNamespace(device="cpu"),
    )
    tile_artifact = SimpleNamespace(
        sample_id="slide-a",
        path=tmp_path / "tile_embeddings" / "slide-a.pt",
        metadata={
            "coordinates_npz_path": "/tmp/c.npz",
            "coordinates_meta_path": "/tmp/c.meta.json",
            "image_path": "/tmp/slide-a.svs",
            "compatibility": {
                "encoder_name": "virchow",
                "requested_tile_size_px": 224,
                "encoder_input_size_px": 224,
                "transform": NORMALIZE_ONLY,
            },
        },
    )

    [slide_artifact] = inference.aggregate_tiles(
        model, [tile_artifact], execution=ExecutionOptions(output_dir=tmp_path, precision="fp16")
    )

    assert load_metadata(slide_artifact.metadata_path)["compatibility"] == {
        "encoder_name": "prism",
        "output_variant": "default",
        "precision": "fp16",
        "feature_dtype": "fp16",
        "tile_encoder": "virchow",
        "tile_encoder_output_variant": "cls_patch_mean",
        "requested_tile_size_px": 224,
        "encoder_input_size_px": 224,
        "transform": NORMALIZE_ONLY,
    }


def test_patient_pipeline_tile_and_slide_sidecars_record_the_feature_identity(monkeypatch, tmp_path):
    from slide2vec.runtime import patient_pipeline

    monkeypatch.setattr(
        patient_pipeline, "compute_tile_embeddings_for_slide",
        lambda *args, **kwargs: torch.ones((1, 4)),
    )
    monkeypatch.setattr(
        patient_pipeline, "encode_slide_from_tiles", lambda *args, **kwargs: torch.ones(4)
    )
    model = _FakeModel("moozy", level="patient")
    model._loaded.model = SimpleNamespace(encode_patient=lambda stacked: stacked.mean(dim=0))
    model._loaded.device = torch.device("cpu")
    slide = SimpleNamespace(sample_id="slide-a", image_path=Path("/tmp/slide-a.svs"), mask_path=None)

    tile_artifacts, slide_artifacts, _ = patient_pipeline.run_patient_pipeline(
        model,
        embeddable_slides=[slide],
        embeddable_tiling_results=[_tiling_result()],
        patient_id_map={"slide-a": "patient-a"},
        preprocessing=PREPROCESSING,
        execution=ExecutionOptions(
            output_dir=tmp_path, precision="fp16",
            save_tile_embeddings=True, save_slide_embeddings=True,
        ),
        output_dir=tmp_path,
    )

    expected = {
        "encoder_name": "moozy",
        "output_variant": "default",
        "precision": "fp16",
        "feature_dtype": "fp16",
        "tile_encoder": "lunit",
        "tile_encoder_output_variant": "default",
        "requested_tile_size_px": 224,
        "encoder_input_size_px": 224,
        "transform": NORMALIZE_ONLY,
    }
    assert load_metadata(tile_artifacts[0].metadata_path)["compatibility"] == expected
    assert load_metadata(slide_artifacts[0].metadata_path)["compatibility"] == expected


def _run_direct_embed_worker(monkeypatch, tmp_path, *, strategy: str, **request):
    """Run one rank of the in-memory multi-GPU worker on CPU; return its coordination dir."""
    import slide2vec.distributed as distributed
    import slide2vec.runtime.serialization as serialization
    from slide2vec.api import Model
    from slide2vec.distributed import direct_embed_worker
    from slide2vec.runtime import embedding_pipeline, manifest

    coordination_dir = tmp_path / "coordination"
    coordination_dir.mkdir()
    request_path = tmp_path / "request.json"
    request_path.write_text(
        json.dumps(
            {
                "model": {"name": "virchow2", "allow_non_recommended_settings": False},
                "preprocessing": {},
                "execution": {},
                "coordination_dir": str(coordination_dir),
                "strategy": strategy,
                **request,
            }
        ),
        encoding="utf-8",
    )
    slide = SimpleNamespace(sample_id="slide-a", image_path=Path("/tmp/slide-a.svs"), mask_path=None)
    monkeypatch.setattr(distributed, "enable", lambda overwrite=True: None)
    monkeypatch.setattr(distributed, "get_local_rank", lambda: 0)
    monkeypatch.setattr(distributed, "get_global_rank", lambda: 0)
    monkeypatch.setattr(distributed, "get_global_size", lambda: 1)
    monkeypatch.setattr(Model, "from_preset", lambda *args, **kwargs: _FakeModel("virchow2"))
    monkeypatch.setattr(serialization, "deserialize_preprocessing", lambda payload: PREPROCESSING)
    monkeypatch.setattr(
        serialization, "deserialize_execution", lambda payload: ExecutionOptions(output_dir=tmp_path)
    )
    monkeypatch.setattr(
        manifest, "load_successful_tiled_slides", lambda output_dir: ([slide], [_tiling_result()])
    )
    monkeypatch.setattr(
        embedding_pipeline, "compute_tile_embeddings_for_slide",
        lambda *args, **kwargs: torch.ones((1, 2)),
    )
    assert direct_embed_worker.main(
        ["--output-dir", str(tmp_path), "--request-path", str(request_path)]
    ) == 0
    return coordination_dir


def test_tile_shard_worker_hands_the_transform_to_the_parent(monkeypatch, tmp_path):
    """The multi-GPU parent writes the sidecar but never loads the encoder that encoded."""
    from slide2vec.runtime.distributed import load_tile_embedding_shards

    coordination_dir = _run_direct_embed_worker(
        monkeypatch, tmp_path, strategy="tile_shard", work_unit="slide-a"
    )

    [shard] = load_tile_embedding_shards(coordination_dir, "slide-a")
    assert shard["transform"] == NORMALIZE_ONLY


def test_slide_shard_worker_hands_the_transform_to_the_parent(monkeypatch, tmp_path):
    from slide2vec.runtime.distributed import load_embedded_slide_payload

    coordination_dir = _run_direct_embed_worker(
        monkeypatch, tmp_path, strategy="slide_shard", assignments={"0": ["slide-a"]}
    )

    assert load_embedded_slide_payload(coordination_dir, "slide-a")["transform"] == NORMALIZE_ONLY


@pytest.mark.parametrize("num_slides", [1, 2], ids=["tile-shards", "slide-shards"])
def test_distributed_embed_slides_parent_receives_the_worker_transform(
    monkeypatch, tmp_path, num_slides
):
    import slide2vec.inference as inference
    from slide2vec.runtime import distributed_stage

    monkeypatch.setattr(
        distributed_stage, "run_distributed_direct_embedding_stage", lambda *args, **kwargs: None
    )
    payload = {
        "tile_index": torch.tensor([0]),
        "tile_embeddings": torch.ones((1, 2)),
        "transform": NORMALIZE_ONLY,
    }
    monkeypatch.setattr(distributed_stage, "load_tile_embedding_shards", lambda *args: [payload])
    monkeypatch.setattr(distributed_stage, "load_embedded_slide_payload", lambda *args: payload)
    slides = [
        SimpleNamespace(sample_id=f"slide-{index}", image_path=Path("/tmp/s.svs"), mask_path=None)
        for index in range(num_slides)
    ]

    embedded = inference._select_embedding_path(
        model=_FakeModel("virchow2"),
        slide_records=slides,
        tiling_results=[_tiling_result() for _ in slides],
        preprocessing=PREPROCESSING,
        execution=ExecutionOptions(num_gpus=2),
        work_dir=tmp_path,
    )

    assert [slide.transform for slide in embedded] == [NORMALIZE_ONLY] * num_slides


def test_pooled_slide_sidecars_record_the_tile_encoder_dependency(tmp_path):
    """A slide-level run writes one identity into both its tile and its slide sidecar."""
    tile_artifact, slide_artifact = persist_embedded_slide(
        SimpleNamespace(name="prism", level="slide", _output_variant=None),
        _embedded_slide(
            tile_embeddings=np.zeros((1, 4), dtype=np.float32),
            slide_embedding=np.zeros((8,), dtype=np.float32),
        ),
        _tiling_result(),
        preprocessing=PreprocessingConfig(requested_spacing_um=0.5, requested_tile_size_px=224),
        execution=ExecutionOptions(
            output_dir=tmp_path, precision="fp16", output_dtype="fp32", save_tile_embeddings=True
        ),
    )

    expected = {
        "encoder_name": "prism",
        "output_variant": "default",
        "precision": "fp16",
        "feature_dtype": "fp32",
        "tile_encoder": "virchow",
        "tile_encoder_output_variant": "cls_patch_mean",
        "requested_tile_size_px": 224,
        "encoder_input_size_px": 224,
        "transform": NORMALIZE_ONLY,
    }
    assert load_metadata(slide_artifact.metadata_path)["compatibility"] == expected
    assert load_metadata(tile_artifact.metadata_path)["compatibility"] == expected


def test_pooled_hierarchical_sidecar_records_the_region_geometry(tmp_path):
    hierarchical_artifact, _ = persist_embedded_slide(
        SimpleNamespace(name="virchow2", level="tile", _output_variant=None),
        _embedded_slide(tile_embeddings=np.zeros((1, 4, 8), dtype=np.float32)),
        _tiling_result(base_spacing_um=0.5, level_downsamples=[1.0]),
        preprocessing=PreprocessingConfig(
            requested_spacing_um=0.5,
            requested_tile_size_px=224,
            requested_region_size_px=448,
            region_tile_multiple=2,
        ),
        execution=ExecutionOptions(output_dir=tmp_path, precision="fp16"),
    )

    assert load_metadata(hierarchical_artifact.metadata_path)["compatibility"] == {
        "encoder_name": "virchow2",
        "output_variant": "cls_patch_mean",
        "precision": "fp16",
        "feature_dtype": "fp16",
        "requested_tile_size_px": 224,
        "encoder_input_size_px": 224,
        "region_tile_multiple": 2,
        "requested_region_size_px": 448,
        "transform": NORMALIZE_ONLY,
    }
