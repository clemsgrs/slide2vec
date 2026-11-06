"""Loading tiled slides from a real ``process_list.csv``: which rows load, and zero-tile sidecars.

Ways it could go wrong, each covered below:

* a run reads, warns about or fails on rows of a shared tiling directory it did not ask for;
* a requested zero-tile slide is dropped, so it never gets its metadata sidecar;
* the zero-tile warning repeats in each multi-GPU rank, or a rank loads another rank's units;
* a multi-class slide's zero-tile class is loaded by the ranks that embed its other class,
  including when that other class is ``tissue``, whose work unit is the bare sample id;
* a zero-tile slide is embedded, or the set of embedded slides changes.
"""

import importlib
import logging
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from hs2p import FilterConfig, MaskSamplingConfig, PreviewConfig, SegmentationConfig, SlideSpec
from PIL import Image

import slide2vec.inference as inference
from slide2vec.api import ExecutionOptions, PreprocessingConfig
from slide2vec.artifacts import load_array, load_metadata
from slide2vec.runtime import distributed_stage, embedding_pipeline, manifest, tiling_pipeline
from slide2vec.runtime.distributed import encode_work_unit
from tests.tiling_helpers import tiling_config

PREPROCESSING = PreprocessingConfig(
    tiling=tiling_config(0.5, 32),
    segmentation=SegmentationConfig(method="hsv", downsample=1),
    filter=FilterConfig(ref_tile_size=32, a_t=0, a_h=0),
    preview=PreviewConfig(),
)

#: A tissue mask that keeps all 16 tiles of a 128x128 slide, and one that keeps none.
TISSUE = np.ones((32, 32), dtype=np.uint8)
BACKGROUND = np.zeros((32, 32), dtype=np.uint8)


def _slide(root: Path, sample_id: str, labels: np.ndarray) -> SlideSpec:
    slides_dir = root / "slides"
    slides_dir.mkdir(parents=True, exist_ok=True)
    image_path = slides_dir / f"{sample_id}.png"
    mask_path = slides_dir / f"{sample_id}-mask.png"
    Image.new("RGB", (128, 128), (160, 70, 110)).save(image_path)
    Image.fromarray(labels).save(mask_path)
    return SlideSpec(
        sample_id=sample_id,
        image_path=image_path,
        mask_path=mask_path,
        spacing_at_level_0=0.5,
    )


def _tile(root: Path, slides, preprocessing: PreprocessingConfig = PREPROCESSING) -> Path:
    """Tile ``slides`` with hs2p into ``root/coordinates``; return that directory."""
    coordinates_dir = root / "coordinates"
    tiling_pipeline.prepare_tiled_slides(
        slides, preprocessing, output_dir=coordinates_dir, num_workers=1
    )
    return coordinates_dir


def _shared_tiling_dir(root: Path) -> Path:
    """A tiling directory shared with another dataset.

    ``requested`` has tiles and ``requested-empty`` has none; ``foreign-empty`` has none and
    ``foreign-missing`` lost its coordinate files.
    """
    coordinates_dir = _tile(
        root,
        [
            _slide(root, "requested", TISSUE),
            _slide(root, "requested-empty", BACKGROUND),
            _slide(root, "foreign-empty", BACKGROUND),
            _slide(root, "foreign-missing", TISSUE),
        ],
    )
    for path in (coordinates_dir / "tiles").glob("foreign-missing.*"):
        path.unlink()
    return coordinates_dir


def _no_tiles_warnings(caplog) -> list[str]:
    return [
        record.getMessage()
        for record in caplog.records
        if "no tiles extracted" in record.getMessage()
    ]


# --- load_successful_tiled_slides ---------------------------------------------------------------


def test_loader_skips_rows_outside_the_requested_work_units(tmp_path, caplog):
    coordinates_dir = _shared_tiling_dir(tmp_path)
    caplog.clear()

    with caplog.at_level(logging.WARNING):
        slides, tiling_results = manifest.load_successful_tiled_slides(
            coordinates_dir, work_units=["requested"]
        )

    assert [slide.sample_id for slide in slides] == ["requested"]
    assert [len(result.x) for result in tiling_results] == [16]
    assert _no_tiles_warnings(caplog) == []


def test_loader_returns_a_requested_zero_tile_slide_and_warns_once(tmp_path, caplog):
    coordinates_dir = _shared_tiling_dir(tmp_path)
    caplog.clear()

    with caplog.at_level(logging.WARNING):
        slides, tiling_results = manifest.load_successful_tiled_slides(
            coordinates_dir, work_units=["requested", "requested-empty"]
        )

    assert [slide.sample_id for slide in slides] == ["requested", "requested-empty"]
    assert [len(result.x) for result in tiling_results] == [16, 0]
    assert _no_tiles_warnings(caplog) == ["Skipping requested-empty: no tiles extracted"]


#: Samples ``tumor`` and ``stroma`` per class; a slide of only ``tumor`` pixels has no stroma tiles.
CLASSES = MaskSamplingConfig(
    pixel_mapping={"background": 0, "tissue": 1, "tumor": 2, "stroma": 3},
    colors={
        "background": None,
        "tissue": (157, 219, 129),
        "tumor": (255, 0, 0),
        "stroma": (0, 0, 255),
    },
    min_coverage={"tumor": 0.5, "stroma": 0.5},
    output_mode="per_annotation",
    strategy="independent",
)
TUMOR = np.full((32, 32), 2, dtype=np.uint8)
#: Samples ``tissue`` and ``stroma`` per class; a slide of only ``tissue`` pixels has no stroma tiles.
TISSUE_AND_STROMA = replace(CLASSES, min_coverage={"tissue": 0.5, "stroma": 0.5})

#: ``(labels, classes, the class with tiles)``: each slide has 16 tiles of one class, none of stroma.
MULTI_CLASS_CASES = {
    "tumor-and-stroma": (TUMOR, CLASSES, "tumor"),
    "tissue-and-stroma": (TISSUE, TISSUE_AND_STROMA, "tissue"),
}


def _multi_class_tiling_dir(root: Path, case: str = "tumor-and-stroma") -> Path:
    """Slide ``slide`` has 16 tiles of one class and no stroma tile."""
    labels, classes, _ = MULTI_CLASS_CASES[case]
    return _tile(root, [_slide(root, "slide", labels)], replace(PREPROCESSING, masks=classes))


def test_loader_selects_one_class_row_by_its_work_unit(tmp_path, caplog):
    coordinates_dir = _multi_class_tiling_dir(tmp_path)
    caplog.clear()

    with caplog.at_level(logging.WARNING):
        slides, tiling_results = manifest.load_successful_tiled_slides(
            coordinates_dir, work_units=[encode_work_unit("slide", "tumor")]
        )

    assert [slide.sample_id for slide in slides] == ["slide"]
    assert [(result.annotation, len(result.x)) for result in tiling_results] == [("tumor", 16)]
    assert _no_tiles_warnings(caplog) == []


def test_loader_selects_only_the_tissue_row_of_a_bare_work_unit(tmp_path, caplog):
    coordinates_dir = _multi_class_tiling_dir(tmp_path, "tissue-and-stroma")
    caplog.clear()

    with caplog.at_level(logging.WARNING):
        _, tiling_results = manifest.load_successful_tiled_slides(
            coordinates_dir, work_units=[encode_work_unit("slide", "tissue")]
        )

    assert [(result.annotation, len(result.x)) for result in tiling_results] == [("tissue", 16)]
    assert _no_tiles_warnings(caplog) == []


def test_loader_selects_every_class_row_of_a_sample_id(tmp_path):
    coordinates_dir = _multi_class_tiling_dir(tmp_path)

    _, tiling_results = manifest.load_successful_tiled_slides(coordinates_dir, sample_ids=["slide"])

    assert [(result.annotation, len(result.x)) for result in tiling_results] == [
        ("tumor", 16),
        ("stroma", 0),
    ]


def test_loader_without_work_units_keeps_every_successful_row(tmp_path, caplog):
    coordinates_dir = _tile(
        tmp_path,
        [_slide(tmp_path, "slide-a", TISSUE), _slide(tmp_path, "slide-empty", BACKGROUND)],
    )
    caplog.clear()

    with caplog.at_level(logging.WARNING):
        slides, tiling_results = manifest.load_successful_tiled_slides(coordinates_dir)

    assert [slide.sample_id for slide in slides] == ["slide-a", "slide-empty"]
    assert [len(result.x) for result in tiling_results] == [16, 0]
    assert _no_tiles_warnings(caplog) == ["Skipping slide-empty: no tiles extracted"]


# --- entry points -------------------------------------------------------------------------------


def _model():
    return SimpleNamespace(
        name="virchow2",
        level="tile",
        _requested_device="cpu",
        _output_variant=None,
        allow_non_recommended_settings=False,
        _declare_encoder_input=lambda *args, **kwargs: None,
        _load_backend=lambda: SimpleNamespace(
            feature_dim=2, device="cpu", model=SimpleNamespace(), transforms=None
        ),
    )


@pytest.fixture
def encoded(monkeypatch) -> list[tuple[str, str | None]]:
    """Stub the tile encoder; collect the ``(sample_id, annotation)`` of each unit it encodes."""
    encoded_units: list[tuple[str, str | None]] = []

    def encode_tiles(_loaded, _model, slide, tiling_result, *, tile_indices=None, **_kwargs):
        assert len(tiling_result.x) > 0, f"{slide.sample_id} has no tile to encode"
        encoded_units.append((slide.sample_id, tiling_result.annotation))
        count = len(tiling_result.x) if tile_indices is None else len(tile_indices)
        return np.ones((count, 2), dtype=np.float32)

    monkeypatch.setattr(embedding_pipeline, "compute_tile_embeddings_for_slide", encode_tiles)
    return encoded_units


@pytest.fixture(params=[1, 2], ids=["one-gpu", "two-gpus"])
def num_gpus(request, monkeypatch) -> int:
    """Run in-process, or through mocked torchrun ranks.

    The mocked launch runs the real worker ``main`` once per rank, in this process.
    """
    if request.param == 1:
        return 1
    import slide2vec.api as api
    import slide2vec.distributed as distributed

    rank = {"value": 0}

    def fake_torchrun(*, module, num_gpus, output_dir, request_path, **_kwargs):
        worker = importlib.import_module(module)
        for rank["value"] in range(num_gpus):
            assert worker.main(
                ["--output-dir", str(output_dir), "--request-path", str(request_path)]
            ) == 0

    monkeypatch.setattr(distributed_stage, "run_torchrun_worker", fake_torchrun)
    monkeypatch.setattr(distributed_stage, "validate_multi_gpu_execution", lambda *a, **k: None)
    monkeypatch.setattr(distributed, "enable", lambda **_kwargs: None)
    monkeypatch.setattr(distributed, "get_global_rank", lambda: rank["value"])
    monkeypatch.setattr(distributed, "get_local_rank", lambda: rank["value"])
    monkeypatch.setattr(distributed, "get_global_size", lambda: request.param)
    monkeypatch.setattr(distributed, "get_device_ordinal", lambda: 0)
    monkeypatch.setattr(api.Model, "from_preset", classmethod(lambda cls, name, **_kwargs: _model()))
    return request.param


def _zero_tile_sidecar(output_dir: Path, sample_id: str) -> dict:
    return load_metadata(output_dir / "tile_embeddings" / f"{sample_id}.meta.json")


def test_run_with_coordinates_loads_only_the_requested_slides(tmp_path, caplog, encoded, num_gpus):
    coordinates_dir = _shared_tiling_dir(tmp_path)
    output_dir = tmp_path / "out"
    requested = [_slide(tmp_path, "requested", TISSUE), _slide(tmp_path, "requested-empty", BACKGROUND)]
    caplog.clear()

    with caplog.at_level(logging.WARNING):
        result = inference.run_pipeline_with_coordinates(
            _model(),
            coordinates_dir=coordinates_dir,
            slides=requested,
            preprocessing=PREPROCESSING,
            execution=ExecutionOptions(output_dir=output_dir, output_format="npz", num_gpus=num_gpus),
        )

    assert encoded == [("requested", None)]
    assert [artifact.sample_id for artifact in result.tile_artifacts] == ["requested"]
    assert _no_tiles_warnings(caplog) == ["Skipping requested-empty: no tiles extracted"]
    sidecar = _zero_tile_sidecar(output_dir, "requested-empty")
    assert (sidecar["num_tiles"], sidecar["feature_dim"]) == (0, None)


#: 64 px regions of 2x2 subtiles of 32 px.
HIERARCHICAL = replace(PREPROCESSING, requested_region_size_px=64, region_tile_multiple=2)


@pytest.fixture
def encoded_regions(monkeypatch) -> list[str]:
    """Stub the region encoder; collect the ``sample_id`` of each slide it encodes."""
    encoded_ids: list[str] = []

    def encode_regions(_loaded, slide, tiling_result, **_kwargs):
        assert len(tiling_result.x) > 0, f"{slide.sample_id} has no region to encode"
        encoded_ids.append(slide.sample_id)
        return np.ones((len(tiling_result.x), 4, 2), dtype=np.float32)

    def encode_region_shard(_loaded, slide, tiling_result, *, flat_indices, **_kwargs):
        assert len(tiling_result.x) > 0, f"{slide.sample_id} has no region to encode"
        encoded_ids.append(slide.sample_id)
        return np.asarray(flat_indices), np.ones((len(flat_indices), 2), dtype=np.float32)

    monkeypatch.setattr(embedding_pipeline, "compute_hierarchical_embeddings_for_slide", encode_regions)
    monkeypatch.setattr(
        embedding_pipeline, "compute_hierarchical_embedding_shard_for_slide", encode_region_shard
    )
    return encoded_ids


@pytest.mark.parametrize("entry_point", ["run_with_coordinates", "run", "embed_slides"])
def test_a_hierarchical_zero_tile_slide_gets_its_empty_result(
    tmp_path, encoded_regions, num_gpus, entry_point
):
    slides = [_slide(tmp_path, "slide-a", TISSUE), _slide(tmp_path, "slide-empty", BACKGROUND)]
    output_dir = tmp_path / "out"
    execution = ExecutionOptions(output_dir=output_dir, output_format="npz", num_gpus=num_gpus)

    if entry_point == "run_with_coordinates":
        inference.run_pipeline_with_coordinates(
            _model(),
            coordinates_dir=_tile(tmp_path, slides, HIERARCHICAL),
            preprocessing=HIERARCHICAL,
            execution=execution,
        )
    elif entry_point == "run":
        inference.run_pipeline(_model(), slides=slides, preprocessing=HIERARCHICAL, execution=execution)
    else:
        inference.embed_slides(_model(), slides, preprocessing=HIERARCHICAL, execution=execution)

    assert set(encoded_regions) == {"slide-a"}
    hierarchical_dir = output_dir / "hierarchical_embeddings"
    # Four 64 px regions cover the 128 px slide.
    assert load_array(hierarchical_dir / "slide-a.npz").shape == (4, 4, 2)
    sidecar = load_metadata(hierarchical_dir / "slide-empty.meta.json")
    assert (sidecar["num_regions"], sidecar["tiles_per_region"]) == (0, 4)
    assert load_array(hierarchical_dir / "slide-empty.npz").shape == (0, 4, 0)


def test_run_writes_the_zero_tile_sidecar(tmp_path, caplog, encoded, num_gpus):
    slides = [_slide(tmp_path, "slide-a", TISSUE), _slide(tmp_path, "slide-empty", BACKGROUND)]
    output_dir = tmp_path / "out"

    with caplog.at_level(logging.WARNING):
        result = inference.run_pipeline(
            _model(),
            slides=slides,
            preprocessing=PREPROCESSING,
            execution=ExecutionOptions(output_dir=output_dir, output_format="npz", num_gpus=num_gpus),
        )

    assert encoded == [("slide-a", None)]
    assert [artifact.sample_id for artifact in result.tile_artifacts] == ["slide-a"]
    assert _no_tiles_warnings(caplog) == ["Skipping slide-empty: no tiles extracted"]
    sidecar = _zero_tile_sidecar(output_dir, "slide-empty")
    assert (sidecar["num_tiles"], sidecar["feature_dim"]) == (0, None)


@pytest.mark.parametrize(
    "tiled_ids",
    [["slide-a"], ["slide-a", "slide-b"]],
    ids=["one-slide-with-tiles", "two-slides-with-tiles"],
)
def test_embed_slides_writes_the_zero_tile_sidecar(tmp_path, caplog, encoded, num_gpus, tiled_ids):
    slides = [_slide(tmp_path, sample_id, TISSUE) for sample_id in tiled_ids]
    slides.append(_slide(tmp_path, "slide-empty", BACKGROUND))
    output_dir = tmp_path / "out"

    with caplog.at_level(logging.WARNING):
        embedded = inference.embed_slides(
            _model(),
            slides,
            preprocessing=PREPROCESSING,
            execution=ExecutionOptions(output_dir=output_dir, output_format="npz", num_gpus=num_gpus),
        )

    assert sorted(set(encoded)) == [(sample_id, None) for sample_id in tiled_ids]
    assert [slide.sample_id for slide in embedded] == tiled_ids
    assert _no_tiles_warnings(caplog) == ["Skipping slide-empty: no tiles extracted"]
    sidecar = _zero_tile_sidecar(output_dir, "slide-empty")
    assert (sidecar["num_tiles"], sidecar["feature_dim"]) == (0, None)


def _multi_class_run(entry_point: str, case: str, root: Path, execution: ExecutionOptions):
    """Return a call that embeds ``slide`` (16 tiles of one class, no stroma tile) through ``entry_point``."""
    labels, classes, _ = MULTI_CLASS_CASES[case]
    preprocessing = replace(PREPROCESSING, masks=classes)
    slide = _slide(root, "slide", labels)
    if entry_point == "run_with_coordinates":
        coordinates_dir = _tile(root, [slide], preprocessing)
        return lambda: inference.run_pipeline_with_coordinates(
            _model(),
            coordinates_dir=coordinates_dir,
            slides=[slide],
            preprocessing=preprocessing,
            execution=execution,
        )
    if entry_point == "run":
        return lambda: inference.run_pipeline(
            _model(), slides=[slide], preprocessing=preprocessing, execution=execution
        )
    return lambda: inference.embed_slides(
        _model(), [slide], preprocessing=preprocessing, execution=execution
    )


@pytest.mark.parametrize("case", list(MULTI_CLASS_CASES))
@pytest.mark.parametrize("entry_point", ["run_with_coordinates", "run", "embed_slides"])
def test_a_zero_tile_class_gets_its_sidecar_and_one_warning(
    tmp_path, caplog, encoded, num_gpus, entry_point, case
):
    output_dir = tmp_path / "out"
    run = _multi_class_run(
        entry_point,
        case,
        tmp_path,
        ExecutionOptions(output_dir=output_dir, output_format="npz", num_gpus=num_gpus),
    )
    caplog.clear()

    with caplog.at_level(logging.WARNING):
        run()

    assert set(encoded) == {("slide", MULTI_CLASS_CASES[case][2])}
    assert _no_tiles_warnings(caplog) == ["Skipping slide: no tiles extracted"]
    sidecar = load_metadata(output_dir / "tile_embeddings" / "stroma" / "slide.meta.json")
    assert (sidecar["num_tiles"], sidecar["feature_dim"]) == (0, None)
