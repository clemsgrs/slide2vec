"""``save_latents`` carries a slide encoder's latents to ``EmbeddedSlide`` and to disk.

Ways this can fail:

- aggregation drops the latents, so ``slide_latents/`` is never written;
- the resume completeness check waits for a latent file that is never written, so a
  finished run re-embeds every slide;
- the same check waits for latents from an encoder that has none;
- latents are computed when nobody asked for them;
- a distributed run loses the latents between the ranks and the parent.

The slide encoders here are small deterministic stand-ins registered under real preset
names; the tiles are read from the fixture WSI.
"""

import json
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image
from torchvision.transforms import v2

from slide2vec.api import ExecutionOptions, Model, Pipeline, PreprocessingConfig
from slide2vec.artifacts import load_array
from slide2vec.encoders.base import SlideEncoder
from slide2vec.runtime import distributed_stage, embedding_pipeline, manifest
from slide2vec.runtime.types import LoadedModel

pytest.importorskip("openslide")

WSI_PATH = Path(__file__).parent / "fixtures" / "input" / "test-wsi.tif"


class _PlainSlideEncoder(SlideEncoder):
    """Tiles -> channel means, slide -> mean of its tiles. No latents."""

    encode_dim = 3
    device = torch.device("cpu")

    def __init__(self) -> None:
        self.slides_encoded = 0

    def to(self, device):
        return self

    def encode_tiles(self, batch):
        return batch.float().mean(dim=(2, 3))

    def encode_slide(self, tile_features, coordinates=None, *, tile_size_lv0=None):
        self.slides_encoded += 1
        return tile_features.float().mean(dim=0)


class _LatentSlideEncoder(_PlainSlideEncoder):
    """PRISM-shaped stand-in: the latents are the per-channel min and max over the tiles."""

    def encode_slide_with_latents(self, tile_features, coordinates=None, *, tile_size_lv0=None):
        features = tile_features.float()
        embedding = self.encode_slide(features, coordinates, tile_size_lv0=tile_size_lv0)
        return embedding, torch.stack([features.min(dim=0).values, features.max(dim=0).values])


def _model(monkeypatch, preset: str, encoder: SlideEncoder) -> Model:
    model = Model.from_preset(preset, device="cpu")
    loaded = LoadedModel(
        name=preset,
        level="slide",
        model=encoder,
        transforms=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
        feature_dim=3,
        device=torch.device("cpu"),
        tile_feature_dim=3,
    )
    monkeypatch.setattr(model, "_load_backend", lambda: loaded)
    monkeypatch.setattr(model, "_load_backend_without_transform", lambda: loaded)
    return model


def _slide(tmp_path: Path) -> dict:
    """The fixture WSI with a tissue mask covering a block of about 4x4 tiles."""
    mask_path = tmp_path / "mask.png"
    labels = np.zeros((200, 216), dtype=np.uint8)
    labels[60:88, 60:88] = 1
    Image.fromarray(labels).save(mask_path)
    return {"sample_id": "slide-a", "image_path": WSI_PATH, "mask_path": mask_path}


def _preprocessing(**kwargs) -> PreprocessingConfig:
    return PreprocessingConfig(
        backend="openslide",
        requested_spacing_um=0.5,
        requested_tile_size_px=224,
        filtering={"a_t": 0, "a_h": 0},
        preview={"save_mask_preview": False, "save_tiling_preview": False},
        **kwargs,
    )


def _execution(output_dir: Path | None, *, save_latents: bool) -> ExecutionOptions:
    return ExecutionOptions(
        output_dir=output_dir,
        num_gpus=1,
        precision="fp32",
        num_workers_per_gpu=0,
        num_preprocessing_workers=1,
        save_tile_embeddings=True,
        save_latents=save_latents,
    )


def _min_max(tile_embeddings: torch.Tensor) -> torch.Tensor:
    return torch.stack([tile_embeddings.min(dim=0).values, tile_embeddings.max(dim=0).values])


def test_pipeline_saves_latents_and_resume_skips_the_completed_slide(tmp_path, monkeypatch):
    encoder = _LatentSlideEncoder()
    output_dir = tmp_path / "out"
    pipeline = Pipeline(
        _model(monkeypatch, "prism", encoder),
        _preprocessing(resume=True),
        execution=_execution(output_dir, save_latents=True),
    )

    artifact, = pipeline.run(slides=[_slide(tmp_path)]).slide_artifacts

    latent_path = (output_dir / "slide_latents" / "slide-a.pt").resolve()
    assert artifact.latent_path.resolve() == latent_path
    tile_embeddings = load_array(output_dir / "tile_embeddings" / "slide-a.pt")
    assert tile_embeddings.shape[0] > 1
    torch.testing.assert_close(load_array(latent_path), _min_max(tile_embeddings))
    assert encoder.slides_encoded == 1

    resumed, = pipeline.run(slides=[_slide(tmp_path)]).slide_artifacts

    assert encoder.slides_encoded == 1  # complete on disk, so nothing was embedded again
    assert resumed.latent_path.resolve() == latent_path


def test_pipeline_writes_no_latents_unless_requested(tmp_path, monkeypatch):
    output_dir = tmp_path / "out"

    artifact, = Pipeline(
        _model(monkeypatch, "prism", _LatentSlideEncoder()),
        _preprocessing(),
        execution=_execution(output_dir, save_latents=False),
    ).run(slides=[_slide(tmp_path)]).slide_artifacts

    assert artifact.latent_path is None
    assert not (output_dir / "slide_latents").exists()


def test_pipeline_resume_completes_for_an_encoder_without_latents(tmp_path, monkeypatch):
    encoder = _PlainSlideEncoder()
    output_dir = tmp_path / "out"
    pipeline = Pipeline(
        _model(monkeypatch, "moozy-slide", encoder),
        _preprocessing(resume=True),
        execution=_execution(output_dir, save_latents=True),
    )

    artifact, = pipeline.run(slides=[_slide(tmp_path)]).slide_artifacts
    assert artifact.latent_path is None
    assert encoder.slides_encoded == 1

    pipeline.run(slides=[_slide(tmp_path)])

    assert encoder.slides_encoded == 1


def test_aggregate_tiles_saves_latents(tmp_path, monkeypatch):
    model = _model(monkeypatch, "prism", _LatentSlideEncoder())
    tile_artifacts = Pipeline(
        model, _preprocessing(), execution=_execution(tmp_path / "tiles", save_latents=False)
    ).run(slides=[_slide(tmp_path)]).tile_artifacts

    artifact, = model.aggregate_tiles(
        tile_artifacts, execution=_execution(tmp_path / "slides", save_latents=True)
    )

    assert artifact.latent_path == (tmp_path / "slides" / "slide_latents" / "slide-a.pt").resolve()
    torch.testing.assert_close(
        load_array(artifact.latent_path), _min_max(load_array(tile_artifacts[0].path))
    )


def test_embed_slide_returns_latents_only_when_requested(tmp_path, monkeypatch):
    model = _model(monkeypatch, "prism", _LatentSlideEncoder())
    slide = _slide(tmp_path)

    with_latents = model.embed_slide(
        slide, preprocessing=_preprocessing(), execution=_execution(None, save_latents=True)
    )
    without_latents = model.embed_slide(
        slide, preprocessing=_preprocessing(), execution=_execution(None, save_latents=False)
    )

    torch.testing.assert_close(with_latents.latents, _min_max(with_latents.tile_embeddings))
    assert without_latents.latents is None


def test_embed_slide_keeps_latents_none_for_an_encoder_without_latents(tmp_path, monkeypatch):
    embedded = _model(monkeypatch, "moozy-slide", _PlainSlideEncoder()).embed_slide(
        _slide(tmp_path),
        preprocessing=_preprocessing(),
        execution=_execution(None, save_latents=True),
    )

    assert embedded.slide_embedding.shape == (3,)
    assert embedded.latents is None


# --------------------------------------------------------------------------------------
# Distributed paths: the torchrun launch is replaced, everything after it is real.
# --------------------------------------------------------------------------------------

_TILE_EMBEDDINGS = torch.tensor([[1.0, 2.0, 3.0], [5.0, 0.0, 4.0]])
_EXPECTED_LATENTS = torch.tensor([[1.0, 0.0, 3.0], [5.0, 2.0, 4.0]])


def _tiling_result() -> SimpleNamespace:
    return SimpleNamespace(
        x=np.array([0, 448], dtype=np.int64),
        y=np.array([0, 0], dtype=np.int64),
        tile_size_lv0=448,
        base_spacing_um=0.25,
        requested_spacing_um=0.5,
    )


def _spec(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(sample_id="slide-a", image_path=tmp_path / "slide-a.svs", mask_path=None)


def _distributed_execution() -> ExecutionOptions:
    return ExecutionOptions(num_gpus=2, precision="fp32", save_latents=True)


def test_tile_sharded_slide_carries_latents_from_the_parent_aggregation(tmp_path, monkeypatch):
    """One slide split across ranks: the ranks return tile shards, the parent aggregates."""
    loaded = SimpleNamespace(device=torch.device("cpu"), model=_LatentSlideEncoder())
    model = SimpleNamespace(
        level="slide", name="prism", _load_backend_without_transform=lambda: loaded
    )

    @contextmanager
    def coordination_dir(_work_dir):
        yield tmp_path / "coordination"

    monkeypatch.setattr(distributed_stage, "distributed_coordination_dir", coordination_dir)
    monkeypatch.setattr(
        distributed_stage, "run_distributed_direct_embedding_stage", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        distributed_stage,
        "load_tile_embedding_shards",
        lambda *args, **kwargs: [
            {"tile_index": np.array([0, 1], dtype=np.int64), "tile_embeddings": _TILE_EMBEDDINGS}
        ],
    )

    embedded = distributed_stage.embed_single_slide_distributed(
        model,
        slide=_spec(tmp_path),
        tiling_result=_tiling_result(),
        preprocessing=_preprocessing(),
        execution=_distributed_execution(),
        work_dir=tmp_path,
    )

    torch.testing.assert_close(embedded.latents, _EXPECTED_LATENTS)


def test_slide_sharded_run_carries_latents_from_the_rank_to_the_parent(tmp_path, monkeypatch):
    """Whole slides per rank: the rank aggregates and the parent reads its payload back."""
    import slide2vec.distributed as distributed
    from slide2vec.distributed import direct_embed_worker

    loaded = SimpleNamespace(device=torch.device("cpu"), model=_LatentSlideEncoder())
    rank_model = SimpleNamespace(
        level="slide",
        name="prism",
        _declare_encoder_input=lambda *args, **kwargs: None,
        _load_backend=lambda: loaded,
    )
    slide, tiling_result = _spec(tmp_path), _tiling_result()
    monkeypatch.setattr(distributed, "enable", lambda overwrite=True: None)
    monkeypatch.setattr(distributed, "get_local_rank", lambda: 0)
    monkeypatch.setattr(distributed, "get_global_rank", lambda: 0)
    monkeypatch.setattr(distributed, "get_global_size", lambda: 1)
    monkeypatch.setattr(distributed, "get_device_ordinal", lambda: 0)
    monkeypatch.setattr(Model, "from_preset", lambda *args, **kwargs: rank_model)
    monkeypatch.setattr(
        manifest, "load_successful_tiled_slides", lambda output_dir: ([slide], [tiling_result])
    )
    monkeypatch.setattr(
        embedding_pipeline,
        "compute_tile_embeddings_for_slide",
        lambda *args, **kwargs: _TILE_EMBEDDINGS,
    )

    def run_rank_in_process(module, *, output_dir, request_path, **kwargs):
        assert module == "slide2vec.distributed.direct_embed_worker"
        request = json.loads(Path(request_path).read_text())
        assert request["execution"]["save_latents"] is True
        assert direct_embed_worker.main(
            ["--output-dir", str(output_dir), "--request-path", str(request_path)]
        ) == 0

    monkeypatch.setattr(
        distributed_stage,
        "run_torchrun_worker",
        lambda *, module, **kwargs: run_rank_in_process(module, **kwargs),
    )
    monkeypatch.setattr(
        distributed_stage, "assign_slides_to_ranks", lambda *args, **kwargs: {0: ["slide-a"]}
    )

    embedded, = distributed_stage.embed_multi_slides_distributed(
        SimpleNamespace(level="slide", name="prism", allow_non_recommended_settings=False),
        slide_records=[slide],
        tiling_results=[tiling_result],
        preprocessing=_preprocessing(),
        execution=_distributed_execution(),
        work_dir=tmp_path,
    )

    torch.testing.assert_close(embedded.latents, _EXPECTED_LATENTS)
    torch.testing.assert_close(embedded.slide_embedding, torch.tensor([3.0, 1.0, 3.5]))
