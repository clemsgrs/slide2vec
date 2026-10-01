"""Tests for the parent-side given-image orchestration (issue #234).

``embed_images`` normalizes the caller's images, resume-filters them, then either runs the
encode/write loop in-process (``num_gpus=1``) or fans it out over torchrun ranks
(``num_gpus>1``) — reusing the same distributed machinery the dense path uses. These tests
run on CPU with a random-weight encoder; the torchrun launch is captured rather than executed.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
timm = pytest.importorskip("timm")
PIL = pytest.importorskip("PIL")

from PIL import Image  # noqa: E402

from slide2vec.api import ExecutionOptions, ImageSpec  # noqa: E402
from slide2vec.encoders.base import TimmTileEncoder  # noqa: E402
from slide2vec.runtime import image_specs, image_stage  # noqa: E402
from slide2vec.runtime.types import LoadedModel  # noqa: E402


def _encoder() -> TimmTileEncoder:
    return TimmTileEncoder("vit_tiny_patch16_224", pretrained=False, num_classes=0)


def _loaded(encoder: TimmTileEncoder) -> LoadedModel:
    return LoadedModel(
        name="fake-encoder",
        level="tile",
        model=encoder,
        transforms=encoder.get_transform(),
        feature_dim=int(encoder.encode_dim),
        device=torch.device("cpu"),
    )


class _FakeModel:
    """Minimal ``Model`` stand-in exposing what the in-process image path reads.

    ``_load_backend`` refuses to hand out a backend until the Given encoder-input contract
    has been declared, mirroring the real ``Model``.
    """

    def __init__(self, encoder, *, name: str = "fake-encoder") -> None:
        self._loaded = _loaded(encoder)
        self.name = name
        self.level = "tile"
        self._output_variant = None
        self._requested_device = "cpu"
        self.allow_non_recommended_settings = False
        self.declared_given = False

    def _declare_given_encoder_input(self, *, emit_run_info):
        self.declared_given = True

    def _load_backend(self):
        assert self.declared_given, "the image path must declare Given before it loads"
        return self._loaded


def _images(tmp_path, names, *, width=64, height=64) -> list[ImageSpec]:
    specs = []
    for name in names:
        path = tmp_path / "images" / f"{name}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(abs(hash(name)) % (2**32))
        pixels = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
        Image.fromarray(pixels).save(path)
        specs.append(ImageSpec(sample_id=name, image_path=path))
    return specs


def test_embed_images_num_gpus_one_runs_in_process(tmp_path, monkeypatch):
    """``num_gpus=1`` encodes in-process — no torchrun subprocess — and writes every image."""
    monkeypatch.setattr(
        image_stage, "run_torchrun_worker",
        lambda **kwargs: pytest.fail("num_gpus=1 must not launch torchrun"),
    )
    model = _FakeModel(_encoder())
    execution = ExecutionOptions(
        output_dir=tmp_path / "out",
        num_gpus=1,
        precision="fp32",
        num_workers_per_gpu=0,
    )

    artifacts = image_stage.embed_images(model, _images(tmp_path, ["a", "b", "c"]), execution=execution)

    assert [artifact.sample_id for artifact in artifacts] == ["a", "b", "c"]
    assert model.declared_given
    embeddings_dir = tmp_path / "out" / "image_embeddings"
    assert sorted(p.name for p in embeddings_dir.glob("*.pt")) == ["a.pt", "b.pt", "c.pt"]
    for artifact in artifacts:
        assert artifact.metadata_path.exists()


def test_embed_images_auto_workers_complete_in_a_subprocess(
    tmp_path,
    assert_auto_worker_workflow_completes_in_subprocess,
    build_auto_worker_model,
):
    """Auto workers must not fork after the in-process encoder initialized its runtime."""
    def workflow():
        model, execution = build_auto_worker_model(_loaded(_encoder()))
        return model.embed_images(
            _images(tmp_path, ["a"]),
            execution=execution,
        )

    assert_auto_worker_workflow_completes_in_subprocess(
        child_env_name="SLIDE2VEC_IMAGE_AUTO_WORKER_CHILD",
        workflow=workflow,
        expected_sample_ids=["a"],
    )


def test_embed_images_num_gpus_gt_one_launches_image_worker(tmp_path, monkeypatch):
    """``num_gpus>1`` launches ``slide2vec.distributed.image_worker`` under torchrun; the
    parent itself encodes nothing and collects the artifacts back off disk."""
    from slide2vec.runtime.image_shard import run_image_shard
    from slide2vec.runtime.sharding import plan_contiguous_shards

    caller_dir = tmp_path / "caller"
    caller_dir.mkdir()
    monkeypatch.chdir(caller_dir)
    captured: dict = {}
    rank_loaded = _loaded(_encoder())  # every rank loads the same checkpoint

    def _fake_run(*, module, num_gpus, output_dir, request_path, **kwargs):
        request = json.loads(Path(request_path).read_text())
        captured.update(module=module, num_gpus=num_gpus, output_dir=output_dir, request=request)
        specs = image_specs.image_specs_from_request(request)
        for shard in plan_contiguous_shards(specs, num_gpus):
            run_image_shard(shard, loaded=rank_loaded, out_dir=Path(output_dir), batch_size=2,
                            output_precision="fp32", num_workers=0,
                            identity={"encoder_name": "fake-encoder"})

    monkeypatch.setattr(image_stage, "run_torchrun_worker", _fake_run)
    monkeypatch.setattr(image_stage, "validate_multi_gpu_execution", lambda *a, **k: None)
    monkeypatch.setattr(
        image_stage, "_run_images_in_process",
        lambda *a, **k: pytest.fail("num_gpus>1 must not encode in-process"),
    )
    model = _FakeModel(_encoder())
    execution = ExecutionOptions(output_dir=Path("output"), num_gpus=3, precision="fp32")

    artifacts = image_stage.embed_images(model, _images(tmp_path, ["a", "b", "c"]), execution=execution)

    assert captured["module"] == "slide2vec.distributed.image_worker"
    assert captured["num_gpus"] == 3
    expected_output_dir = (caller_dir / "output").resolve()
    assert Path(captured["output_dir"]) == expected_output_dir
    assert captured["request"]["execution"]["output_dir"] == str(expected_output_dir)
    assert [image["sample_id"] for image in captured["request"]["images"]] == ["a", "b", "c"]
    assert all(Path(image["image_path"]).is_absolute() for image in captured["request"]["images"])
    assert len(artifacts) == 3
    assert (expected_output_dir / "image_embeddings" / "a.pt").exists()


def test_embed_images_resume_skips_existing_and_logs(tmp_path, caplog):
    """Resume: images already on disk are filtered before dispatch; the skip count is logged."""
    model = _FakeModel(_encoder())
    execution = ExecutionOptions(output_dir=tmp_path / "out", num_gpus=1, precision="fp32")
    first = _images(tmp_path, ["a", "b"])
    image_stage.embed_images(model, first, execution=execution)
    written_at = {spec.sample_id: (tmp_path / "out" / "image_embeddings" / f"{spec.sample_id}.pt").stat().st_mtime_ns
                  for spec in first}

    with caplog.at_level("INFO", logger="slide2vec.runtime.image_stage"):
        artifacts = image_stage.embed_images(model, _images(tmp_path, ["a", "b", "c"]), execution=execution)

    assert len(artifacts) == 3
    assert "2/3 images already on disk, encoding 1" in caplog.text
    for sample_id, mtime in written_at.items():
        payload = tmp_path / "out" / "image_embeddings" / f"{sample_id}.pt"
        assert payload.stat().st_mtime_ns == mtime  # untouched


def test_embed_images_all_present_does_not_dispatch(tmp_path, monkeypatch):
    """A fully-resumed run encodes nothing yet still returns every artifact."""
    model = _FakeModel(_encoder())
    execution = ExecutionOptions(output_dir=tmp_path / "out", num_gpus=1, precision="fp32")
    specs = _images(tmp_path, ["a", "b"])
    image_stage.embed_images(model, specs, execution=execution)

    monkeypatch.setattr(image_stage, "_run_images_in_process",
                        lambda *a, **k: pytest.fail("nothing to encode"))
    monkeypatch.setattr(image_stage, "run_torchrun_worker",
                        lambda **k: pytest.fail("nothing to encode"))
    assert len(image_stage.embed_images(model, specs, execution=execution)) == 2


SHIPPED_TRANSFORM = {
    "normalize": {"mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]},
    "resize": {"size": [248], "interpolation": "bicubic"},
    "center_crop": {"size": [224, 224]},
}


def _sidecar(tmp_path, sample_id: str) -> dict:
    path = tmp_path / "out" / "image_embeddings" / f"{sample_id}.meta.json"
    return json.loads(path.read_text(encoding="utf-8"))


def test_embed_images_sidecar_records_the_feature_identity(tmp_path):
    """A given image declares no tile geometry: the identity is the encoder and its recipe."""
    execution = ExecutionOptions(
        output_dir=tmp_path / "out", num_gpus=1, precision="fp16", output_dtype="fp32",
        num_workers_per_gpu=0,
    )

    image_stage.embed_images(_FakeModel(_encoder()), _images(tmp_path, ["a"]), execution=execution)

    assert _sidecar(tmp_path, "a")["compatibility"] == {
        "encoder_name": "fake-encoder",
        "output_variant": None,
        "precision": "fp16",
        "feature_dtype": "fp32",
        "transform": SHIPPED_TRANSFORM,
    }


def test_embed_images_resume_refuses_artifacts_from_a_different_encoder(tmp_path, monkeypatch):
    execution = ExecutionOptions(
        output_dir=tmp_path / "out", num_gpus=1, precision="fp32", num_workers_per_gpu=0
    )
    specs = _images(tmp_path, ["a"])
    image_stage.embed_images(_FakeModel(_encoder()), specs, execution=execution)
    monkeypatch.setattr(
        image_stage, "_run_images_in_process",
        lambda *a, **k: pytest.fail("encoding must not start on a stale resume"),
    )

    with pytest.raises(ValueError) as error:
        image_stage.embed_images(
            _FakeModel(_encoder(), name="other-encoder"), specs, execution=execution
        )

    assert str(error.value) == (
        "Cannot resume 'a': the existing image embeddings at "
        f"{tmp_path / 'out' / 'image_embeddings' / 'a.pt'} were computed with a different "
        "feature identity: encoder_name (recorded 'fake-encoder', requested 'other-encoder'). "
        "Re-run into a new output_dir, delete the stale artifacts, or request the recorded "
        "values."
    )


def test_embed_images_resume_refuses_artifacts_from_a_different_transform(tmp_path):
    from torchvision.transforms import v2

    execution = ExecutionOptions(
        output_dir=tmp_path / "out", num_gpus=1, precision="fp32", num_workers_per_gpu=0
    )
    specs = _images(tmp_path, ["a"])
    image_stage.embed_images(_FakeModel(_encoder()), specs, execution=execution)
    changed = _FakeModel(_encoder())
    changed._loaded.transforms = v2.Compose([
        v2.ToImage(),
        v2.Resize((224, 224), antialias=True),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ])

    with pytest.raises(ValueError) as error:
        image_stage.embed_images(changed, specs, execution=execution)

    assert (
        "transform.center_crop (recorded {'size': [224, 224]}, requested None); "
        "transform.resize.interpolation (recorded 'bicubic', requested 'bilinear'); "
        "transform.resize.size (recorded [248], requested [224, 224])"
    ) in str(error.value)


def test_embed_images_resume_accepts_sidecars_that_lack_the_identity(tmp_path, caplog):
    """Artifacts written before the identity existed are reused, with one warning per run."""
    execution = ExecutionOptions(
        output_dir=tmp_path / "out", num_gpus=1, precision="fp32", num_workers_per_gpu=0
    )
    specs = _images(tmp_path, ["a", "b"])
    image_stage.embed_images(_FakeModel(_encoder()), specs, execution=execution)
    for spec in specs:
        path = tmp_path / "out" / "image_embeddings" / f"{spec.sample_id}.meta.json"
        legacy = json.loads(path.read_text(encoding="utf-8"))
        del legacy["compatibility"]
        path.write_text(json.dumps(legacy), encoding="utf-8")
    model = _FakeModel(_encoder(), name="other-encoder")
    model._load_backend = lambda: pytest.fail("no recorded transform, so no encoder to load")

    with caplog.at_level("WARNING", logger="slide2vec.runtime.image_stage"):
        artifacts = image_stage.embed_images(model, specs, execution=execution)

    assert [artifact.sample_id for artifact in artifacts] == ["a", "b"]
    assert [record.getMessage() for record in caplog.records] == [
        "Resuming over 2 completed sidecar(s) that do not record encoder_name, "
        "feature_dtype, output_variant, precision, transform; cannot verify those fields "
        "against this run."
    ]


def test_embed_images_rejects_duplicate_sample_ids(tmp_path):
    """Two images sharing a sample id would silently overwrite one another's artifact."""
    model = _FakeModel(_encoder())
    execution = ExecutionOptions(output_dir=tmp_path / "out", num_gpus=1, precision="fp32")
    specs = _images(tmp_path, ["a"])
    duplicated = [*specs, ImageSpec(sample_id="a", image_path=tmp_path / "images" / "a.png")]

    with pytest.raises(ValueError, match="duplicate sample_id"):
        image_stage.embed_images(model, duplicated, execution=execution)


def test_embed_images_rejects_raster_level0_spacing_override(tmp_path, monkeypatch):
    model = _FakeModel(_encoder())
    monkeypatch.setattr(
        model,
        "_load_backend",
        lambda: pytest.fail("override validation must precede backend loading"),
    )
    execution = ExecutionOptions(output_dir=tmp_path / "out", num_gpus=1, precision="fp32")
    spec = ImageSpec(
        sample_id="raster",
        image_path=tmp_path / "image.png",
        spacing_at_level_0=0.25,
    )

    with pytest.raises(ValueError, match=r"spacing_at_level_0.*raster"):
        image_stage.embed_images(model, [spec], execution=execution)


def test_image_specs_round_trip_through_request(tmp_path):
    """The request rebuilds the exact spec list the parent sharded (the worker's inverse)."""
    specs = [
        ImageSpec(sample_id="a", image_path="/data/a.png"),
        ImageSpec(sample_id="b", image_path="/data/nested/b.tif"),
    ]
    request = image_specs.build_image_specs_request(specs)
    assert image_specs.image_specs_from_request(request) == specs


def test_declaring_given_selects_the_shipped_transform(monkeypatch):
    """The path states Given explicitly — the contract's own mechanism, not an absent one."""
    import slide2vec.inference as inference
    from slide2vec.api import Model

    class _StandInEncoder:
        shipped = object()

        def __init__(self, *, output_variant=None, allow_non_recommended_settings=False):
            self.device = torch.device("cpu")
            self.encode_dim = 4
            self.patch_size = (16, 16)

        def get_transform(self):
            return self.shipped

        def to(self, device):
            return self

    monkeypatch.setattr(inference.encoder_registry, "require", lambda name: _StandInEncoder)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    model = Model.from_preset("gigapath", device="cpu")

    with pytest.raises(ValueError, match="No encoder-input contract"):
        model._load_backend()

    model._declare_given_encoder_input(emit_run_info=False)

    assert model._encoder_input.regime == "given"
    assert model._encoder_input.plan is None
    assert model._load_backend().transforms is _StandInEncoder.shipped


def test_worker_encodes_only_its_rank_shard(tmp_path, monkeypatch):
    """The torchrun entry is near logic-free: env rank → shared shard planner → shard loop."""
    import slide2vec.api as api
    from slide2vec.distributed import image_worker
    from slide2vec.runtime.serialization import serialize_execution

    specs = _images(tmp_path, ["a", "b", "c", "d"])
    loaded = _loaded(_encoder())
    declared: list = []

    monkeypatch.setattr(
        api.Model, "from_preset",
        classmethod(lambda cls, name, **kwargs: SimpleNamespace(
            name=name,
            level="tile",
            _declare_given_encoder_input=lambda *, emit_run_info: declared.append(True),
            _load_backend=lambda: (declared and loaded) or pytest.fail("must declare first"),
        )),
    )

    request = {
        "model": {"name": "fake", "output_variant": None, "allow_non_recommended_settings": False},
        "execution": serialize_execution(
            ExecutionOptions(
                output_dir=tmp_path / "out",
                num_gpus=2,
                precision="fp32",
                batch_size=2,
                num_workers_per_gpu=1,
            )
        ),
        "output_dir": str(tmp_path / "out"),
        "progress_events_path": None,
        **image_specs.build_image_specs_request(specs),
    }
    request_path = tmp_path / "image_request.json"
    request_path.write_text(json.dumps(request), encoding="utf-8")

    monkeypatch.setenv("RANK", "1")
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("LOCAL_RANK", "0")

    rc = image_worker.main(["--output-dir", str(tmp_path / "out"), "--request-path", str(request_path)])

    assert rc == 0
    # World of 2 ranks: plan_contiguous_shards([a,b,c,d], 2) => rank 1 owns [c, d].
    embeddings_dir = tmp_path / "out" / "image_embeddings"
    assert sorted(p.name for p in embeddings_dir.glob("*.pt")) == ["c.pt", "d.pt"]
