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

    def __init__(self, encoder) -> None:
        self._loaded = _loaded(encoder)
        self.name = "fake-encoder"
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
                            output_precision="fp32", num_workers=0)

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


def test_embed_images_rejects_duplicate_sample_ids(tmp_path):
    """Two images sharing a sample id would silently overwrite one another's artifact."""
    model = _FakeModel(_encoder())
    execution = ExecutionOptions(output_dir=tmp_path / "out", num_gpus=1, precision="fp32")
    specs = _images(tmp_path, ["a"])
    duplicated = [*specs, ImageSpec(sample_id="a", image_path=tmp_path / "images" / "a.png")]

    with pytest.raises(ValueError, match="duplicate sample_id"):
        image_stage.embed_images(model, duplicated, execution=execution)


def test_embed_images_requires_at_least_one_image(tmp_path):
    model = _FakeModel(_encoder())
    execution = ExecutionOptions(output_dir=tmp_path / "out", num_gpus=1, precision="fp32")
    with pytest.raises(ValueError, match="At least one image"):
        image_stage.embed_images(model, [], execution=execution)


def test_image_specs_round_trip_through_request(tmp_path):
    """The request rebuilds the exact spec list the parent sharded (the worker's inverse)."""
    specs = [
        ImageSpec(sample_id="a", image_path="/data/a.png"),
        ImageSpec(sample_id="b", image_path="/data/nested/b.tif"),
    ]
    request = image_specs.build_image_specs_request(specs)
    assert image_specs.image_specs_from_request(request) == specs


def test_model_embed_images_delegates_and_requires_output_dir(monkeypatch, tmp_path):
    """The public API coerces execution, requires an output dir, and delegates to the stage."""
    import slide2vec.runtime.image_stage as stage_mod
    from slide2vec.api import Model

    seen: dict = {}

    def _fake_stage(model, images, *, execution):
        seen.update(model=model, images=images, execution=execution)
        return ["artifact"]

    monkeypatch.setattr(stage_mod, "embed_images", _fake_stage)
    model = Model(name="virchow2")  # constructing a Model does not load weights
    specs = [ImageSpec(sample_id="a", image_path=tmp_path / "a.png")]

    with pytest.raises(ValueError):
        model.embed_images(specs, execution=ExecutionOptions(num_gpus=1))

    result = model.embed_images(specs, execution=ExecutionOptions(output_dir=tmp_path, num_gpus=1))
    assert result == ["artifact"]
    assert seen["model"] is model
    assert seen["execution"].output_dir == tmp_path


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


def test_embed_images_is_exported_from_the_package():
    import slide2vec

    assert "ImageSpec" in slide2vec.__all__
    assert "ImageEmbeddingArtifact" in slide2vec.__all__
    assert hasattr(slide2vec.Model, "embed_images")


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
