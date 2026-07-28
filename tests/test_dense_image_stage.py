"""Tests for the parent-side dense-over-images orchestration (issue #235).

``embed_images_dense`` declares the dense encoder-input contract, normalizes the caller's
images, resume-filters them, then either runs the encode/write loop in-process
(``num_gpus=1``) or fans it out over torchrun ranks (``num_gpus>1``) — reusing the same
distributed machinery the dense ROI and pooled image paths use. These tests run on CPU with a
random-weight encoder; the torchrun launch is captured rather than executed.
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

from slide2vec.api import DenseImageOptions, ExecutionOptions, ImageSpec  # noqa: E402
from slide2vec.encoders.base import TimmTileEncoder  # noqa: E402
from slide2vec.runtime import dense_image_stage  # noqa: E402
from slide2vec.runtime.types import LoadedModel  # noqa: E402


def _encoder() -> TimmTileEncoder:
    return TimmTileEncoder(
        "vit_tiny_patch16_224", pretrained=False, num_classes=0, dynamic_img_size=True
    )


def _loaded(encoder: TimmTileEncoder) -> LoadedModel:
    return LoadedModel(
        name="fake-encoder",
        level="tile",
        model=encoder,
        transforms=encoder.get_normalization_transform(),
        feature_dim=int(encoder.encode_dim),
        device=torch.device("cpu"),
    )


class _FakeModel:
    """Minimal ``Model`` stand-in: refuses a backend until the dense contract is declared."""

    def __init__(self, encoder) -> None:
        self._loaded = _loaded(encoder)
        self.name = "fake-encoder"
        self._output_variant = None
        self._requested_device = "cpu"
        self.allow_non_recommended_settings = False
        self.declared: list = []

    def _declare_dense_encoder_input(self, dense, *, emit_run_info):
        self.declared.append(dense)

    def _load_backend(self):
        assert self.declared, "the dense image path must declare its geometry before loading"
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


def _dense(**kwargs) -> DenseImageOptions:
    return DenseImageOptions(**{"target_size": 64, **kwargs})


def test_embed_images_dense_num_gpus_one_runs_in_process(tmp_path, monkeypatch):
    """``num_gpus=1`` encodes in-process — no torchrun subprocess — and writes every grid."""
    monkeypatch.setattr(
        dense_image_stage, "run_torchrun_worker",
        lambda **kwargs: pytest.fail("num_gpus=1 must not launch torchrun"),
    )
    model = _FakeModel(_encoder())
    execution = ExecutionOptions(
        output_dir=tmp_path / "out",
        num_gpus=1,
        precision="fp32",
        num_workers_per_gpu=0,
    )

    artifacts = dense_image_stage.embed_images_dense(
        model, _images(tmp_path, ["a", "b", "c"]), dense=_dense(), execution=execution
    )

    assert [artifact.sample_id for artifact in artifacts] == ["a", "b", "c"]
    assert model.declared, "the dense contract is declared before anything is read"
    dense_dir = tmp_path / "out" / "dense_image_embeddings"
    assert sorted(p.name for p in dense_dir.glob("*.pt")) == ["a.pt", "b.pt", "c.pt"]
    for artifact in artifacts:
        assert artifact.grid_shape == (4, 4)
        assert artifact.metadata_path.exists()


def test_embed_images_dense_auto_workers_complete_in_a_subprocess(
    tmp_path,
    assert_auto_worker_workflow_completes_in_subprocess,
    build_auto_worker_model,
):
    """Auto workers must not fork after the in-process encoder initialized its runtime."""
    def workflow():
        model, execution = build_auto_worker_model(_loaded(_encoder()))
        return model.embed_images_dense(
            _images(tmp_path, ["a", "b", "c"]),
            dense=_dense(),
            execution=execution,
        )

    assert_auto_worker_workflow_completes_in_subprocess(
        child_env_name="SLIDE2VEC_DENSE_IMAGE_AUTO_WORKER_CHILD",
        workflow=workflow,
        expected_sample_ids=["a", "b", "c"],
    )


def test_embed_images_dense_num_gpus_gt_one_launches_the_worker(tmp_path, monkeypatch):
    """``num_gpus>1`` launches ``slide2vec.distributed.dense_image_worker`` under torchrun;
    the parent itself encodes nothing and collects the artifacts back off disk."""
    from slide2vec.runtime.dense_image_shard import run_dense_image_shard
    from slide2vec.runtime.image_specs import image_specs_from_request
    from slide2vec.runtime.sharding import plan_contiguous_shards

    caller_dir = tmp_path / "caller"
    caller_dir.mkdir()
    monkeypatch.chdir(caller_dir)
    captured: dict = {}
    rank_loaded = _loaded(_encoder())  # every rank loads the same checkpoint

    def _fake_run(*, module, num_gpus, output_dir, request_path, **kwargs):
        request = json.loads(Path(request_path).read_text())
        captured.update(module=module, num_gpus=num_gpus, output_dir=output_dir, request=request)
        specs = image_specs_from_request(request)
        for shard in plan_contiguous_shards(specs, num_gpus):
            run_dense_image_shard(shard, loaded=rank_loaded, out_dir=Path(output_dir),
                                  dense=_dense(), batch_size=2, num_workers=0)

    monkeypatch.setattr(dense_image_stage, "run_torchrun_worker", _fake_run)
    monkeypatch.setattr(dense_image_stage, "validate_multi_gpu_execution", lambda *a, **k: None)
    monkeypatch.setattr(
        dense_image_stage, "_run_dense_images_in_process",
        lambda *a, **k: pytest.fail("num_gpus>1 must not encode in-process"),
    )
    model = _FakeModel(_encoder())
    execution = ExecutionOptions(output_dir=Path("output"), num_gpus=3, precision="fp32")

    artifacts = dense_image_stage.embed_images_dense(
        model, _images(tmp_path, ["a", "b", "c"]), dense=_dense(), execution=execution
    )

    assert captured["module"] == "slide2vec.distributed.dense_image_worker"
    assert captured["num_gpus"] == 3
    expected_output_dir = (caller_dir / "output").resolve()
    assert Path(captured["output_dir"]) == expected_output_dir
    assert captured["request"]["dense"]["target_size"] == 64
    assert [image["sample_id"] for image in captured["request"]["images"]] == ["a", "b", "c"]
    assert all(Path(image["image_path"]).is_absolute() for image in captured["request"]["images"])
    assert len(artifacts) == 3
    assert (expected_output_dir / "dense_image_embeddings" / "a.pt").exists()


def test_embed_images_dense_resume_skips_existing_and_logs(tmp_path, caplog):
    """Resume: images already on disk are filtered before dispatch; the skip count is logged."""
    model = _FakeModel(_encoder())
    execution = ExecutionOptions(output_dir=tmp_path / "out", num_gpus=1, precision="fp32")
    dense_dir = tmp_path / "out" / "dense_image_embeddings"
    first = _images(tmp_path, ["a", "b"])
    dense_image_stage.embed_images_dense(model, first, dense=_dense(), execution=execution)
    written_at = {
        spec.sample_id: (dense_dir / f"{spec.sample_id}.pt").stat().st_mtime_ns for spec in first
    }

    with caplog.at_level("INFO", logger="slide2vec.runtime.dense_image_stage"):
        artifacts = dense_image_stage.embed_images_dense(
            model, _images(tmp_path, ["a", "b", "c"]), dense=_dense(), execution=execution
        )

    assert len(artifacts) == 3
    assert "2/3 images already on disk, encoding 1" in caplog.text
    for sample_id, mtime in written_at.items():
        assert (dense_dir / f"{sample_id}.pt").stat().st_mtime_ns == mtime  # untouched


def test_embed_images_dense_all_present_does_not_dispatch(tmp_path, monkeypatch):
    """A fully-resumed run encodes nothing yet still returns every artifact."""
    model = _FakeModel(_encoder())
    execution = ExecutionOptions(output_dir=tmp_path / "out", num_gpus=1, precision="fp32")
    specs = _images(tmp_path, ["a", "b"])
    dense_image_stage.embed_images_dense(model, specs, dense=_dense(), execution=execution)

    monkeypatch.setattr(dense_image_stage, "_run_dense_images_in_process",
                        lambda *a, **k: pytest.fail("nothing to encode"))
    monkeypatch.setattr(dense_image_stage, "run_torchrun_worker",
                        lambda **k: pytest.fail("nothing to encode"))
    assert len(dense_image_stage.embed_images_dense(
        model, specs, dense=_dense(), execution=execution)) == 2


def test_embed_images_dense_rejects_duplicate_sample_ids(tmp_path):
    """Two images sharing a sample id would silently overwrite one another's grid."""
    model = _FakeModel(_encoder())
    execution = ExecutionOptions(output_dir=tmp_path / "out", num_gpus=1, precision="fp32")
    specs = _images(tmp_path, ["a"])
    duplicated = [*specs, ImageSpec(sample_id="a", image_path=tmp_path / "images" / "a.png")]

    with pytest.raises(ValueError, match="duplicate sample_id"):
        dense_image_stage.embed_images_dense(
            model, duplicated, dense=_dense(), execution=execution
        )


def test_embed_images_dense_requires_at_least_one_image(tmp_path):
    model = _FakeModel(_encoder())
    execution = ExecutionOptions(output_dir=tmp_path / "out", num_gpus=1, precision="fp32")
    with pytest.raises(ValueError, match="At least one image"):
        dense_image_stage.embed_images_dense(model, [], dense=_dense(), execution=execution)


def test_dense_image_options_round_trip_through_the_request():
    """Every dense knob crosses to the ranks, including a non-square target size."""
    from slide2vec.runtime.serialization import (
        deserialize_dense_image_options,
        serialize_dense_image_options,
    )

    dense = DenseImageOptions(
        target_size=(64, 96),
        pad_mode="constant",
        image_pad_value=0.5,
        window_size=32,
        overlap=0.25,
        feature_kind="cls_attention",
        attention_blocks=(-2, -1),
        attention_include_registers=True,
    )
    payload = json.loads(json.dumps(serialize_dense_image_options(dense)))
    assert deserialize_dense_image_options(payload) == dense


def test_model_embed_images_dense_delegates_and_requires_output_dir(monkeypatch, tmp_path):
    """The public API coerces execution, requires an output dir, and delegates to the stage."""
    import slide2vec.runtime.dense_image_stage as stage_mod
    from slide2vec.api import Model

    seen: dict = {}

    def _fake_stage(model, images, *, dense, execution):
        seen.update(model=model, images=images, dense=dense, execution=execution)
        return ["artifact"]

    monkeypatch.setattr(stage_mod, "embed_images_dense", _fake_stage)
    model = Model(name="virchow2")  # constructing a Model does not load weights
    specs = [ImageSpec(sample_id="a", image_path=tmp_path / "a.png")]
    dense = _dense(target_size=224)

    with pytest.raises(ValueError):
        model.embed_images_dense(specs, dense=dense, execution=ExecutionOptions(num_gpus=1))

    result = model.embed_images_dense(
        specs, dense=dense, execution=ExecutionOptions(output_dir=tmp_path, num_gpus=1)
    )
    assert result == ["artifact"]
    assert seen["model"] is model
    assert seen["dense"] is dense
    assert seen["execution"].output_dir == tmp_path


def test_declaring_the_dense_contract_rejects_a_geometry_the_encoder_cannot_take(tmp_path):
    """The #233 capability check runs before any image is decoded: phikon is fixed-input."""
    from slide2vec.api import Model

    model = Model(name="phikon", device="cpu")
    execution = ExecutionOptions(output_dir=tmp_path / "out", num_gpus=1, precision="fp32")

    with pytest.raises(ValueError, match="does not support a variable encoder input"):
        dense_image_stage.embed_images_dense(
            model, _images(tmp_path, ["a"]), dense=_dense(target_size=512), execution=execution
        )


def test_declared_dense_contract_accepts_a_non_square_image_geometry():
    """A non-square declared geometry is checked per dimension, not rejected for not being
    square — the padded image, not a transform-normalised square, is the encoder input."""
    from slide2vec.runtime.encoder_input_contract import EncoderInputContract

    contract = EncoderInputContract.declared_dense(
        "virchow2", target_size_px=(224, 448), window_size=None
    )

    assert contract.plan.target_size_px == (224, 448)
    assert contract.plan.effective_encoder_input_size_px == (224, 448)
    assert contract.plan.requires_variable_model_input is True


def test_embed_images_dense_is_exported_from_the_package():
    import slide2vec

    assert "DenseImageOptions" in slide2vec.__all__
    assert "DenseImageArtifact" in slide2vec.__all__
    assert hasattr(slide2vec.Model, "embed_images_dense")


def test_worker_encodes_only_its_rank_shard(tmp_path, monkeypatch):
    """The torchrun entry is near logic-free: env rank → shared shard planner → shard loop."""
    import slide2vec.api as api
    from slide2vec.distributed import dense_image_worker
    from slide2vec.runtime.image_specs import build_image_specs_request
    from slide2vec.runtime.serialization import (
        serialize_dense_image_options,
        serialize_execution,
    )

    specs = _images(tmp_path, ["a", "b", "c", "d"])
    loaded = _loaded(_encoder())
    declared: list = []

    monkeypatch.setattr(
        api.Model, "from_preset",
        classmethod(lambda cls, name, **kwargs: SimpleNamespace(
            _declare_dense_encoder_input=lambda dense, *, emit_run_info: declared.append(dense),
            _load_backend=lambda: (declared and loaded) or pytest.fail("must declare first"),
        )),
    )

    request = {
        "model": {"name": "fake", "output_variant": None, "allow_non_recommended_settings": False},
        "dense": serialize_dense_image_options(_dense()),
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
        **build_image_specs_request(specs),
    }
    request_path = tmp_path / "dense_image_request.json"
    request_path.write_text(json.dumps(request), encoding="utf-8")

    monkeypatch.setenv("RANK", "1")
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("LOCAL_RANK", "0")

    rc = dense_image_worker.main(
        ["--output-dir", str(tmp_path / "out"), "--request-path", str(request_path)]
    )

    assert rc == 0
    assert declared, "each rank declares the dense contract for itself"
    # World of 2 ranks: plan_contiguous_shards([a,b,c,d], 2) => rank 1 owns [c, d].
    dense_dir = tmp_path / "out" / "dense_image_embeddings"
    assert sorted(p.name for p in dense_dir.glob("*.pt")) == ["c.pt", "d.pt"]
