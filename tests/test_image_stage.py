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

    #: The latest model built per name: what a mocked torchrun rank loads for it.
    latest: dict = {}

    def __init__(self, encoder, *, name: str = "fake-encoder") -> None:
        self._loaded = _loaded(encoder)
        self.name = name
        self.level = "tile"
        self._output_variant = None
        self._requested_device = "cpu"
        self._encoder_input = None
        self.allow_non_recommended_settings = False
        self.declared_given = False
        self.loads = 0
        _FakeModel.latest[name] = self

    @property
    def feature_dim(self):
        # Loads the backend and may describe slide output: collection must not use it.
        raise AssertionError("embed_images must not read Model.feature_dim")

    def _declare_given_encoder_input(self, *, emit_run_info):
        self.declared_given = True

    def _load_backend(self):
        assert self.declared_given, "the image path must declare Given before it loads"
        self.loads += 1
        return self._loaded


@pytest.fixture(params=[1, 2], ids=["in-process", "torchrun"])
def num_gpus(request, monkeypatch):
    """Run ``embed_images`` in-process, or through mocked torchrun ranks.

    The mocked launch runs the real ``image_worker.main`` once per rank, in this process,
    with each rank loading the latest :class:`_FakeModel` of the requested name.
    """
    if request.param == 1:
        return 1
    import slide2vec.api as api
    from slide2vec.distributed import image_worker

    def rank_model(name, **kwargs):
        parent = _FakeModel.latest[name]

        def load_backend():
            parent.rank_loads = getattr(parent, "rank_loads", 0) + 1
            return parent._loaded

        return SimpleNamespace(
            name=name,
            level="tile",
            _output_variant=None,
            _encoder_input=None,
            allow_non_recommended_settings=False,
            _declare_given_encoder_input=lambda *, emit_run_info: None,
            _load_backend=load_backend,
        )

    def fake_torchrun(*, module, num_gpus, output_dir, request_path, **kwargs):
        assert module == "slide2vec.distributed.image_worker"
        for rank in range(num_gpus):
            with monkeypatch.context() as env:
                env.setenv("RANK", str(rank))
                env.setenv("WORLD_SIZE", str(num_gpus))
                env.setenv("LOCAL_RANK", str(rank))
                assert image_worker.main(
                    ["--output-dir", str(output_dir), "--request-path", str(request_path)]
                ) == 0

    monkeypatch.setattr(
        api.Model, "from_preset", classmethod(lambda cls, name, **kwargs: rank_model(name))
    )
    monkeypatch.setattr(image_stage, "run_torchrun_worker", fake_torchrun)
    monkeypatch.setattr(image_stage, "validate_multi_gpu_execution", lambda *a, **k: None)
    monkeypatch.setattr(
        image_stage, "_run_images_in_process",
        lambda *a, **k: pytest.fail("num_gpus>1 must not encode in-process"),
    )
    return request.param


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
        for rank, shard in enumerate(plan_contiguous_shards(specs, num_gpus)):
            artifacts = run_image_shard(
                shard, loaded=rank_loaded, out_dir=Path(output_dir), batch_size=2,
                output_precision="fp32", num_workers=0,
                identity={"encoder_name": "fake-encoder"},
            )
            Path(request["result_dir"], f"image_result.rank{rank}.json").write_text(
                json.dumps({"feature_dim": artifacts[0].feature_dim}), encoding="utf-8"
            )

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


def _edit_sidecar(tmp_path, sample_id: str, edit) -> None:
    path = tmp_path / "out" / "image_embeddings" / f"{sample_id}.meta.json"
    metadata = json.loads(path.read_text(encoding="utf-8"))
    edit(metadata)
    path.write_text(json.dumps(metadata), encoding="utf-8")


def _payload(tmp_path, sample_id: str, output_format: str = "pt"):
    path = tmp_path / "out" / "image_embeddings" / f"{sample_id}.{output_format}"
    if output_format == "pt":
        return torch.load(path, weights_only=True)
    return torch.from_numpy(np.load(path)["features"])


def test_embed_images_reencodes_sidecars_without_an_identity_for_another_encoder(tmp_path, num_gpus):
    """No recorded identity, so nothing proves the old features match the new encoder."""
    specs = _images(tmp_path, ["a", "b"])
    torch.manual_seed(0)
    image_stage.embed_images(_FakeModel(_encoder()), specs, execution=_execution(tmp_path, num_gpus=num_gpus))
    old = {spec.sample_id: _payload(tmp_path, spec.sample_id) for spec in specs}
    for spec in specs:
        _edit_sidecar(tmp_path, spec.sample_id, lambda metadata: metadata.pop("compatibility"))
    torch.manual_seed(1)
    other = _FakeModel(_encoder(), name="other-encoder")

    artifacts = image_stage.embed_images(other, specs, execution=_execution(tmp_path, num_gpus=num_gpus))

    assert [artifact.sample_id for artifact in artifacts] == ["a", "b"]
    for spec in specs:
        assert not torch.equal(_payload(tmp_path, spec.sample_id), old[spec.sample_id])
        assert _sidecar(tmp_path, spec.sample_id)["compatibility"]["encoder_name"] == (
            "other-encoder"
        )


@pytest.mark.parametrize(
    "edit",
    [
        pytest.param(lambda metadata: metadata["compatibility"].pop("transform"), id="transform"),
        pytest.param(
            lambda metadata: metadata["compatibility"].pop("feature_dtype"), id="feature-dtype"
        ),
        pytest.param(lambda metadata: metadata.pop("image_path"), id="image-path"),
        pytest.param(lambda metadata: metadata.update(image_path=""), id="empty-image-path"),
        pytest.param(lambda metadata: metadata.pop("format"), id="format"),
    ],
)
def test_embed_images_reencodes_an_image_with_missing_provenance(tmp_path, num_gpus, edit):
    specs = _images(tmp_path, ["a", "b"])
    torch.manual_seed(0)
    image_stage.embed_images(_FakeModel(_encoder()), specs, execution=_execution(tmp_path, num_gpus=num_gpus))
    old_a, old_b = _payload(tmp_path, "a"), _payload(tmp_path, "b")
    _edit_sidecar(tmp_path, "a", edit)
    torch.manual_seed(1)  # same identity, different weights: a re-encode is observable

    image_stage.embed_images(_FakeModel(_encoder()), specs, execution=_execution(tmp_path, num_gpus=num_gpus))

    assert not torch.equal(_payload(tmp_path, "a"), old_a)
    assert torch.equal(_payload(tmp_path, "b"), old_b)
    sidecar = _sidecar(tmp_path, "a")
    assert sidecar["image_path"] == str(specs[0].image_path)
    assert sidecar["format"] == "pt"
    assert sidecar["compatibility"]["transform"] == SHIPPED_TRANSFORM


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
        "result_dir": str(tmp_path),
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
    assert json.loads((tmp_path / "image_result.rank1.json").read_text()) == {
        "feature_dim": loaded.model.encode_dim,
        "images": 2,
    }


# --------------------------------------------------------------------------------------
# Source provenance, format-specific completion and recovery (issue #364)
# --------------------------------------------------------------------------------------


def _image(tmp_path, name: str, *, seed: int) -> Path:
    path = tmp_path / "images" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    pixels = np.random.default_rng(seed).integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
    Image.fromarray(pixels).save(path)
    return path


def _execution(tmp_path, *, num_gpus=1, output_format="pt", **kwargs) -> ExecutionOptions:
    kwargs.setdefault("num_workers_per_gpu", 0)
    return ExecutionOptions(
        output_dir=tmp_path / "out", num_gpus=num_gpus, precision="fp32",
        output_format=output_format, **kwargs,
    )


def test_embed_images_refuses_a_sample_repointed_to_another_image(tmp_path, num_gpus):
    image_a = _image(tmp_path, "a.png", seed=1)
    image_b = _image(tmp_path, "b.png", seed=2)
    model = _FakeModel(_encoder())
    image_stage.embed_images(model, [ImageSpec(sample_id="s", image_path=image_a)],
                             execution=_execution(tmp_path, num_gpus=num_gpus))

    with pytest.raises(ValueError) as error:
        image_stage.embed_images(model, [ImageSpec(sample_id="s", image_path=image_b)],
                                 execution=_execution(tmp_path, num_gpus=num_gpus))

    message = str(error.value)
    assert "'s'" in message
    assert str(image_a) in message
    assert str(image_b) in message


def test_embed_images_refuses_a_repointed_sample_when_the_format_changes_too(tmp_path, num_gpus):
    """The sidecar is shared by both payload formats, so a format switch is no bypass."""
    image_a = _image(tmp_path, "a.png", seed=1)
    image_b = _image(tmp_path, "b.png", seed=2)
    model = _FakeModel(_encoder())
    image_stage.embed_images(model, [ImageSpec(sample_id="s", image_path=image_a)],
                             execution=_execution(tmp_path, num_gpus=num_gpus, output_format="pt"))

    with pytest.raises(ValueError) as error:
        image_stage.embed_images(model, [ImageSpec(sample_id="s", image_path=image_b)],
                                 execution=_execution(tmp_path, num_gpus=num_gpus, output_format="npz"))

    assert str(image_a) in str(error.value)
    assert str(image_b) in str(error.value)
    assert not (tmp_path / "out" / "image_embeddings" / "s.npz").exists()


@pytest.mark.parametrize("policy", ["warn", "", None, "RAISE"])
def test_execution_options_reject_an_unknown_image_mismatch_policy(policy):
    with pytest.raises(ValueError, match="on_image_mismatch"):
        ExecutionOptions(num_gpus=1, on_image_mismatch=policy)


def test_image_mismatch_policy_crosses_to_the_torchrun_ranks():
    from slide2vec.runtime.serialization import deserialize_execution, serialize_execution

    execution = ExecutionOptions(num_gpus=1, on_image_mismatch="reencode")

    assert ExecutionOptions(num_gpus=1).on_image_mismatch == "raise"
    assert deserialize_execution(serialize_execution(execution)).on_image_mismatch == "reencode"


def _reference_embedding(tmp_path, model, image_path):
    """The image's embedding computed from scratch in its own output directory."""
    from slide2vec.runtime.image_shard import run_image_shard

    [artifact] = run_image_shard(
        [ImageSpec(sample_id="reference", image_path=str(image_path))],
        loaded=model._loaded, out_dir=tmp_path / "reference", batch_size=1,
        output_precision="fp32", identity={}, num_workers=0,
    )
    return torch.load(artifact.path, weights_only=True)


def test_reencode_policy_replaces_a_repointed_sample_and_skips_the_rest(tmp_path, num_gpus):
    image_a = _image(tmp_path, "a.png", seed=1)
    image_b = _image(tmp_path, "b.png", seed=2)
    image_c = _image(tmp_path, "c.png", seed=3)
    model = _FakeModel(_encoder())
    image_stage.embed_images(
        model,
        [ImageSpec(sample_id="s", image_path=image_a), ImageSpec(sample_id="t", image_path=image_c)],
        execution=_execution(tmp_path, num_gpus=num_gpus),
    )
    t_written_at = (tmp_path / "out" / "image_embeddings" / "t.pt").stat().st_mtime_ns

    artifacts = image_stage.embed_images(
        model,
        [ImageSpec(sample_id="s", image_path=image_b), ImageSpec(sample_id="t", image_path=image_c)],
        execution=_execution(tmp_path, num_gpus=num_gpus, on_image_mismatch="reencode"),
    )

    assert [artifact.sample_id for artifact in artifacts] == ["s", "t"]
    torch.testing.assert_close(
        _payload(tmp_path, "s"), _reference_embedding(tmp_path, model, image_b)
    )
    assert _sidecar(tmp_path, "s")["image_path"] == str(image_b)
    assert (tmp_path / "out" / "image_embeddings" / "t.pt").stat().st_mtime_ns == t_written_at


def test_reencode_policy_tracks_the_source_across_format_switches(tmp_path, num_gpus):
    """A/PT -> B/NPZ -> B/PT: the final PT holds B, never the stale A payload."""
    image_a = _image(tmp_path, "a.png", seed=1)
    image_b = _image(tmp_path, "b.png", seed=2)
    model = _FakeModel(_encoder())

    def run(image, output_format):
        return image_stage.embed_images(
            model,
            [ImageSpec(sample_id="s", image_path=image)],
            execution=_execution(
                tmp_path, num_gpus=num_gpus, output_format=output_format, on_image_mismatch="reencode"
            ),
        )

    run(image_a, "pt")
    run(image_b, "npz")
    assert not (tmp_path / "out" / "image_embeddings" / "s.pt").exists()
    [artifact] = run(image_b, "pt")

    assert artifact.format == "pt"
    torch.testing.assert_close(
        torch.load(artifact.path, weights_only=True),
        _reference_embedding(tmp_path, model, image_b),
    )
    assert _sidecar(tmp_path, "s")["format"] == "pt"
    assert _sidecar(tmp_path, "s")["image_path"] == str(image_b)


def test_a_later_validation_error_deletes_no_earlier_artifact(tmp_path, num_gpus):
    """Validation finishes before anything is invalidated."""
    image_a = _image(tmp_path, "a.png", seed=1)
    image_b = _image(tmp_path, "b.png", seed=2)
    model = _FakeModel(_encoder())
    image_stage.embed_images(
        model,
        [ImageSpec(sample_id="s", image_path=image_a), ImageSpec(sample_id="t", image_path=image_a)],
        execution=_execution(tmp_path, num_gpus=num_gpus),
    )
    embeddings_dir = tmp_path / "out" / "image_embeddings"
    before = {path.name: path.read_bytes() for path in embeddings_dir.iterdir()}

    with pytest.raises(ValueError, match="Cannot resume 't'"):
        image_stage.embed_images(
            model,
            # s needs a new format; t is repointed, which raises under the default policy.
            [ImageSpec(sample_id="s", image_path=image_a), ImageSpec(sample_id="t", image_path=image_b)],
            execution=_execution(tmp_path, num_gpus=num_gpus, output_format="npz"),
        )

    assert {path.name: path.read_bytes() for path in embeddings_dir.iterdir()} == before


def test_a_rejected_multi_gpu_request_deletes_no_earlier_artifact(tmp_path, monkeypatch):
    """Multi-GPU request validation also runs before anything is invalidated."""
    image_a = _image(tmp_path, "a.png", seed=1)
    image_b = _image(tmp_path, "b.png", seed=2)
    model = _FakeModel(_encoder())
    image_stage.embed_images(
        model,
        [ImageSpec(sample_id="s", image_path=image_a), ImageSpec(sample_id="t", image_path=image_a)],
        execution=_execution(tmp_path),
    )
    embeddings_dir = tmp_path / "out" / "image_embeddings"
    before = {path.name: path.read_bytes() for path in embeddings_dir.iterdir()}

    def reject(model, execution):
        raise ValueError("ExecutionOptions.num_gpus=8 exceeds available CUDA devices (4)")

    monkeypatch.setattr(image_stage, "validate_multi_gpu_execution", reject)
    monkeypatch.setattr(
        image_stage, "run_torchrun_worker",
        lambda **kwargs: pytest.fail("a rejected request must not launch torchrun"),
    )
    with pytest.raises(ValueError, match="exceeds available CUDA devices"):
        image_stage.embed_images(
            model,
            # s needs a new format; t is repointed and would be replaced.
            [ImageSpec(sample_id="s", image_path=image_a), ImageSpec(sample_id="t", image_path=image_b)],
            execution=_execution(
                tmp_path, num_gpus=8, output_format="npz", on_image_mismatch="reencode"
            ),
        )

    assert {path.name: path.read_bytes() for path in embeddings_dir.iterdir()} == before


@pytest.mark.parametrize("workers", [-1, -4])
def test_execution_options_reject_a_negative_worker_count(workers):
    with pytest.raises(ValueError, match="num_workers_per_gpu"):
        ExecutionOptions(num_gpus=1, num_workers_per_gpu=workers)


def test_invalid_loader_settings_delete_no_earlier_artifact(tmp_path, num_gpus):
    """Loader settings are request validation too: they fail before any invalidation."""
    image_a = _image(tmp_path, "a.png", seed=1)
    model = _FakeModel(_encoder())
    image_stage.embed_images(model, [ImageSpec(sample_id="s", image_path=image_a)],
                             execution=_execution(tmp_path, num_gpus=num_gpus))
    embeddings_dir = tmp_path / "out" / "image_embeddings"
    before = {path.name: path.read_bytes() for path in embeddings_dir.iterdir()}

    with pytest.raises(ValueError, match="num_workers_per_gpu"):
        image_stage.embed_images(
            model,
            # a format switch would invalidate the PT sidecar
            [ImageSpec(sample_id="s", image_path=image_a)],
            execution=_execution(
                tmp_path, num_gpus=num_gpus, output_format="npz", num_workers_per_gpu=-1
            ),
        )

    assert {path.name: path.read_bytes() for path in embeddings_dir.iterdir()} == before


@pytest.mark.parametrize(
    "request_kind", ["format-switch", "reencode-to-missing-source", "reencode-to-directory-source"]
)
def test_a_missing_source_deletes_no_earlier_artifact(tmp_path, num_gpus, request_kind):
    """An image is never invalidated for a source that cannot be read to replace it."""
    image_a = _image(tmp_path, "a.png", seed=1)
    image_c = _image(tmp_path, "c.png", seed=3)
    model = _FakeModel(_encoder())
    image_stage.embed_images(
        model,
        [ImageSpec(sample_id="s", image_path=image_a), ImageSpec(sample_id="t", image_path=image_c)],
        execution=_execution(tmp_path, num_gpus=num_gpus),
    )
    embeddings_dir = tmp_path / "out" / "image_embeddings"
    before = {path.name: path.read_bytes() for path in embeddings_dir.iterdir()}
    if request_kind == "format-switch":
        image_a.unlink()  # e.g. a scratch copy cleaned up after extraction
        missing, options = image_a, {"output_format": "npz"}
    elif request_kind == "reencode-to-missing-source":
        missing, options = tmp_path / "images" / "b.png", {"on_image_mismatch": "reencode"}
    else:
        missing = tmp_path / "images" / "adir"  # exists, but PIL can never open it
        missing.mkdir()
        options = {"on_image_mismatch": "reencode"}

    with pytest.raises(FileNotFoundError) as error:
        image_stage.embed_images(
            model,
            [ImageSpec(sample_id="s", image_path=missing), ImageSpec(sample_id="t", image_path=image_c)],
            execution=_execution(tmp_path, num_gpus=num_gpus, **options),
        )

    assert "'s'" in str(error.value)
    assert str(missing) in str(error.value)
    assert "'t'" not in str(error.value)
    assert {path.name: path.read_bytes() for path in embeddings_dir.iterdir()} == before


def test_a_missing_new_source_deletes_no_earlier_artifact(tmp_path, num_gpus):
    """A new image that cannot be read must not cost another image its artifacts."""
    image_a = _image(tmp_path, "a.png", seed=1)
    model = _FakeModel(_encoder())
    image_stage.embed_images(model, [ImageSpec(sample_id="s", image_path=image_a)],
                             execution=_execution(tmp_path, num_gpus=num_gpus))
    embeddings_dir = tmp_path / "out" / "image_embeddings"
    before = {path.name: path.read_bytes() for path in embeddings_dir.iterdir()}
    missing = tmp_path / "images" / "missing.png"

    with pytest.raises(FileNotFoundError) as error:
        image_stage.embed_images(
            model,
            # s needs a new format, which invalidates its sidecar; "new" has no artifacts.
            [ImageSpec(sample_id="new", image_path=missing),
             ImageSpec(sample_id="s", image_path=image_a)],
            execution=_execution(tmp_path, num_gpus=num_gpus, output_format="npz"),
        )

    assert "'new'" in str(error.value)
    assert str(missing) in str(error.value)
    assert {path.name: path.read_bytes() for path in embeddings_dir.iterdir()} == before


@pytest.mark.parametrize("spelling", ["PT", "NPZ", "Npz"])
def test_an_uppercase_output_format_resumes_its_own_artifacts(tmp_path, num_gpus, monkeypatch, spelling):
    """Format spelling is canonicalized once, so a run reuses what it wrote."""
    image_a = _image(tmp_path, "a.png", seed=1)
    model = _FakeModel(_encoder())
    run = lambda: image_stage.embed_images(  # noqa: E731
        model, [ImageSpec(sample_id="s", image_path=image_a)],
        execution=_execution(tmp_path, num_gpus=num_gpus, output_format=spelling),
    )
    [first] = run()
    canonical = spelling.lower()
    assert first.format == canonical
    assert first.path.name == f"s.{canonical}"
    assert _sidecar(tmp_path, "s")["format"] == canonical

    monkeypatch.setattr(image_stage, "_run_images_in_process",
                        lambda *a, **k: pytest.fail("nothing to encode"))
    monkeypatch.setattr(image_stage, "run_torchrun_worker",
                        lambda **k: pytest.fail("nothing to encode"))
    [second] = run()

    assert second.format == canonical
    assert second.path == first.path


def _fail_once(monkeypatch, module, name):
    original = getattr(module, name)
    calls = {"count": 0}

    def fail_first(*args, **kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            raise OSError(f"simulated failure in {name}")
        return original(*args, **kwargs)

    monkeypatch.setattr(module, name, fail_first)


@pytest.mark.parametrize(
    ("module_name", "function"),
    [
        pytest.param("slide2vec.artifacts", "_write_metadata", id="before-sidecar"),
        pytest.param("torch", "save", id="during-payload"),
    ],
)
@pytest.mark.parametrize("output_format", ["pt", "npz"])
def test_an_interrupted_replacement_leaves_the_image_incomplete(
    tmp_path, num_gpus, monkeypatch, module_name, function, output_format
):
    """No old sidecar ever certifies a replacement payload; the next run recovers."""
    import importlib

    image_a = _image(tmp_path, "a.png", seed=1)
    image_b = _image(tmp_path, "b.png", seed=2)
    model = _FakeModel(_encoder())
    image_stage.embed_images(model, [ImageSpec(sample_id="s", image_path=image_a)],
                             execution=_execution(tmp_path, num_gpus=num_gpus))
    run_b = lambda: image_stage.embed_images(  # noqa: E731
        model, [ImageSpec(sample_id="s", image_path=image_b)],
        execution=_execution(
            tmp_path, num_gpus=num_gpus, output_format=output_format,
            on_image_mismatch="reencode",
        ),
    )
    if function == "save" and output_format == "npz":
        module_name, function = "numpy", "savez_compressed"
    with monkeypatch.context() as patch:
        _fail_once(patch, importlib.import_module(module_name), function)
        with pytest.raises(OSError, match="simulated failure"):
            run_b()

    assert not (tmp_path / "out" / "image_embeddings" / "s.meta.json").exists()
    [artifact] = run_b()
    assert _sidecar(tmp_path, "s")["image_path"] == str(image_b)
    torch.testing.assert_close(
        _payload(tmp_path, "s", output_format).to(torch.float32),
        _reference_embedding(tmp_path, model, image_b),
    )
    assert artifact.format == output_format


class _FileSystemSpy:
    """Counts the filesystem calls ``embed_images`` makes under the artifact directory."""

    def __init__(self, monkeypatch, out_dir: Path) -> None:
        import os

        self.out_dir = str(out_dir)
        self.sidecar_reads = 0
        self.listings = 0
        self.probes = 0
        self.resolves = 0
        original_read_text = Path.read_text
        original_listdir = os.listdir
        original_exists = Path.exists
        original_is_file = Path.is_file
        original_resolve = Path.resolve
        spy = self

        def read_text(path, *args, **kwargs):
            spy.sidecar_reads += path.name.endswith(".meta.json")
            return original_read_text(path, *args, **kwargs)

        def listdir(path="."):
            spy.listings += str(path).endswith("image_embeddings")
            return original_listdir(path)

        def probe(original):
            def wrapped(path, *args, **kwargs):
                spy.probes += "image_embeddings" in str(path)
                return original(path, *args, **kwargs)
            return wrapped

        def resolve(path, *args, **kwargs):
            spy.resolves += str(path).startswith(spy.out_dir)
            return original_resolve(path, *args, **kwargs)

        monkeypatch.setattr(Path, "read_text", read_text)
        monkeypatch.setattr(os, "listdir", listdir)
        monkeypatch.setattr(Path, "exists", probe(original_exists))
        monkeypatch.setattr(Path, "is_file", probe(original_is_file))
        monkeypatch.setattr(Path, "resolve", resolve)


def test_collection_reads_no_sidecar_and_keeps_input_order(tmp_path, num_gpus, monkeypatch):
    model = _FakeModel(_encoder())
    width = int(model._loaded.model.encode_dim)
    specs = {spec.sample_id: spec for spec in _images(tmp_path, ["a", "b", "c"])}
    with monkeypatch.context() as patch:
        first_run = _FileSystemSpy(patch, tmp_path / "out")
        image_stage.embed_images(
            model, [specs["a"], specs["b"]], execution=_execution(tmp_path, num_gpus=num_gpus)
        )
    _edit_sidecar(tmp_path, "b", lambda metadata: metadata.update(feature_dim=7))

    with monkeypatch.context() as patch:
        resumed_run = _FileSystemSpy(patch, tmp_path / "out")
        artifacts = image_stage.embed_images(
            model,
            [specs["c"], specs["b"], specs["a"]],
            execution=_execution(tmp_path, num_gpus=num_gpus),
        )

    assert first_run.sidecar_reads == 0
    assert resumed_run.sidecar_reads == 2  # resume only: one read per reused image
    assert [(a.sample_id, a.feature_dim, a.format) for a in artifacts] == [
        ("c", width, "pt"), ("b", 7, "pt"), ("a", width, "pt"),
    ]
    embeddings_dir = (tmp_path / "out" / "image_embeddings").resolve()
    for artifact in artifacts:
        assert artifact.path == embeddings_dir / f"{artifact.sample_id}.pt"
        assert artifact.metadata_path == embeddings_dir / f"{artifact.sample_id}.meta.json"
        assert artifact.metadata["sample_id"] == artifact.sample_id


def test_collection_reports_the_image_vector_width_without_loading(tmp_path, num_gpus):
    """A slide encoder's slide width (PRISM: 1280) is not its image-vector width (2560)."""
    model = _FakeModel(_encoder())
    width = int(model._loaded.model.encode_dim)
    model._loaded.feature_dim = width + 1000  # the slide output, never an image vector
    specs = _images(tmp_path, ["a", "b", "c"])

    artifacts = image_stage.embed_images(
        model, specs, execution=_execution(tmp_path, num_gpus=num_gpus)
    )

    assert [artifact.feature_dim for artifact in artifacts] == [width] * 3
    assert model.loads == (1 if num_gpus == 1 else 0)  # encoding only, never collection


def test_resume_lists_the_artifacts_once_and_probes_no_image(tmp_path, num_gpus, monkeypatch):
    model = _FakeModel(_encoder())
    resolves = []
    for names in (["a"], ["a", "b", "c", "d"]):
        specs = _images(tmp_path, names)
        image_stage.embed_images(model, specs, execution=_execution(tmp_path, num_gpus=num_gpus))
        with monkeypatch.context() as patch:
            spy = _FileSystemSpy(patch, tmp_path / "out")
            image_stage.embed_images(
                model, specs, execution=_execution(tmp_path, num_gpus=num_gpus)
            )
        assert spy.listings == 1
        assert spy.probes == 0
        resolves.append(spy.resolves)

    assert resolves[0] == resolves[1]  # output-root resolution does not scale with images


@pytest.mark.parametrize("removed", ["s.meta.json", "s.pt"], ids=["no-sidecar", "no-payload"])
def test_an_incomplete_pair_is_reencoded(tmp_path, num_gpus, removed):
    image_a = _image(tmp_path, "a.png", seed=1)
    model = _FakeModel(_encoder())
    run = lambda: image_stage.embed_images(  # noqa: E731
        model, [ImageSpec(sample_id="s", image_path=image_a)],
        execution=_execution(tmp_path, num_gpus=num_gpus),
    )
    run()
    (tmp_path / "out" / "image_embeddings" / removed).unlink()

    [artifact] = run()

    assert artifact.path.exists()
    assert artifact.metadata_path.exists()
    assert _sidecar(tmp_path, "s")["format"] == "pt"


def _drop_format(tmp_path):
    _edit_sidecar(tmp_path, "s", lambda metadata: metadata.pop("format"))


def _unknown_format(tmp_path):
    _edit_sidecar(tmp_path, "s", lambda metadata: metadata.update(format="h5"))


def _drop_sidecar(tmp_path):
    (tmp_path / "out" / "image_embeddings" / "s.meta.json").unlink()


@pytest.mark.parametrize(
    ("lose_provenance", "next_image"),
    [
        pytest.param(_drop_sidecar, "b.png", id="no-sidecar"),
        pytest.param(_drop_format, "a.png", id="no-format"),
        pytest.param(_unknown_format, "a.png", id="unknown-format"),
    ],
)
def test_incomplete_provenance_removes_every_payload_variant(
    tmp_path, num_gpus, lose_provenance, next_image
):
    """No payload of unknown provenance survives the image's re-encode, in any format."""
    image_a = _image(tmp_path, "a.png", seed=1)
    _image(tmp_path, "b.png", seed=2)
    model = _FakeModel(_encoder())

    def run(image, output_format):
        return image_stage.embed_images(
            model, [ImageSpec(sample_id="s", image_path=image)],
            execution=_execution(tmp_path, num_gpus=num_gpus, output_format=output_format),
        )

    run(image_a, "pt")
    run(image_a, "npz")  # s.pt stays on disk; the sidecar now certifies s.npz
    lose_provenance(tmp_path)
    image = tmp_path / "images" / next_image

    [artifact] = run(image, "pt")

    embeddings_dir = tmp_path / "out" / "image_embeddings"
    assert sorted(path.name for path in embeddings_dir.iterdir()) == ["s.meta.json", "s.pt"]
    assert _sidecar(tmp_path, "s")["image_path"] == str(image)
    torch.testing.assert_close(
        torch.load(artifact.path, weights_only=True), _reference_embedding(tmp_path, model, image)
    )


def test_completion_is_specific_to_the_requested_format(tmp_path, num_gpus):
    """The shared sidecar certifies only the format it records."""
    image_a = _image(tmp_path, "a.png", seed=1)
    model = _FakeModel(_encoder())

    def run(output_format):
        [artifact] = image_stage.embed_images(
            model, [ImageSpec(sample_id="s", image_path=image_a)],
            execution=_execution(tmp_path, num_gpus=num_gpus, output_format=output_format),
        )
        return artifact

    run("pt")
    npz = run("npz")
    assert npz.format == "npz"
    assert _sidecar(tmp_path, "s")["format"] == "npz"
    npz_written_at = npz.path.stat().st_mtime_ns

    pt = run("pt")  # a.pt is still on disk, but the sidecar certifies the npz payload

    assert pt.format == "pt"
    assert _sidecar(tmp_path, "s")["format"] == "pt"
    torch.testing.assert_close(
        torch.load(pt.path, weights_only=True), _reference_embedding(tmp_path, model, image_a)
    )
    assert npz.path.stat().st_mtime_ns == npz_written_at
