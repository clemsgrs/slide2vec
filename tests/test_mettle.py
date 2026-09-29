"""Tests for the Mettle tile encoder.

Mettle (``slideflow-labs/Mettle``) is H-optimus-0's ViT-g/14 reg4 backbone,
fine-tuned, plus a small ``MettleRefineHead`` applied to the CLS token. It ships
as a ``trust_remote_code`` HF repo; slide2vec rebuilds it from timm + a vendored
head, so these tests pin the contracts that break silently (no exception, just
different features):

* ``cls`` is the *headed* CLS token; ``cls_patch_mean`` appends the *unheaded*
  mean of the spatial patch tokens, with CLS and the 4 register tokens excluded;
* the head and the patch mean run in fp32 outside autocast (upstream casts to
  ``float32`` explicitly), so a fp16 run still returns fp32 features.

Light tests run a tiny random reg4 ViT offline. Heavy tests load the real
weights and compare against upstream ``AutoModel(trust_remote_code=True)``.
The transform contracts (H-optimus mean/std, not the generic arch's ImageNet
``pretrained_cfg``) live in ``test_encoder_preprocessing.py``.
"""

from __future__ import annotations

import pytest
import timm
import torch

_IMAGE = 28  # -> 2x2 patch grid, 1 CLS + 4 registers + 4 patches
_DIM = 384


def _tiny_encoder(*, output_variant: str = "cls_patch_mean"):
    """A tiny random reg4 ViT + refine head wired into Mettle (no download)."""
    from slide2vec.encoders.models.mettle import Mettle, MettleRefineHead

    torch.manual_seed(0)
    backbone = timm.create_model(
        "vit_small_patch14_reg4_dinov2",
        pretrained=False,
        num_classes=0,
        init_values=1e-5,
        img_size=_IMAGE,
    ).eval()
    head = MettleRefineHead(dim=_DIM, num_atoms=4, rank=2, hidden_size=16).eval()
    with torch.no_grad():  # upstream initializes P/Q to zero; make the head non-trivial
        head.P.normal_()
        head.Q.normal_()

    encoder = Mettle.__new__(Mettle)
    encoder._model = backbone
    encoder._head = head
    encoder._device = torch.device("cpu")
    encoder._output_variant = output_variant
    return encoder


def test_cls_patch_mean_is_headed_cls_then_unheaded_spatial_patch_mean():
    encoder = _tiny_encoder()
    batch = torch.randn(2, 3, _IMAGE, _IMAGE)
    with torch.inference_mode():
        tokens = encoder._model.forward_features(batch)
        pooled = encoder.encode_tiles(batch)
        headed_cls = encoder._head(tokens[:, 0])

    assert tokens.shape == (2, 9, _DIM)  # CLS + 4 registers + 2x2 patches
    assert pooled.shape == (2, 2 * _DIM)
    torch.testing.assert_close(pooled[:, :_DIM], headed_cls, rtol=0, atol=0)
    torch.testing.assert_close(pooled[:, _DIM:], tokens[:, 5:].mean(dim=1), rtol=0, atol=0)
    assert not torch.allclose(pooled[:, :_DIM], tokens[:, 0], atol=1e-3)


def test_cls_variant_is_the_headed_cls_prefix_of_cls_patch_mean():
    batch = torch.randn(2, 3, _IMAGE, _IMAGE)
    with torch.inference_mode():
        cls = _tiny_encoder(output_variant="cls").encode_tiles(batch)
        cls_patch_mean = _tiny_encoder(output_variant="cls_patch_mean").encode_tiles(batch)

    assert cls.shape == (2, _DIM)
    torch.testing.assert_close(cls, cls_patch_mean[:, :_DIM], rtol=0, atol=0)


@pytest.mark.parametrize("output_variant", ["cls", "cls_patch_mean"])
def test_head_and_patch_mean_run_in_fp32_outside_autocast(output_variant):
    """Under low-precision autocast only the backbone is cast; features come back fp32.

    The expected value applies the head in plain fp32 to the autocast backbone's
    CLS token; a head run under autocast would drift by the bf16 rounding error.
    """
    encoder = _tiny_encoder(output_variant=output_variant)
    batch = torch.randn(2, 3, _IMAGE, _IMAGE)
    with torch.inference_mode():
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            tokens = encoder._model.forward_features(batch)
            features = encoder.encode_tiles(batch)
        expected = encoder._head(tokens[:, 0].float())
        if output_variant == "cls_patch_mean":
            expected = torch.cat([expected, tokens[:, 5:].float().mean(dim=1)], dim=-1)

    assert features.dtype == torch.float32
    torch.testing.assert_close(features, expected, rtol=0, atol=0)


# --- Heavy: real weights -----------------------------------------------------


def _require_hub_files(*filenames: str) -> None:
    """Skip when the weights / network are unavailable; fetches into the HF cache."""
    from huggingface_hub import hf_hub_download

    for filename in filenames:
        try:
            hf_hub_download(repo_id="slideflow-labs/Mettle", filename=filename)
        except Exception as exc:  # network / weights unavailable
            pytest.skip(f"mettle {filename} unavailable: {type(exc).__name__}: {exc}")


@pytest.fixture(scope="module")
def mettle_encoders():
    """Load the real encoder once per output variant (on CPU).

    Only the download may skip: a strict-load mismatch in the constructor must fail.
    """
    from slide2vec.encoders import encoder_registry

    _require_hub_files("model.safetensors")
    loaded = {}

    def load(output_variant: str = "cls_patch_mean"):
        if output_variant not in loaded:
            encoder = encoder_registry.require("mettle")(output_variant=output_variant)
            loaded[output_variant] = encoder.to("cpu")
        return loaded[output_variant]

    return load


@pytest.fixture(scope="module")
def upstream_mettle():
    """Upstream ``AutoModel(trust_remote_code=True)`` reference, fp32 on CPU."""
    transformers = pytest.importorskip("transformers")
    _require_hub_files(
        "config.json", "configuration_mettle.py", "modeling_mettle.py", "model.safetensors",
    )
    model = transformers.AutoModel.from_pretrained(
        "slideflow-labs/Mettle", trust_remote_code=True,
    )
    return model.to("cpu").eval()


def _tile_batch() -> torch.Tensor:
    generator = torch.Generator().manual_seed(20260730)
    return torch.randn(1, 3, 224, 224, generator=generator)


@pytest.mark.heavy
def test_mettle_loads_strict_and_emits_registered_shapes(mettle_encoders):
    mettle_encoder = mettle_encoders()  # default variant: cls_patch_mean
    cls_encoder = mettle_encoders("cls")
    batch = _tile_batch()
    with torch.inference_mode():
        cls_patch_mean = mettle_encoder.encode_tiles(batch)
        cls = cls_encoder.encode_tiles(batch)
        dense = mettle_encoder.encode_tiles_dense(batch)

    assert mettle_encoder.encode_dim == 3072 and cls_encoder.encode_dim == 1536
    assert mettle_encoder.patch_size == (14, 14)
    assert cls_patch_mean.shape == (1, 3072)
    assert cls.shape == (1, 1536)
    # patch 14: 224 / 14 = 16 -> a 16x16 grid of unheaded 1536-d patch tokens.
    assert dense.shape == (1, 1536, 16, 16)


@pytest.mark.heavy
@pytest.mark.parametrize("output_variant, feature_view", [
    ("cls", "cls"), ("cls_patch_mean", "cls_mean"),
])
def test_mettle_fp32_matches_upstream_remote_code(
    mettle_encoders, upstream_mettle, output_variant, feature_view,
):
    batch = _tile_batch()
    with torch.inference_mode():
        ours = mettle_encoders(output_variant).encode_tiles(batch)
        reference = upstream_mettle.encode(batch, feature_view=feature_view)

    torch.testing.assert_close(ours, reference, rtol=0, atol=1e-5)


@pytest.mark.heavy
def test_mettle_fp16_default_preserves_fp32_direction_on_cuda(mettle_encoders):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    from slide2vec.encoders import encoder_registry
    from slide2vec.runtime.batching import autocast_dtype

    precision = encoder_registry.info("mettle")["precision"]
    assert precision == "fp16"
    mettle_encoder = mettle_encoders()
    batch = _tile_batch()
    with torch.inference_mode():
        reference = mettle_encoder.encode_tiles(batch)
        try:
            encoder = mettle_encoder.to("cuda")
            with torch.autocast(device_type="cuda", dtype=autocast_dtype(torch, precision)):
                fp16 = encoder.encode_tiles(batch.to("cuda")).cpu()
        except torch.cuda.OutOfMemoryError as exc:
            pytest.skip(f"not enough CUDA memory for ViT-g: {exc}")
        finally:
            mettle_encoder.to("cpu")
            torch.cuda.empty_cache()

    assert fp16.dtype == torch.float32
    cosine = torch.nn.functional.cosine_similarity(fp16, reference, dim=-1)
    assert float(cosine.min()) >= 0.9999
