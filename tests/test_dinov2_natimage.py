"""Tests for the natural-image DINOv2 control encoder (``dinov2-vitb14``).

``dinov2-vitb14`` is a non-pathology ViT: the original DINOv2 ViT-B/14
self-supervised on LVD-142M (natural images), registered as a tile encoder so it
can act as a "does pathology-pretraining pay off?" control in soma's detection
benchmark. It must behave exactly like the pathology timm ViT tile encoders it
mirrors (registration + dense + attention), so these tests reuse the same offline
(``pretrained=False``) oracle checks as the shared encoder suite.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
timm = pytest.importorskip("timm")

from slide2vec.encoders import encoder_registry  # noqa: E402


def test_dinov2_natimage_metadata_contract():
    info = encoder_registry.info("dinov2-vitb14")
    assert info["level"] == "tile"
    assert info["input_size"] == 224
    assert info["patch_size"] == 14
    # Natural-image encoders have no intrinsic micron spacing; 0.5 is the
    # convention that matches the pathology tile encoders' default task-spacing,
    # so the control runs at identical tile geometry when selected by name.
    assert info["supported_spacing_um"] == pytest.approx(0.5)
    assert info["precision"] == "fp16"
    assert info["source"] == "timm/vit_base_patch14_dinov2.lvd142m"
    assert info["output_variants"]["default"]["encode_dim"] == 768
    assert info["default_output_variant"] == "default"


def test_dinov2_alias_resolves_to_canonical():
    from slide2vec.runtime.model_settings import canonicalize_model_name

    assert canonicalize_model_name("dinov2") == "dinov2-vitb14"
    assert canonicalize_model_name("dinov2-base") == "dinov2-vitb14"


def test_dinov2_natimage_dense_matches_timm_oracle_at_detection_geometry():
    """Dense grid at a 224 tile == timm's own reshape-aware patch grid.

    Runs offline with random weights (``pretrained=False``): the natural-image
    DINOv2 is a plain timm ViT, so the inherited dense path must be bit-identical
    to timm's ``get_intermediate_layers`` oracle, pinning spatial registration
    (not just tensor shape) at the detection tile geometry (224 / 14 = 16).
    """
    from slide2vec.encoders.models.dinov2 import DINOv2ViTB14

    enc = DINOv2ViTB14.__new__(DINOv2ViTB14)
    enc._model = timm.create_model(
        "vit_base_patch14_dinov2.lvd142m",
        pretrained=False,
        num_classes=0,
        dynamic_img_size=True,
    ).eval()
    enc._device = torch.device("cpu")
    enc._output_variant = "default"

    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        mine = enc.encode_tiles_dense(x)
        oracle = enc._model.get_intermediate_layers(
            x, n=1, reshape=True, return_prefix_tokens=False, norm=True
        )[0]
    assert mine.shape == (2, 768, 16, 16)
    torch.testing.assert_close(mine, oracle, rtol=0, atol=1e-6)


def test_dinov2_natimage_attention_shape_cls_only():
    """CLS-only per-head attention grid at the detection geometry (single prefix)."""
    from slide2vec.encoders.models.dinov2 import DINOv2ViTB14

    enc = DINOv2ViTB14.__new__(DINOv2ViTB14)
    enc._model = timm.create_model(
        "vit_base_patch14_dinov2.lvd142m",
        pretrained=False,
        num_classes=0,
        dynamic_img_size=True,
    ).eval()
    enc._device = torch.device("cpu")
    enc._output_variant = "default"

    nh = enc._model.blocks[-1].attn.num_heads
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        out = enc.encode_tiles_attention(x)  # blocks=(-1,), CLS only
    assert out.shape == (1, nh, 16, 16)
