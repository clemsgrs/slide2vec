"""Tests for the natural-image DINOv3 ViT-B/16 encoder (``dinov3-vitb16``).

Offline (``pretrained=False``) checks against timm's ``Eva`` backbone: registry
metadata, both pooling variants, dense grids, and RoPE-aware attention. The one
real-weight forward is ``heavy``-marked and skips without weights.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
timm = pytest.importorskip("timm")

from slide2vec.encoders import encoder_registry  # noqa: E402

TIMM_NAME = "vit_base_patch16_dinov3.lvd1689m"


def test_dinov3_metadata_contract():
    info = encoder_registry.info("dinov3-vitb16")
    assert info["level"] == "tile"
    assert info["input_size"] == 256
    assert info["patch_size"] == 16
    assert info["supports_variable_input_size"] is True
    assert info["supported_spacing_um"] is None
    assert info["default_spacing_um"] == pytest.approx(0.5)
    assert info["precision"] == "fp16"
    assert info["source"] == "timm/vit_base_patch16_dinov3.lvd1689m"
    assert info["output_variants"] == {
        "patch_mean": {"encode_dim": 768},
        "cls": {"encode_dim": 768},
    }
    assert info["default_output_variant"] == "patch_mean"


@pytest.fixture(scope="module")
def offline_model():
    """The timm Eva backbone with random weights (no download), built once."""
    torch.manual_seed(0)
    return timm.create_model(
        TIMM_NAME, pretrained=False, num_classes=0, dynamic_img_size=True
    ).eval()


def offline_encoder(model, output_variant: str = "patch_mean"):
    from slide2vec.encoders.models.dinov3 import DINOv3ViTB16

    enc = DINOv3ViTB16.__new__(DINOv3ViTB16)
    enc._model = model
    enc._device = torch.device("cpu")
    enc._output_variant = output_variant
    return enc


def test_dinov3_pooling_variants_match_normalized_token_reference(offline_model):
    """patch_mean == mean of the 256 spatial tokens after the final norm (CLS and
    the 4 registers excluded); cls == the normalized CLS token. Both 768-d."""
    enc = offline_encoder(offline_model)
    torch.manual_seed(1)
    x = torch.randn(2, 3, 256, 256)
    with torch.no_grad():
        tokens = enc._model.forward_features(x)  # post final-norm (B, 261, 768)
        patch_mean = enc.encode_tiles(x)
        cls = offline_encoder(offline_model, "cls").encode_tiles(x)
    assert tokens.shape == (2, 1 + 4 + 256, 768)
    assert patch_mean.shape == (2, 768) and cls.shape == (2, 768)
    torch.testing.assert_close(patch_mean, tokens[:, 5:].mean(dim=1), rtol=0, atol=1e-6)
    torch.testing.assert_close(cls, tokens[:, 0], rtol=0, atol=0)
    # The shipped timm forward (global_pool="avg") is the default variant.
    torch.testing.assert_close(patch_mean, enc._model(x), rtol=0, atol=0)
    assert not torch.allclose(cls, patch_mean)
    assert offline_encoder(offline_model, "cls").encode_dim == 768


@pytest.mark.parametrize("size, grid", [(224, 14), (256, 16)])
def test_dinov3_dense_grid_excludes_prefix_tokens(offline_model, size: int, grid: int):
    """Dense grid == timm's own NCHW last-layer intermediates (post-norm), so the
    CLS + 4 register tokens are stripped and the 16px patch grid is row-major."""
    enc = offline_encoder(offline_model)
    torch.manual_seed(2)
    x = torch.randn(2, 3, size, size)
    with torch.no_grad():
        mine = enc.encode_tiles_dense(x)
        oracle = enc._model.forward_intermediates(
            x, indices=[-1], norm=True, output_fmt="NCHW", intermediates_only=True
        )[0]
    assert enc.patch_size == (16, 16)
    assert mine.shape == (2, 768, grid, grid)
    torch.testing.assert_close(mine, oracle, rtol=0, atol=1e-6)


def test_dinov3_dense_rejects_non_patch_multiple(offline_model):
    enc = offline_encoder(offline_model)
    with pytest.raises(ValueError, match="divisible by the patch size"):
        enc.encode_tiles_dense(torch.randn(1, 3, 225, 225))


def backend_attention_weights(model, block_index: int, x: torch.Tensor) -> torch.Tensor:
    """Softmax matrix ``(B, nh, N, N)`` from timm's explicit non-fused Eva path.

    With ``fused_attn=False`` the block materializes ``softmax(q k^T)`` with RoPE
    applied to the spatial rows; it passes through ``attn_drop`` (a no-op under
    eval), where a forward hook records it.
    """
    attn = model.blocks[block_index].attn
    captured = {}
    handle = attn.attn_drop.register_forward_hook(lambda _m, _i, out: captured.__setitem__("w", out))
    prev = attn.fused_attn
    attn.fused_attn = False
    try:
        with torch.no_grad():
            model.forward_features(x)
    finally:
        attn.fused_attn = prev
        handle.remove()
    return captured["w"]


@pytest.mark.parametrize("size, grid", [(224, 14), (256, 16)])
@pytest.mark.parametrize("include_registers", [False, True])
def test_dinov3_attention_matches_backend_rope_path(
    offline_model, size: int, grid: int, include_registers: bool
):
    from slide2vec.encoders.base import prefix_attention_to_grid

    enc = offline_encoder(offline_model)
    nh = enc._model.blocks[-1].attn.num_heads
    torch.manual_seed(3)
    x = torch.randn(2, 3, size, size)
    with torch.no_grad():
        mine = enc.encode_tiles_attention(x, blocks=(-1, -2), include_registers=include_registers)
    expected = torch.cat(
        [
            prefix_attention_to_grid(
                backend_attention_weights(enc._model, index, x),
                num_prefix_tokens=5,
                include_registers=include_registers,
                grid_h=grid,
                grid_w=grid,
                encoder_name="ref",
            )
            for index in (-1, -2)
        ],
        dim=1,
    )
    num_query = 5 if include_registers else 1
    assert nh == 12
    assert mine.shape == (2, 2 * num_query * nh, grid, grid)
    torch.testing.assert_close(mine, expected, rtol=0, atol=1e-6)


def test_dinov3_requires_timm_1_0_20(monkeypatch):
    """The timm registration for ``vit_base_patch16_dinov3`` first shipped in 1.0.20."""
    from slide2vec.encoders.models.dinov3 import DINOv3ViTB16

    monkeypatch.setattr(timm, "__version__", "1.0.19")
    monkeypatch.setattr(timm, "create_model", lambda *a, **k: pytest.fail("must fail before create_model"))
    with pytest.raises(ImportError, match=r"timm>=1\.0\.20"):
        DINOv3ViTB16()


def test_dinov3_alias_resolves_to_canonical():
    from slide2vec.runtime.model_settings import canonicalize_model_name

    assert canonicalize_model_name("dinov3") == "dinov3-vitb16"
    assert canonicalize_model_name("dinov3-vitb") == "dinov3-vitb16"


@pytest.mark.heavy
def test_dinov3_pretrained_pooled_and_dense_forward():
    """Real ``timm/vit_base_patch16_dinov3.lvd1689m`` weights: one pooled and one
    dense forward at the 256 default. Skips when the weights cannot be fetched."""
    try:
        encoder = encoder_registry.require("dinov3-vitb16")()
    except Exception as exc:  # network / weights unavailable
        pytest.skip(f"dinov3-vitb16 weights unavailable: {type(exc).__name__}: {exc}")
    encoder = encoder.to("cpu")
    assert encoder.encode_dim == 768
    assert encoder.patch_size == (16, 16)
    torch.manual_seed(0)
    batch = torch.rand(1, 3, 256, 256)
    with torch.inference_mode():
        pooled = encoder.encode_tiles(batch)
        dense = encoder.encode_tiles_dense(batch)
    assert pooled.shape == (1, 768)
    assert dense.shape == (1, 768, 16, 16)
    torch.testing.assert_close(pooled, dense.mean(dim=(2, 3)), rtol=1e-4, atol=1e-5)
