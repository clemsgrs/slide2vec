"""Tests for attention-map tile extraction: ``encode_tiles_attention``.

Run fully offline (``pretrained=False``) — no weight downloads, no HF token. The
keystone correctness check proves the recomputed timm attention weights are *the*
weights the fused SDPA kernel applies (``attn_w @ v`` == ``SDPA(q, k, v)``), pinning
the recompute, not just the output shape. Shape / channel-order / multi-block
tests pin the locked ``[block][cls, reg…][head]`` contract.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
timm = pytest.importorskip("timm")
import torch.nn.functional as F  # noqa: E402

from slide2vec.encoders.base import (  # noqa: E402
    TileEncoder,
    TimmTileEncoder,
    prefix_attention_to_grid,
    resolve_block_indices,
    timm_self_attention_weights,
)


def _make_timm_encoder(model_name: str, **timm_kwargs) -> TimmTileEncoder:
    return TimmTileEncoder(model_name, pretrained=False, num_classes=0, **timm_kwargs)


# --------------------------------------------------------------------------- #
# Keystone: recomputed weights == the weights SDPA applies internally.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("reg_tokens", [0, 4])
def test_recomputed_weights_match_sdpa(reg_tokens: int):
    kwargs = {"dynamic_img_size": True}
    if reg_tokens:
        kwargs.update(reg_tokens=reg_tokens, no_embed_class=True, class_token=True)
    enc = _make_timm_encoder("vit_tiny_patch16_224", **kwargs)
    attn = enc._model.blocks[-1].attn
    x = torch.randn(2, 3, 224, 224)

    captured = {}
    handle = attn.register_forward_pre_hook(lambda _m, inp: captured.__setitem__("x", inp[0]))
    with torch.no_grad():
        enc._model.forward_features(x)
    handle.remove()

    xa = captured["x"]
    B, N, _ = xa.shape
    head_dim = int(attn.head_dim)
    qkv = attn.qkv(xa).reshape(B, N, 3, attn.num_heads, head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    q, k = attn.q_norm(q), attn.k_norm(k)
    reference = F.scaled_dot_product_attention(q, k, v)  # fused kernel ground truth

    weights = timm_self_attention_weights(attn, xa)
    mine = weights @ v
    # weights are the exact softmax SDPA applies: applying them to v reproduces SDPA.
    torch.testing.assert_close(mine, reference, rtol=0, atol=1e-5)
    # and every query row is a proper distribution over keys.
    torch.testing.assert_close(weights.sum(-1), torch.ones(B, attn.num_heads, N), atol=1e-5, rtol=0)


# --------------------------------------------------------------------------- #
# encode_tiles_attention: shape + locked channel-order contract.
# --------------------------------------------------------------------------- #


def test_attention_shape_cls_only_patch16():
    enc = _make_timm_encoder("vit_tiny_patch16_224", dynamic_img_size=True)
    nh = enc._model.blocks[-1].attn.num_heads
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        out = enc.encode_tiles_attention(x)  # blocks=(-1,), no registers
    assert out.shape == (2, 1 * nh, 14, 14)  # K = 1 (CLS) * nh


def test_attention_shape_with_registers_and_multiblock():
    enc = _make_timm_encoder(
        "vit_tiny_patch16_224",
        dynamic_img_size=True,
        reg_tokens=4,
        no_embed_class=True,
        class_token=True,
    )
    nh = enc._model.blocks[-1].attn.num_heads
    x = torch.randn(1, 3, 256, 256)  # grid 16x16
    with torch.no_grad():
        out = enc.encode_tiles_attention(x, blocks=(-1, -2), include_registers=True)
    # K = len(blocks) * (1 CLS + 4 reg) * nh
    assert out.shape == (1, 2 * (1 + 4) * nh, 16, 16)


def test_cls_channels_are_register_invariant():
    """The first nh channels (block's CLS-from-each-head) must not depend on the
    include_registers flag — registers only *append* channels after the CLS block."""
    enc = _make_timm_encoder(
        "vit_tiny_patch16_224",
        dynamic_img_size=True,
        reg_tokens=4,
        no_embed_class=True,
        class_token=True,
    )
    nh = enc._model.blocks[-1].attn.num_heads
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        cls_only = enc.encode_tiles_attention(x, include_registers=False)
        with_reg = enc.encode_tiles_attention(x, include_registers=True)
    assert cls_only.shape[1] == nh
    assert with_reg.shape[1] == (1 + 4) * nh
    torch.testing.assert_close(cls_only, with_reg[:, :nh], rtol=0, atol=1e-6)


def test_multiblock_order_is_block_outer():
    """blocks=(-1, -2) must concat block -1's channels then block -2's, in order."""
    enc = _make_timm_encoder("vit_tiny_patch16_224", dynamic_img_size=True)
    nh = enc._model.blocks[-1].attn.num_heads
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        last = enc.encode_tiles_attention(x, blocks=(-1,))
        second_last = enc.encode_tiles_attention(x, blocks=(-2,))
        both = enc.encode_tiles_attention(x, blocks=(-1, -2))
    torch.testing.assert_close(both[:, :nh], last, rtol=0, atol=1e-6)
    torch.testing.assert_close(both[:, nh:], second_last, rtol=0, atol=1e-6)


def test_attention_rows_form_a_distribution_over_grid_is_not_assumed():
    """CLS->patch rows are a *slice* of the full softmax row, so they need NOT sum to
    1 over the patch grid alone (prefix columns carry mass). Sanity: non-negative,
    bounded by the full-row sum (<= 1)."""
    enc = _make_timm_encoder("vit_tiny_patch16_224", dynamic_img_size=True)
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        out = enc.encode_tiles_attention(x)
    assert (out >= 0).all()
    assert (out.flatten(2).sum(-1) <= 1.0 + 1e-5).all()


# --------------------------------------------------------------------------- #
# prefix_attention_to_grid: row-major layout + channel order, in isolation.
# --------------------------------------------------------------------------- #


def test_prefix_attention_to_grid_row_major_and_channel_order():
    nh, grid_h, grid_w, num_prefix = 2, 2, 3, 3  # CLS + 2 registers
    num_patches = grid_h * grid_w
    N = num_prefix + num_patches
    attn = torch.zeros(1, nh, N, N)
    # For each (query prefix q, head) put a unique pattern in the patch columns:
    # patch column p gets value (q*10 + head*100 + p), so we can read order back.
    for q in range(num_prefix):
        for head in range(nh):
            for p in range(num_patches):
                attn[0, head, q, num_prefix + p] = q * 10 + head * 100 + p
    grid = prefix_attention_to_grid(
        attn,
        num_prefix_tokens=num_prefix,
        include_registers=True,
        grid_h=grid_h,
        grid_w=grid_w,
        encoder_name="t",
    )
    assert grid.shape == (1, num_prefix * nh, grid_h, grid_w)
    # channel = q * nh + head  -> [cls, reg…][head]
    for q in range(num_prefix):
        for head in range(nh):
            ch = q * nh + head
            expected = torch.tensor(
                [q * 10 + head * 100 + p for p in range(num_patches)], dtype=torch.float
            ).reshape(grid_h, grid_w)
            torch.testing.assert_close(grid[0, ch], expected, rtol=0, atol=0)


def test_prefix_attention_to_grid_cls_only_drops_registers():
    nh, grid_h, grid_w, num_prefix = 2, 2, 2, 3
    N = num_prefix + grid_h * grid_w
    attn = torch.rand(1, nh, N, N)
    grid = prefix_attention_to_grid(
        attn, num_prefix_tokens=num_prefix, include_registers=False,
        grid_h=grid_h, grid_w=grid_w, encoder_name="t",
    )
    assert grid.shape == (1, nh, grid_h, grid_w)  # CLS only


def test_prefix_attention_to_grid_token_mismatch_fails_loud():
    attn = torch.rand(1, 2, 1 + 9, 1 + 9)  # 9 patches
    with pytest.raises(ValueError, match="token accounting mismatch"):
        prefix_attention_to_grid(
            attn, num_prefix_tokens=1, include_registers=False,
            grid_h=4, grid_w=4, encoder_name="t",  # 16 != 9
        )


def test_prefix_attention_to_grid_rejects_non_4d():
    with pytest.raises(ValueError, match=r"\(B, nh, N, N\)"):
        prefix_attention_to_grid(
            torch.rand(2, 5, 5), num_prefix_tokens=1, include_registers=False,
            grid_h=2, grid_w=2, encoder_name="t",
        )


def test_resolve_block_indices_normalizes_and_validates():
    assert resolve_block_indices((-1, 0, -2), 12, encoder_name="e") == [11, 0, 10]
    with pytest.raises(ValueError, match="out of range"):
        resolve_block_indices((12,), 12, encoder_name="e")


# --------------------------------------------------------------------------- #
# Family B (HF transformers): output_attentions path, faked offline.
# --------------------------------------------------------------------------- #


class _FakeHFAttnOutput:
    def __init__(self, attentions):
        self.attentions = attentions


class _FakeHFViTWithAttn:
    """Minimal HF ViT double exposing per-layer output_attentions."""

    def __init__(self, *, patch_size: int, num_heads: int, num_layers: int):
        self.config = SimpleNamespace(patch_size=patch_size)
        self._num_heads = num_heads
        self._num_layers = num_layers

    def __call__(self, *, pixel_values, output_attentions=False):
        assert output_attentions is True
        b, _, h, w = pixel_values.shape
        n = 1 + (h // self.config.patch_size) * (w // self.config.patch_size)
        # Per-layer softmax-normalized attention so the slice invariants hold.
        attentions = tuple(
            torch.rand(b, self._num_heads, n, n).softmax(dim=-1)
            for _ in range(self._num_layers)
        )
        return _FakeHFAttnOutput(attentions)


def test_phikon_attention_uses_output_attentions():
    from slide2vec.encoders.models.phikon import Phikon

    enc = Phikon.__new__(Phikon)
    enc._model = _FakeHFViTWithAttn(patch_size=16, num_heads=12, num_layers=12)
    x = torch.randn(2, 3, 224, 224)
    out = enc.encode_tiles_attention(x, blocks=(-1,))
    assert out.shape == (2, 12, 14, 14)  # 1 CLS * 12 heads, grid 224/16=14
    multi = enc.encode_tiles_attention(x, blocks=(-1, -2))
    assert multi.shape == (2, 24, 14, 14)


def test_phikon_attention_rejects_indivisible_input():
    from slide2vec.encoders.models.phikon import Phikon

    enc = Phikon.__new__(Phikon)
    enc._model = _FakeHFViTWithAttn(patch_size=16, num_heads=12, num_layers=12)
    with pytest.raises(ValueError, match="divisible by the patch size"):
        enc.encode_tiles_attention(torch.randn(1, 3, 220, 220))


# --------------------------------------------------------------------------- #
# Gating: non-attention-capable encoders raise.
# --------------------------------------------------------------------------- #


def test_attention_unsupported_encoder_raises():
    class _NonAttn(TileEncoder):
        def get_transform(self):  # pragma: no cover - stub
            return lambda x: x

        def encode_tiles(self, batch):  # pragma: no cover - stub
            return batch

        @property
        def encode_dim(self) -> int:  # pragma: no cover - stub
            return 0

        @property
        def device(self):  # pragma: no cover - stub
            return torch.device("cpu")

        def to(self, device):  # pragma: no cover - stub
            return self

    with pytest.raises(NotImplementedError, match="does not support attention-map"):
        _NonAttn().encode_tiles_attention(torch.randn(1, 3, 224, 224))
