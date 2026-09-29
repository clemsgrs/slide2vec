"""Regression tests for the TITAN slide encoder input contract."""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")


class _FakeTitan:
    """Mirrors TITAN's preprocess_features: index_add_ of the input into an fp32
    grid, which raises on fp16 features."""

    def eval(self):
        return self

    def encode_slide_from_patch_features(self, patch_features, patch_coords, patch_size_lv0):
        grid = torch.zeros(4, patch_features.size(-1))
        indices = torch.zeros(patch_features.size(1), dtype=torch.long)
        grid.index_add_(0, indices, patch_features.squeeze(0))
        return torch.zeros(1, 768)


def test_titan_encode_slide_accepts_fp16_features(monkeypatch):
    # _FakeTitan has no vision_encoder, so the remote-code patch declines and the
    # encoder must fall back to fp32 features to satisfy the fp32-grid index_add_
    import slide2vec.encoders.models.titan as titan_mod

    monkeypatch.setattr(
        titan_mod,
        "AutoModel",
        SimpleNamespace(from_pretrained=lambda *args, **kwargs: _FakeTitan()),
    )
    encoder = titan_mod.TitanSlideEncoder()
    assert encoder._remote_code_patched is False

    features = torch.randn(5, 768, dtype=torch.float16)
    coordinates = torch.zeros(5, 2, dtype=torch.int64)
    embedding = encoder.encode_slide(features, coordinates, tile_size_lv0=512)

    assert embedding.shape[-1] == 768


def _reference_get_alibi(num_heads, w, h, bg_mask=None, dtype=torch.float16):
    """Verbatim port of TITAN's get_alibi (revision dac6773) + the caller's fp16 cast."""
    import math

    import numpy as np

    x, y = np.meshgrid(np.arange(w), np.arange(h), indexing="ij")
    if bg_mask is not None:
        x = x[bg_mask.cpu().squeeze(0)]
        y = y[bg_mask.cpu().squeeze(0)]
    points = np.stack([x.ravel(), y.ravel()], axis=1)
    diffs = points[:, None, :] - points[None, :, :]
    dists = np.sqrt(np.sum(diffs**2, axis=-1))

    def get_slopes(n):
        if math.log2(n).is_integer():
            p = 2 ** (-(2 ** -(math.log2(n) - 3)))
            return [p * (p**i) for i in range(n)]
        nearest = 2 ** math.floor(math.log2(n))
        return get_slopes(nearest) + get_slopes(2 * nearest)[0::2][: n - nearest]

    slopes = torch.tensor(get_slopes(num_heads), dtype=torch.float32).view(num_heads, 1, 1)
    n_patches = dists.shape[-1]
    dists_tensor = torch.tensor(dists, dtype=torch.float32).view(1, n_patches, n_patches)
    bias_matrix = dists_tensor * slopes * -1
    embed_len = n_patches + 1
    all_bias = torch.zeros(1, num_heads, embed_len, embed_len)
    all_bias[:, :, 1:, 1:] = bias_matrix
    return all_bias.to(dtype)


@pytest.mark.parametrize("w,h,use_mask", [(7, 5, False), (9, 11, True), (40, 45, True)])
def test_lean_alibi_bias_matches_reference(w, h, use_mask):
    from slide2vec.encoders.models.titan import _lean_alibi_bias

    torch.manual_seed(0)
    bg_mask = (torch.rand(1, w, h) > 0.3) if use_mask else None
    reference = _reference_get_alibi(12, w, h, bg_mask)
    lean = _lean_alibi_bias(
        SimpleNamespace(num_heads=12), w, h, bg_mask, device="cpu", dtype=torch.float16
    )

    assert torch.equal(lean, reference)
    # padded row stride is what makes SDPA's memory-efficient kernel accept the bias
    assert lean.stride(-2) % 8 == 0


def _fake_remote_module(name):
    """A module shaped like TITAN's vision_transformer.py at the pinned revision:
    same class layout, same forward signatures, same single-slide alibi math."""
    import sys
    import types

    import torch.nn as nn
    import torch.nn.functional as F

    module = types.ModuleType(name)

    def preprocess_features(features, coords, patch_size_lv0):
        # fp32 zero-grid + index_add_, as in the real remote code: raises on fp16
        grid = torch.zeros(4, features.size(-1))
        grid.index_add_(0, torch.arange(features.size(0)) % 4, features)
        return grid, torch.zeros(4, 2, dtype=torch.int64), torch.ones(4, dtype=torch.bool)

    class Attention(nn.Module):
        def __init__(self, dim, num_heads):
            super().__init__()
            self.num_heads = num_heads
            self.head_dim = dim // num_heads
            self.pos_encode = "alibi"
            self.qkv = nn.Linear(dim, dim * 3)
            self.q_norm = self.k_norm = nn.Identity()
            self.proj = nn.Linear(dim, dim)
            self.proj_drop = nn.Dropout(0.0)
            self.attn_drop_prob = 0.0

        def forward(self, x, attn_bias, bg_mask=None):
            B, N, C = x.shape
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)
            x = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias)
            return self.proj_drop(self.proj(x.transpose(1, 2).reshape(B, N, C)))

    class Block(nn.Module):
        def __init__(self, dim, num_heads):
            super().__init__()
            self.norm1 = nn.LayerNorm(dim)
            self.attn = Attention(dim, num_heads)

        def forward(self, x, attn_bias, bg_mask=None):
            return x + self.attn(self.norm1(x), attn_bias=attn_bias, bg_mask=bg_mask)

    class CustomSequential(nn.Module):
        def __init__(self, *modules):
            super().__init__()
            self.modules_list = nn.ModuleList(modules)

        def forward(self, x, attn_mask, bg_mask=None):
            for block in self.modules_list:
                x = block(x, attn_mask, bg_mask)
            return x

    class VisionTransformer(nn.Module):
        pos_encode_type = "alibi"
        masked_im_modeling = False

        def __init__(self, dim=24, num_heads=12, depth=2):
            super().__init__()
            self.num_heads = num_heads
            self.patch_embed = nn.Linear(dim, dim)
            self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
            self.norm_pre = nn.Identity()
            self.blocks = CustomSequential(*[Block(dim, num_heads) for _ in range(depth)])
            self.norm = nn.LayerNorm(dim)

        def _pos_embed(self, x, coords, w, h):
            return torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)

        def forward_features(self, x, coords=None, mask=None, bg_mask=None):
            B, nc, w, h = x.shape
            x = x.flatten(2, 3).transpose(1, 2)
            attn_bias = _reference_get_alibi(self.num_heads, w, h, bg_mask, dtype=x.dtype)
            x = self.norm_pre(self._pos_embed(self.patch_embed(x), coords, w, h))
            if bg_mask is not None:
                keep = torch.cat((torch.ones((1, 1), dtype=torch.bool), bg_mask.view(1, -1)), dim=1)
                x = x[keep].unsqueeze(0)
            return self.norm(self.blocks(x, attn_bias, bg_mask))

    module.preprocess_features = preprocess_features
    for cls in (Attention, Block, CustomSequential, VisionTransformer):
        cls.__module__ = name
        setattr(module, cls.__name__, cls)
    sys.modules[name] = module
    return module


def test_patched_preprocess_keeps_features_dtype():
    import sys

    from slide2vec.encoders.models.titan import _patch_titan_remote_code

    module = _fake_remote_module("_fake_titan_remote")
    try:
        model = SimpleNamespace(vision_encoder=module.VisionTransformer())
        assert _patch_titan_remote_code(model) is True
        # repeat call is an idempotent no-op, not a double wrap
        assert _patch_titan_remote_code(model) is True

        fp16_features = torch.randn(6, 32, dtype=torch.float16)
        grid, _, _ = module.preprocess_features(fp16_features, None, 512)
        assert grid.dtype == torch.float16
    finally:
        del sys.modules[module.__name__]


# Chunked attention: ways it could fail, written down before the code.
# - off by one between token rows (cls first) and tile points: cls row and column
#   must carry a zero bias
# - chunk boundaries: a last partial chunk, a chunk larger than the sequence, and a
#   chunk that holds the cls row together with tile rows
# - background mask: the tile points must follow the order of the kept tokens
# - the bias of a chunk covers more rows than the chunk budget allows (memory)
# - unrecognized remote code is patched anyway instead of falling back
#
# Stated tolerance against the full-bias reference: 1e-5 absolute in fp32 and 2e-3
# absolute in fp16 (fp16 has about 3 significant digits; values are O(1)).
_FP32_ATOL = 1e-5
_FP16_ATOL = 2e-3


@pytest.mark.parametrize(
    "w,h,use_mask,max_elements",
    [
        (7, 5, False, 12 * 36 * 5),  # several chunks, last one partial
        (9, 11, True, 12 * 1),  # budget under one row: one row per chunk
        (6, 6, True, 10**9),  # one chunk larger than the sequence
    ],
)
@pytest.mark.parametrize(
    "dtype,atol", [(torch.float32, _FP32_ATOL), (torch.float16, _FP16_ATOL)]
)
def test_chunked_attention_matches_full_bias_attention(w, h, use_mask, max_elements, dtype, atol):
    import torch.nn.functional as F

    from slide2vec.encoders.models.titan import _ChunkedAlibiBias

    torch.manual_seed(0)
    heads, head_dim = 12, 8
    bg_mask = (torch.rand(1, w, h) > 0.3) if use_mask else None
    reference_bias = _reference_get_alibi(heads, w, h, bg_mask, dtype=dtype)
    length = reference_bias.shape[-1]
    q, k, v = (torch.randn(1, heads, length, head_dim, dtype=dtype) for _ in range(3))
    expected = F.scaled_dot_product_attention(q, k, v, attn_mask=reference_bias)

    bias = _ChunkedAlibiBias(
        SimpleNamespace(num_heads=heads), w, h, bg_mask,
        device="cpu", dtype=dtype, max_elements=max_elements,
    )
    actual = bias.attend(q, k, v)

    assert torch.allclose(actual.float(), expected.float(), atol=atol, rtol=0)


def test_large_slide_is_encoded_without_a_quadratic_bias(monkeypatch):
    import sys

    import torch.nn.functional as F

    import slide2vec.encoders.models.titan as titan_mod

    torch.manual_seed(0)
    w, h, dim = 9, 11, 24
    bg_mask = torch.rand(1, w, h) > 0.3
    n_tokens = int(bg_mask.sum()) + 1
    grid = torch.randn(1, dim, w, h)
    module = _fake_remote_module("_fake_titan_remote_chunked")
    try:
        vit = module.VisionTransformer(dim=dim).eval()
        with torch.no_grad():
            expected = vit.forward_features(grid, bg_mask=bg_mask)

        assert titan_mod._patch_titan_remote_code(SimpleNamespace(vision_encoder=vit)) is True
        # no memory for the full bias; 4 rows per chunk
        monkeypatch.setattr(titan_mod, "_available_bytes", lambda device: 0)
        monkeypatch.setattr(titan_mod, "_CHUNK_ELEMENTS", 12 * n_tokens * 4)
        bias_rows = []
        sdpa = F.scaled_dot_product_attention

        def spy(q, k, v, attn_mask=None, **kwargs):
            bias_rows.append(attn_mask.shape[-2])
            return sdpa(q, k, v, attn_mask=attn_mask, **kwargs)

        monkeypatch.setattr(F, "scaled_dot_product_attention", spy)
        with torch.no_grad():
            actual = vit.forward_features(grid, bg_mask=bg_mask)
    finally:
        del sys.modules[module.__name__]

    assert torch.allclose(actual, expected, atol=_FP32_ATOL, rtol=0)
    assert max(bias_rows) <= 4


def test_unrecognized_attention_is_not_patched():
    import sys

    from slide2vec.encoders.models.titan import _patch_titan_remote_code

    module = _fake_remote_module("_fake_titan_remote_unrecognized")
    try:
        vit = module.VisionTransformer()
        for block in vit.blocks.modules_list:
            block.attn.to_qkv = block.attn.qkv
            del block.attn.qkv
        original = (
            module.preprocess_features,
            module.VisionTransformer.forward_features,
            module.Attention.forward,
        )

        assert _patch_titan_remote_code(SimpleNamespace(vision_encoder=vit)) is False
        assert original == (
            module.preprocess_features,
            module.VisionTransformer.forward_features,
            module.Attention.forward,
        )
    finally:
        del sys.modules[module.__name__]


@pytest.mark.parametrize("tiles,fits", [(5_000, True), (10_000, False)])
def test_full_bias_is_kept_only_where_it_fits(monkeypatch, tiles, fits):
    # fp16 with 12 heads is 24 * N^2 bytes: 0.6 GiB at 5k tiles, 2.2 GiB at 10k
    import slide2vec.encoders.models.titan as titan_mod

    monkeypatch.setattr(titan_mod, "_available_bytes", lambda device: 2**30)

    assert titan_mod._full_bias_fits(tiles, 12, torch.float16, torch.device("cpu")) is fits


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_chunk_bias_rows_are_the_rows_of_the_reference_bias(dtype):
    from slide2vec.encoders.models.titan import _ChunkedAlibiBias

    torch.manual_seed(0)
    w, h = 9, 11
    bg_mask = torch.rand(1, w, h) > 0.3
    reference = _reference_get_alibi(12, w, h, bg_mask, dtype=dtype)
    bias = _ChunkedAlibiBias(
        SimpleNamespace(num_heads=12), w, h, bg_mask, device="cpu", dtype=dtype
    )

    length = reference.shape[-1]
    # a chunk with the cls row and tile rows, then a chunk of tile rows only
    assert torch.equal(bias.rows(0, 5), reference[:, :, 0:5])
    assert torch.equal(bias.rows(5, length), reference[:, :, 5:length])
