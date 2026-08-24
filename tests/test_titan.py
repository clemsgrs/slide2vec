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


def _reference_get_alibi(num_heads, w, h, bg_mask=None):
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
    return all_bias.half()


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


def test_patch_declines_unrecognized_remote_code():
    from slide2vec.encoders.models.titan import _patch_titan_remote_code

    assert _patch_titan_remote_code(_FakeTitan()) is False


def test_patched_preprocess_keeps_features_dtype():
    import sys
    import types

    from slide2vec.encoders.models.titan import _patch_titan_remote_code

    module = types.ModuleType("_fake_titan_remote")

    def preprocess_features(features, coords, patch_size_lv0):
        # fp32 zero-grid + index_add_, as in the real remote code: raises on fp16
        grid = torch.zeros(4, features.size(-1))
        grid.index_add_(0, torch.arange(features.size(0)) % 4, features)
        return grid, torch.zeros(4, 2, dtype=torch.int64), torch.ones(4, dtype=torch.bool)

    module.preprocess_features = preprocess_features

    class _FakeViT:
        pos_encode_type = "alibi"
        num_heads = 12
        patch_embed = _pos_embed = norm_pre = blocks = norm = object()
        masked_im_modeling = False

        def forward_features(self, x, coords=None, mask=None, bg_mask=None):
            return x

    _FakeViT.__module__ = module.__name__
    sys.modules[module.__name__] = module
    try:
        model = SimpleNamespace(vision_encoder=_FakeViT())
        assert _patch_titan_remote_code(model) is True
        # repeat call is an idempotent no-op, not a double wrap
        assert _patch_titan_remote_code(model) is True

        fp16_features = torch.randn(6, 32, dtype=torch.float16)
        grid, _, _ = module.preprocess_features(fp16_features, None, 512)
        assert grid.dtype == torch.float16
    finally:
        del sys.modules[module.__name__]
