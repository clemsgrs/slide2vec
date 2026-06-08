"""Tests for dense (spatial-grid) tile extraction: ``encode_tiles_dense``.

These run fully offline (``pretrained=False``) — no weight downloads, no HF
token. The key correctness check compares our manual prefix-strip + row-major
reshape against timm's own reshape-aware ``get_intermediate_layers`` as a
ground-truth oracle, pinning spatial registration (not just tensor shape).
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
timm = pytest.importorskip("timm")

from slide2vec.encoders.base import (  # noqa: E402
    TileEncoder,
    TimmTileEncoder,
    reshape_tokens_to_grid,
    resolve_recommended_dynamic_img_size,
)


def _make_timm_encoder(model_name: str, **timm_kwargs) -> TimmTileEncoder:
    """Build an offline timm tile encoder (random weights) for shape/order tests."""
    return TimmTileEncoder(
        model_name,
        pretrained=False,
        num_classes=0,
        **timm_kwargs,
    )


def _timm_grid_oracle(model, x: torch.Tensor) -> torch.Tensor:
    """timm's own reshape-aware patch grid: the (B, d, h, w) ground truth."""
    return model.get_intermediate_layers(
        x, n=1, reshape=True, return_prefix_tokens=False, norm=True
    )[0]


# --------------------------------------------------------------------------- #
# Oracle: spatial registration against timm's reshape-aware extraction.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("input_size,grid", [(224, 14), (256, 16)])
def test_dense_matches_timm_oracle_patch16(input_size: int, grid: int):
    enc = _make_timm_encoder("vit_tiny_patch16_224", dynamic_img_size=True)
    x = torch.randn(2, 3, input_size, input_size)
    with torch.no_grad():
        mine = enc.encode_tiles_dense(x)
        oracle = _timm_grid_oracle(enc._model, x)
    assert mine.shape == (2, enc.encode_dim, grid, grid)
    torch.testing.assert_close(mine, oracle, rtol=0, atol=1e-6)


def test_dense_matches_timm_oracle_with_register_tokens():
    # CLS + 8 register tokens, no_embed_class=True (UNI2-style prefix layout).
    enc = _make_timm_encoder(
        "vit_tiny_patch16_224",
        dynamic_img_size=True,
        reg_tokens=8,
        no_embed_class=True,
        class_token=True,
    )
    assert enc._dense_num_prefix_tokens() == 9
    x = torch.randn(2, 3, 256, 256)
    with torch.no_grad():
        mine = enc.encode_tiles_dense(x)
        oracle = _timm_grid_oracle(enc._model, x)
    assert mine.shape == (2, enc.encode_dim, 16, 16)
    torch.testing.assert_close(mine, oracle, rtol=0, atol=1e-6)


def test_dense_matches_timm_oracle_patch14():
    # patch-14 backbone: 518 / 14 = 37, exercises a non-power-of-two grid ratio.
    enc = _make_timm_encoder("vit_small_patch14_reg4_dinov2", dynamic_img_size=True)
    assert enc._dense_patch_size() == (14, 14)
    x = torch.randn(1, 3, 518, 518)
    with torch.no_grad():
        mine = enc.encode_tiles_dense(x)
        oracle = _timm_grid_oracle(enc._model, x)
    assert mine.shape == (1, enc.encode_dim, 37, 37)
    torch.testing.assert_close(mine, oracle, rtol=0, atol=1e-6)


# --------------------------------------------------------------------------- #
# reshape_tokens_to_grid: prefix stripping + row-major layout, in isolation.
# --------------------------------------------------------------------------- #


def test_reshape_strips_prefix_and_is_row_major():
    grid_h, grid_w, num_prefix, dim = 3, 4, 2, 1
    tokens = torch.full((1, num_prefix + grid_h * grid_w, dim), -1.0)
    # Patch token k carries value k; prefix tokens keep the -1 sentinel.
    for k in range(grid_h * grid_w):
        tokens[0, num_prefix + k, 0] = float(k)
    grid = reshape_tokens_to_grid(
        tokens,
        grid_h=grid_h,
        grid_w=grid_w,
        num_prefix_tokens=num_prefix,
        encoder_name="test",
    )
    # grid[0, 0, i, j] must equal flat row-major index i*w + j (no prefix leakage).
    expected = torch.arange(grid_h * grid_w, dtype=torch.float).reshape(
        1, 1, grid_h, grid_w
    )
    assert torch.equal(grid, expected)


def test_reshape_rejects_non_token_sequence():
    with pytest.raises(ValueError, match="token sequence"):
        reshape_tokens_to_grid(
            torch.randn(2, 196),  # missing feature axis
            grid_h=14,
            grid_w=14,
            num_prefix_tokens=1,
            encoder_name="test",
        )


def test_reshape_token_count_mismatch_fails_loud():
    tokens = torch.randn(1, 1 + 14 * 14, 8)
    with pytest.raises(ValueError, match="token accounting mismatch"):
        reshape_tokens_to_grid(
            tokens,
            grid_h=16,  # 16*16=256 != 196
            grid_w=16,
            num_prefix_tokens=1,
            encoder_name="test",
        )


# --------------------------------------------------------------------------- #
# encode_tiles_dense: input validation + fail-loud guards.
# --------------------------------------------------------------------------- #


def test_encode_tiles_dense_rejects_indivisible_input():
    enc = _make_timm_encoder("vit_tiny_patch16_224", dynamic_img_size=True)
    with pytest.raises(ValueError, match="divisible by the patch size"):
        enc.encode_tiles_dense(torch.randn(1, 3, 220, 220))  # 220 % 16 != 0


def test_encode_tiles_dense_rejects_non_4d_input():
    enc = _make_timm_encoder("vit_tiny_patch16_224", dynamic_img_size=True)
    with pytest.raises(ValueError, match=r"\(B, C, H, W\)"):
        enc.encode_tiles_dense(torch.randn(3, 224, 224))


def test_encode_tiles_dense_wrong_prefix_count_fails_loud():
    class _WrongPrefix(TimmTileEncoder):
        def _dense_num_prefix_tokens(self) -> int:
            return super()._dense_num_prefix_tokens() + 3

    enc = _WrongPrefix("vit_tiny_patch16_224", pretrained=False, dynamic_img_size=True)
    with pytest.raises(ValueError, match="token accounting mismatch"):
        enc.encode_tiles_dense(torch.randn(1, 3, 224, 224))


@pytest.mark.parametrize(
    "arch,native",
    [
        ("vit_small_patch16_224", 224),  # patch-16 (UNI / Prost40M family)
        ("vit_small_patch8_224", 224),  # patch-8 (Lunit)
        ("vit_small_patch14_reg4_dinov2", 518),  # patch-14 + reg tokens (H-optimus)
    ],
)
def test_dynamic_img_size_is_native_size_noop(arch: str, native: int):
    """Flipping ``dynamic_img_size`` on must not change native-size output.

    This is why enabling the flag on the previously-static encoders is safe for
    their existing pooled use: at the native input size timm's positional-embed
    resample is an identity, so the dense-vs-static models are bit-identical.
    Weight-independent, so it runs offline with copied random weights.
    """
    static = timm.create_model(
        arch, pretrained=False, num_classes=0, dynamic_img_size=False
    ).eval()
    dynamic = timm.create_model(
        arch, pretrained=False, num_classes=0, dynamic_img_size=True
    ).eval()
    dynamic.load_state_dict(static.state_dict())  # identical weights, only flag differs
    x = torch.randn(2, 3, native, native)
    with torch.no_grad():
        torch.testing.assert_close(
            static.forward_features(x), dynamic.forward_features(x), rtol=0, atol=1e-6
        )


def test_phikon_dense_end_to_end():
    """Exercise the HF (non-timm) dense glue end-to-end on a public encoder.

    Validates ``config.patch_size`` lookup, the ``last_hidden_state`` ``[CLS,
    patches...]`` layout, and num_prefix=1 — the path the reshape unit tests do
    not wire up. Skips when transformers is absent or weights can't be fetched.
    """
    pytest.importorskip("transformers")
    try:
        from slide2vec.encoders.models.phikon import Phikon

        enc = Phikon().to("cpu")
    except Exception as exc:  # network / weights unavailable
        pytest.skip(f"Phikon weights unavailable: {type(exc).__name__}: {exc}")
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        grid = enc.encode_tiles_dense(x)
    assert grid.shape == (2, enc.encode_dim, 14, 14)  # 224 / 16 = 14


def test_gigapath_dense_transform_is_pooled_only_and_crops():
    """GigaPath's get_transform is the POOLED recipe — it center-crops 256->224.

    The dense pipeline must NOT route through it (the crop would drop the tile
    margins and misregister the grid); it supplies its own no-crop transform.
    encode_tiles_dense is transform-agnostic and inherited from TimmTileEncoder,
    so it is NOT overridden/disabled on GigaPath. We assert the pooled transform
    still crops (so the dense pipeline knows to bypass it) without downloading
    weights.
    """
    from slide2vec.encoders.models.gigapath import GigaPath

    enc = GigaPath.__new__(GigaPath)  # no weights needed; get_transform is static
    out = enc.get_transform()(torch.zeros(3, 256, 256, dtype=torch.uint8))
    assert out.shape == (3, 224, 224)  # pooled transform crops to native 224
    # The dense method is the inherited, transform-agnostic one (not disabled).
    assert GigaPath.encode_tiles_dense is TimmTileEncoder.encode_tiles_dense


def test_dense_transform_is_normalization_only_no_resize_or_crop():
    """get_dense_transform must normalize WITHOUT resizing/cropping.

    A resize or center-crop here would shrink/clip the source tile and misregister
    the dense grid against the target mask (the GigaPath/Lunit pooled-transform
    trap). It must use the same resolved mean/std as the pooled transform so dense
    and pooled extraction are photometrically identical.
    """
    from timm.data import resolve_data_config

    enc = _make_timm_encoder("vit_tiny_patch16_224")
    tf = enc.get_dense_transform()
    names = {type(t).__name__ for t in tf.transforms}
    assert "Resize" not in names and "CenterCrop" not in names

    img = torch.randint(0, 256, (3, 320, 288), dtype=torch.uint8)  # non-native, non-square
    out = tf(img)
    assert tuple(out.shape) == (3, 320, 288)  # geometry preserved
    assert out.dtype == torch.float32

    cfg = resolve_data_config(enc._model.pretrained_cfg, model=enc._model)
    mean = torch.tensor(cfg["mean"]).view(3, 1, 1)
    std = torch.tensor(cfg["std"]).view(3, 1, 1)
    expected = (img.float() / 255.0 - mean) / std
    torch.testing.assert_close(out.as_subclass(torch.Tensor), expected, rtol=0, atol=1e-6)


def test_get_dense_transform_unsupported_encoder_raises():
    class _NonDense(TileEncoder):
        def get_transform(self):  # pragma: no cover - trivial stub
            return lambda x: x

        def encode_tiles(self, batch):  # pragma: no cover - trivial stub
            return batch

        @property
        def encode_dim(self) -> int:  # pragma: no cover - trivial stub
            return 0

        @property
        def device(self):  # pragma: no cover - trivial stub
            return torch.device("cpu")

        def to(self, device):  # pragma: no cover - trivial stub
            return self

    with pytest.raises(NotImplementedError, match="does not provide a dense transform"):
        _NonDense().get_dense_transform()


def test_resolve_recommended_dynamic_img_size():
    r = resolve_recommended_dynamic_img_size
    # None -> recommended
    assert r(requested=None, recommended=False, allow_non_recommended=False, encoder_name="e") is False
    assert r(requested=None, recommended=True, allow_non_recommended=False, encoder_name="e") is True
    # matching the recommendation needs no flag
    assert r(requested=False, recommended=False, allow_non_recommended=False, encoder_name="e") is False
    # deviating with the flag is allowed
    assert r(requested=True, recommended=False, allow_non_recommended=True, encoder_name="e") is True
    # deviating without the flag raises
    with pytest.raises(ValueError, match="recommends dynamic_img_size=False"):
        r(requested=True, recommended=False, allow_non_recommended=False, encoder_name="e")


def test_hoptimus_dynamic_img_size_gated_without_download():
    # The guard fires while evaluating __init__ args, before timm downloads weights,
    # so this runs offline: H-optimus recommends dynamic_img_size=False.
    from slide2vec.encoders.models.hoptimus import HOptimus0

    with pytest.raises(ValueError, match="recommends dynamic_img_size=False"):
        HOptimus0(dynamic_img_size=True)


def test_dense_unsupported_encoder_raises_not_implemented():
    class _NonDense(TileEncoder):
        def get_transform(self):  # pragma: no cover - trivial stub
            return lambda x: x

        def encode_tiles(self, batch):  # pragma: no cover - trivial stub
            return batch

        @property
        def encode_dim(self) -> int:  # pragma: no cover - trivial stub
            return 0

        @property
        def device(self):  # pragma: no cover - trivial stub
            return torch.device("cpu")

        def to(self, device):  # pragma: no cover - trivial stub
            return self

    with pytest.raises(NotImplementedError, match="does not support dense"):
        _NonDense().encode_tiles_dense(torch.randn(1, 3, 224, 224))
