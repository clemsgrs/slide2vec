"""Tests for the iSight IHC tile encoder.

iSight is ``openai/clip-vit-large-patch14-336`` fine-tuned on HPA10M, released as
a raw training checkpoint rather than a HF model repo. Its encoder therefore
carries several contracts that no other built-in encoder has, and that are silent
if broken — no exception, just degraded features. These tests pin them offline
against a tiny but *real* ``CLIPModel`` (built from a hand-written config, so no
weights are downloaded), which exercises the genuine HF plumbing the encoder
depends on: ``hidden_states``, ``output_attentions``, and the SDPA default.

The contracts, each traced to https://github.com/zhihuanglab/iSight:

* features come from ``hidden_states[-1]``, never ``pooler_output``
  (``model/patch_encoder_with_clam.py:217-218``);
* the learned ``visual_token_projection`` is applied (``:219``, ``:174``);
* the pooled tile vector is the mean over **all** tokens, CLS included
  (``:255``, which overwrites the CLS line at ``:254``);
* attention extraction must flip the parent ``CLIPModel`` to eager, because
  ``CLIPVisionTransformer`` has no ``set_attn_implementation``.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from slide2vec.encoders import encoder_registry  # noqa: E402

_HIDDEN = 32
_PATCH = 14
_IMAGE = 28  # -> 2x2 patch grid, 5 tokens with CLS
_HEADS = 4
_LAYERS = 2


def _tiny_encoder(*, output_variant: str = "token_mean", projection_scale: float = 2.0):
    """A real (tiny, randomly initialized) CLIPModel wired into ISight.

    ``projection_scale`` makes ``visual_token_projection`` an exact scalar
    multiple of the identity, so a test can assert the projection was applied by
    comparing against the raw tokens.
    """
    from transformers import CLIPConfig, CLIPModel

    from slide2vec.encoders.models.isight import ISight

    config = CLIPConfig(
        vision_config={
            "hidden_size": _HIDDEN,
            "intermediate_size": 2 * _HIDDEN,
            "num_hidden_layers": _LAYERS,
            "num_attention_heads": _HEADS,
            "image_size": _IMAGE,
            "patch_size": _PATCH,
        },
        text_config={
            "hidden_size": _HIDDEN,
            "intermediate_size": 2 * _HIDDEN,
            "num_hidden_layers": 1,
            "num_attention_heads": _HEADS,
        },
        projection_dim=16,
    )
    clip = CLIPModel(config).eval()

    projection = torch.nn.Linear(_HIDDEN, _HIDDEN)
    with torch.no_grad():
        projection.weight.copy_(torch.eye(_HIDDEN) * projection_scale)
        projection.bias.zero_()

    enc = ISight.__new__(ISight)
    enc._clip = clip
    enc._vision = clip.vision_model
    enc._projection = projection.eval()
    enc._device = torch.device("cpu")
    enc._output_variant = output_variant
    return enc


def test_isight_metadata_contract():
    info = encoder_registry.info("isight")
    assert info["level"] == "tile"
    assert info["input_size"] == 336
    assert info["patch_size"] == 14
    assert info["supported_spacing_um"] == pytest.approx(0.5)
    assert info["precision"] == "fp16"
    assert info["source"] == "nirschl-lab/iSight"
    assert info["supports_variable_input_size"] is True
    assert info["output_variants"]["token_mean"]["encode_dim"] == 1024
    assert info["output_variants"]["cls"]["encode_dim"] == 1024
    assert info["default_output_variant"] == "token_mean"


def test_isight_resolves_tiling_defaults():
    from slide2vec.encoders.registry import resolve_preprocessing_defaults

    defaults = resolve_preprocessing_defaults("isight")
    assert defaults["tile_size_px"] == 336
    assert defaults["spacing_um"] == pytest.approx(0.5)


def test_isight_rejects_unknown_output_variant():
    from slide2vec.encoders.registry import resolve_encoder_output

    with pytest.raises(ValueError, match="Unsupported output_variant"):
        resolve_encoder_output("isight", requested_output_variant="pooler")


def test_isight_pooled_is_mean_over_all_tokens_including_cls():
    """The reference reduction is the token-mean, not the CLS token.

    ``patch_encoder_with_clam.py:254`` assigns the CLS token and ``:255``
    immediately overwrites it with ``torch.mean(M, dim=1)``, so the CLS line is
    dead code; ``config/config.ini:3`` corroborates with
    ``model_version = v3_all_tokens``. Getting this backwards silently changes
    the representation, so pin it against an explicit all-token mean.
    """
    enc = _tiny_encoder(projection_scale=2.0)
    x = torch.randn(2, 3, _IMAGE, _IMAGE)
    with torch.no_grad():
        tokens = enc._vision(x, output_hidden_states=True).hidden_states[-1]
        pooled = enc.encode_tiles(x)

    assert tokens.shape == (2, 5, _HIDDEN)  # 1 CLS + 2x2 patches
    torch.testing.assert_close(pooled, 2.0 * tokens.mean(dim=1), rtol=0, atol=1e-6)
    # and it is genuinely different from the CLS-only reduction
    assert not torch.allclose(pooled, 2.0 * tokens[:, 0], atol=1e-4)


def test_isight_cls_variant_selects_the_cls_token():
    enc = _tiny_encoder(output_variant="cls", projection_scale=2.0)
    x = torch.randn(2, 3, _IMAGE, _IMAGE)
    with torch.no_grad():
        tokens = enc._vision(x, output_hidden_states=True).hidden_states[-1]
        pooled = enc.encode_tiles(x)
    torch.testing.assert_close(pooled, 2.0 * tokens[:, 0], rtol=0, atol=1e-6)


def test_isight_never_uses_pooler_output():
    """``pooler_output`` routes through ``post_layernorm``, which is bit-identical
    to stock CLIP in the released checkpoint (it received no gradient). Using it
    would push fine-tuned features through untrained parameters, silently. Pin
    that the pooled feature is NOT the pooled/normalized branch."""
    enc = _tiny_encoder(projection_scale=1.0)
    x = torch.randn(2, 3, _IMAGE, _IMAGE)
    with torch.no_grad():
        out = enc._vision(x, output_hidden_states=True)
        pooled = enc.encode_tiles(x)
    assert not torch.allclose(pooled, out.pooler_output, atol=1e-4)
    # last_hidden_state is the pre-post_layernorm sequence the reference reads
    torch.testing.assert_close(
        pooled, out.hidden_states[-1].mean(dim=1), rtol=0, atol=1e-6
    )


def test_isight_dense_applies_projection_and_preserves_row_major_grid():
    enc = _tiny_encoder(projection_scale=2.0)
    x = torch.randn(2, 3, _IMAGE, _IMAGE)
    with torch.no_grad():
        tokens = enc._vision(x, output_hidden_states=True).hidden_states[-1]
        grid = enc.encode_tiles_dense(x)

    assert grid.shape == (2, _HIDDEN, 2, 2)
    expected = (2.0 * tokens[:, 1:]).transpose(1, 2).reshape(2, _HIDDEN, 2, 2)
    torch.testing.assert_close(grid, expected, rtol=0, atol=1e-6)


def test_isight_interpolates_positions_for_every_variable_extraction_path():
    enc = _tiny_encoder()
    rectangular = torch.randn(1, 3, 42, 28)

    with torch.no_grad():
        tokens = enc._projection(
            enc._vision(
                rectangular,
                output_hidden_states=True,
                interpolate_pos_encoding=True,
            ).hidden_states[-1]
        )
        pooled = enc.encode_tiles(rectangular)
        dense = enc.encode_tiles_dense(rectangular)
        attention = enc.encode_tiles_attention(rectangular)

    assert pooled.shape == (1, _HIDDEN)
    assert dense.shape == (1, _HIDDEN, 3, 2)
    assert attention.shape == (1, _HEADS, 3, 2)
    expected_dense = tokens[:, 1:].transpose(1, 2).reshape(1, _HIDDEN, 3, 2)
    torch.testing.assert_close(dense, expected_dense, rtol=0, atol=0)


def test_isight_positional_interpolation_is_exactly_inert_at_native_size():
    from slide2vec.encoders.base import attentions_tuple_to_grids, hf_eager_attention

    enc = _tiny_encoder()
    native = torch.randn(1, 3, _IMAGE, _IMAGE)

    with torch.no_grad():
        prior_tokens = enc._projection(
            enc._vision(native, output_hidden_states=True).hidden_states[-1]
        )
        tokens = enc._token_features(native)
        with hf_eager_attention(enc._clip):
            prior_with_attention = enc._vision(native, output_attentions=True)
        attention = enc.encode_tiles_attention(native)

    prior_attention = attentions_tuple_to_grids(
        prior_with_attention.attentions,
        num_prefix_tokens=1,
        blocks=(-1,),
        include_registers=False,
        grid_h=2,
        grid_w=2,
        encoder_name="ISight",
    )
    assert torch.equal(tokens, prior_tokens)
    assert torch.equal(attention, prior_attention)


def test_isight_dense_rejects_indivisible_input():
    enc = _tiny_encoder()
    with pytest.raises(ValueError, match="divisible by the patch size"):
        enc.encode_tiles_dense(torch.randn(1, 3, _IMAGE - 1, _IMAGE))


def test_isight_attention_forces_eager_via_the_parent_clip_model():
    """The vision tower alone cannot switch attention implementations.

    ``CLIPVisionTransformer`` is a plain ``nn.Module``, not a
    ``PreTrainedModel``, so it exposes no way to select an attention
    implementation; handing it to ``hf_eager_attention`` would be a silent no-op.
    The encoder keeps the parent ``CLIPModel`` for exactly this reason, and this
    test fails if that is ever "simplified" away.

    Asserted as *structure plus behaviour*, never as a specific ``transformers``
    API: ``PreTrainedModel.set_attn_implementation`` only exists from 4.56, and
    CI pins 4.53 via the ``prism`` extra. Older versions instead fall back to
    eager inside the SDPA attention class when ``output_attentions=True``. Both
    routes must end in materialized attention weights, which is what is checked
    here; the eager flip itself is asserted only where the API exists.
    """
    from transformers.modeling_utils import PreTrainedModel

    from slide2vec.encoders.base import hf_eager_attention

    enc = _tiny_encoder()
    assert not isinstance(enc._vision, PreTrainedModel)
    assert not hasattr(enc._vision, "set_attn_implementation")
    # the parent IS a PreTrainedModel -> hf_eager_attention has something to act on
    assert isinstance(enc._clip, PreTrainedModel)

    before = enc._clip.config.vision_config._attn_implementation
    if hasattr(enc._clip, "set_attn_implementation"):
        # transformers >= 4.56: the flip is real and reaches the vision sub-config
        with hf_eager_attention(enc._clip):
            assert enc._clip.config.vision_config._attn_implementation == "eager"

    with torch.no_grad():
        maps = enc.encode_tiles_attention(torch.randn(1, 3, _IMAGE, _IMAGE))

    # the contract that actually matters: weights materialized, not attentions=None
    assert maps.shape == (1, _HEADS, 2, 2)  # 1 CLS query * heads, 2x2 grid
    assert torch.isfinite(maps).all()
    assert (maps >= 0).all()
    # implementation restored, so a following encode_tiles is unaffected
    assert enc._clip.config.vision_config._attn_implementation == before


def test_isight_attention_multiblock_is_block_outer():
    enc = _tiny_encoder()
    with torch.no_grad():
        multi = enc.encode_tiles_attention(
            torch.randn(1, 3, _IMAGE, _IMAGE), blocks=(0, 1)
        )
        first = enc.encode_tiles_attention(torch.randn(1, 3, _IMAGE, _IMAGE), blocks=(0,))
    assert multi.shape == (1, 2 * _HEADS, 2, 2)
    assert first.shape == (1, _HEADS, 2, 2)


def test_isight_attention_rejects_indivisible_input():
    enc = _tiny_encoder()
    with pytest.raises(ValueError, match="divisible by the patch size"):
        enc.encode_tiles_attention(torch.randn(1, 3, _IMAGE, _IMAGE - 1))


def test_isight_reports_patch_size_from_the_loaded_config():
    enc = _tiny_encoder()
    assert enc.patch_size == (_PATCH, _PATCH)
    assert enc.encode_dim == 1024
