"""Public contract tests for the Aignostics RudolfV 2 encoder family."""

from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch.nn.functional as F

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")


@dataclass(frozen=True)
class _PresetCase:
    name: str
    embed_dim: int
    pooled_dim: int
    num_heads: int


_PRESETS = (
    _PresetCase(
        name="rudolfv2",
        embed_dim=1536,
        pooled_dim=3072,
        num_heads=24,
    ),
    _PresetCase(
        name="rudolfv2-b",
        embed_dim=768,
        pooled_dim=1536,
        num_heads=12,
    ),
    _PresetCase(
        name="rudolfv2-s",
        embed_dim=384,
        pooled_dim=768,
        num_heads=6,
    ),
)
_PRESETS_BY_NAME = {case.name: case for case in _PRESETS}

class _IdentityRope:
    def __init__(self, prefix_tokens: int, width: int):
        self.prefix_tokens = prefix_tokens
        self.width = width
        self.calls = []

    def __call__(self, rows, cols, device):
        self.calls.append((rows, cols))
        total = self.prefix_tokens + rows * cols
        return torch.zeros(total, self.width, device=device), torch.ones(
            total, self.width, device=device
        )


class _SequencePatchEmbed(torch.nn.Module):
    def __init__(self, tokens, patch_size=8):
        super().__init__()
        self.register_buffer("tokens", torch.as_tensor(tokens, dtype=torch.float32))
        self.patch_size = patch_size

    def forward(self, batch):
        return self.tokens.unsqueeze(0).expand(batch.shape[0], -1, -1)


class _ZeroOutput(torch.nn.Module):
    def forward(self, tokens):
        return torch.zeros_like(tokens)


class _FakeBackbone(torch.nn.Module):
    def __init__(self, *, tokens, dim, num_heads=2, depth=0):
        super().__init__()
        self.patch_embed = _SequencePatchEmbed(tokens)
        self.cls_token = torch.nn.Parameter(torch.full((1, 1, dim), -100.0))
        self.register_tokens = torch.nn.Parameter(
            torch.arange(8 * dim, dtype=torch.float32).reshape(1, 8, dim)
        )
        self.blocks = torch.nn.ModuleList()
        self.norm = torch.nn.Identity()
        for _ in range(depth):
            self.blocks.append(
                _FakeBlock(dim, num_heads, _IdentityRope(9, dim // num_heads))
            )


class _FakeBlock(torch.nn.Module):
    def __init__(self, dim, num_heads, rope):
        super().__init__()
        self.norm1 = torch.nn.Identity()
        self.attn = _FakeAttention(dim, num_heads)
        self.norm2 = torch.nn.Identity()
        self.mlp = _ZeroOutput()
        self.ls1 = torch.nn.Identity()
        self.ls2 = torch.nn.Identity()
        self.drop_path1 = torch.nn.Identity()
        self.drop_path2 = torch.nn.Identity()
        self._rope_ref = (rope,)

    @property
    def rope(self):
        return self._rope_ref[0]


class _FakeAttention(torch.nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.qkv = torch.nn.Linear(dim, dim * 3, bias=False)
        with torch.no_grad():
            self.qkv.weight.zero_()
        self.q_norm = torch.nn.Identity()
        self.k_norm = torch.nn.Identity()

    def forward(self, tokens, rope=None):
        return torch.zeros_like(tokens)


def _wrapped_fake_encoder(*, tokens, dim=8, num_heads=2, depth=0):
    from slide2vec.encoders.models.rudolfv2 import _RudolfV2Encoder

    encoder = _RudolfV2Encoder.__new__(_RudolfV2Encoder)
    encoder._model = SimpleNamespace(
        model=_FakeBackbone(
            tokens=tokens,
            dim=dim,
            num_heads=num_heads,
            depth=depth,
        )
    )
    encoder._model_name = "fake-rudolf"
    encoder._preset_key = "rudolfv2-s"
    encoder._output_variant = "cls_patch_mean"
    encoder._device = torch.device("cpu")
    return encoder


def _install_fake_model_loader(monkeypatch):
    class FakeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.ones(1))
            self.model = SimpleNamespace(
                patch_embed=SimpleNamespace(patch_size=8),
                num_register_tokens=8,
            )

    monkeypatch.setattr(
        transformers.AutoModel,
        "from_pretrained",
        lambda *args, **kwargs: FakeModel(),
    )


def _fake_loaded_encoder(monkeypatch, name):
    from slide2vec.encoders import encoder_registry

    _install_fake_model_loader(monkeypatch)
    return encoder_registry.require(name)()


def _load_rudolfv2_with_fake_model(monkeypatch, name):
    from slide2vec.encoders import encoder_registry

    calls = []

    class FakeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = SimpleNamespace(
                patch_embed=SimpleNamespace(patch_size=8),
                num_register_tokens=8,
            )

    fake_model = FakeModel()

    def fake_from_pretrained(model_id, **kwargs):
        calls.append((model_id, kwargs))
        return fake_model

    monkeypatch.setattr(transformers.AutoModel, "from_pretrained", fake_from_pretrained)
    cls = encoder_registry.require(name)
    return cls(), calls, fake_model


@pytest.mark.parametrize("name", [case.name for case in _PRESETS])
def test_rudolfv2_loads_model_in_eval_mode(monkeypatch, name):
    encoder, _calls, fake_model = _load_rudolfv2_with_fake_model(monkeypatch, name)

    assert encoder._model is fake_model
    assert fake_model.training is False


@pytest.mark.parametrize("name", [case.name for case in _PRESETS])
def test_rudolfv2_official_preprocessing_steps(monkeypatch, name):
    encoder = _fake_loaded_encoder(monkeypatch, name)

    transform = encoder.get_transform()
    assert [type(step).__name__ for step in transform.transforms] == [
        "ToImage",
        "Resize",
        "CenterCrop",
        "ToDtype",
        "Normalize",
    ]
    assert transform.transforms[1].interpolation.name == "BICUBIC"


@pytest.mark.parametrize("name", [case.name for case in _PRESETS])
def test_rudolfv2_official_preprocessing_uses_pinned_normalization(monkeypatch, name):
    encoder = _fake_loaded_encoder(monkeypatch, name)

    output = encoder.get_transform()(torch.zeros(3, 96, 192, dtype=torch.uint8))
    expected = torch.empty(3, 224, 224)
    expected[0].fill_(-3.337423312883)
    expected[1].fill_(-2.514993481095)
    expected[2].fill_(-3.963943661972)
    torch.testing.assert_close(
        output.as_subclass(torch.Tensor), expected, rtol=0, atol=1e-6
    )


@pytest.mark.parametrize("name", [case.name for case in _PRESETS])
def test_rudolfv2_normalization_preserves_geometry(monkeypatch, name):
    encoder = _fake_loaded_encoder(monkeypatch, name)

    normalization = encoder.get_normalization_transform()
    assert normalization(torch.zeros(3, 96, 192, dtype=torch.uint8)).shape == (
        3,
        96,
        192,
    )


def test_rudolfv2_native_square_repair_passes_actual_grid_to_rope():
    encoder = _wrapped_fake_encoder(
        tokens=[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]],
        dim=4,
        depth=1,
    )

    _tokens, grid, _captured = encoder._encode_tokens(torch.zeros(1, 3, 16, 16))

    assert grid == (2, 2)
    assert encoder._model.model.blocks[0].rope.calls == [(2, 2)]


def test_rudolfv2_rectangular_dense_grid_is_row_major_after_nine_prefix_tokens():
    encoder = _wrapped_fake_encoder(
        tokens=[[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]], dim=2
    )
    dense = encoder.encode_tiles_dense(torch.zeros(1, 3, 16, 24))
    expected = torch.tensor(
        [[[[1, 3, 5], [7, 9, 11]], [[2, 4, 6], [8, 10, 12]]]],
        dtype=torch.float32,
    )
    torch.testing.assert_close(dense, expected, rtol=0, atol=0)


def test_rudolfv2_dense_rejects_invalid_rank():
    encoder = _wrapped_fake_encoder(tokens=[[1, 2]], dim=2)

    with pytest.raises(
        ValueError, match=r"encode_tiles_dense expects a \(B, C, H, W\) batch"
    ):
        encoder.encode_tiles_dense(torch.zeros(3, 16, 16))


def test_rudolfv2_dense_rejects_patch_indivisible_geometry():
    encoder = _wrapped_fake_encoder(tokens=[[1, 2]], dim=2)

    with pytest.raises(ValueError, match=r"requires input divisible by the patch size"):
        encoder.encode_tiles_dense(torch.zeros(1, 3, 16, 17))


@pytest.fixture(scope="module", params=[case.name for case in _PRESETS])
def _real_rudolfv2(request):
    from slide2vec.encoders import encoder_registry

    name = request.param
    encoder = encoder_registry.require(name)()
    encoder.to("cpu")
    return name, encoder


def _deterministic_batch(height, width):
    return torch.linspace(-1.0, 1.0, 3 * height * width).reshape(1, 3, height, width)


@pytest.mark.heavy
def test_rudolfv2_real_checkpoint_native_tokens_match_published_path(_real_rudolfv2):
    _name, encoder = _real_rudolfv2
    native = _deterministic_batch(224, 224)

    was_deterministic = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        with torch.inference_mode():
            official = encoder._model.model.encode(native)["last_hidden_state"]
            repaired, grid, _ = encoder._encode_tokens(native)
    finally:
        torch.use_deterministic_algorithms(was_deterministic)

    assert grid == (28, 28)
    assert torch.equal(repaired, official)


@pytest.mark.heavy
@pytest.mark.parametrize("output_variant", ["cls", "cls_patch_mean"])
def test_rudolfv2_real_checkpoint_native_pooled_output_matches_published_path(
    _real_rudolfv2, output_variant
):
    _name, encoder = _real_rudolfv2
    native = _deterministic_batch(224, 224)
    previous_variant = encoder._output_variant
    encoder._output_variant = output_variant
    was_deterministic = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        with torch.inference_mode():
            official = encoder._model.model.encode(native)
            pooled = encoder.encode_tiles(native)
    finally:
        torch.use_deterministic_algorithms(was_deterministic)
        encoder._output_variant = previous_variant

    if output_variant == "cls":
        expected = official["x_norm_clstoken"]
    else:
        expected = torch.cat(
            (official["x_norm_clstoken"], official["x_norm_patchtokens"].mean(dim=1)),
            dim=-1,
        )
    torch.testing.assert_close(pooled, expected, rtol=0, atol=0)


@pytest.mark.heavy
def test_rudolfv2_real_checkpoint_rectangular_pooled_output(_real_rudolfv2):
    name, encoder = _real_rudolfv2
    rectangular = _deterministic_batch(224, 232)
    case = _PRESETS_BY_NAME[name]

    with torch.inference_mode():
        pooled = encoder.encode_tiles(rectangular)

    assert pooled.shape == (1, case.pooled_dim)


@pytest.mark.heavy
def test_rudolfv2_real_checkpoint_rectangular_dense_output(_real_rudolfv2):
    name, encoder = _real_rudolfv2
    rectangular = _deterministic_batch(224, 232)
    case = _PRESETS_BY_NAME[name]

    with torch.inference_mode():
        dense = encoder.encode_tiles_dense(rectangular)

    assert dense.shape == (1, case.embed_dim, 28, 29)


def _capture_rudolfv2_rectangular_fused_attention(encoder, rectangular):
    last_block = len(encoder._backbone.blocks) - 1
    native_attention = {}
    original_sdpa = F.scaled_dot_product_attention

    def capture_fused_attention(query, key, value, **kwargs):
        output = original_sdpa(query, key, value, **kwargs)
        sequence = query.shape[-2]
        identity = torch.eye(
            sequence,
            dtype=value.dtype,
            device=value.device,
        ).expand(query.shape[0], query.shape[1], -1, -1)
        native_attention["weights"] = original_sdpa(query, key, identity, **kwargs)
        return output

    with torch.inference_mode(), pytest.MonkeyPatch.context() as patch:
        patch.setattr(F, "scaled_dot_product_attention", capture_fused_attention)
        encoder._encode_tokens(rectangular, capture_indices={last_block})
    return native_attention["weights"]


@pytest.mark.heavy
def test_rudolfv2_real_checkpoint_rectangular_cls_attention_matches_fused_path(
    _real_rudolfv2,
):
    name, encoder = _real_rudolfv2
    rectangular = _deterministic_batch(224, 232)
    case = _PRESETS_BY_NAME[name]

    with torch.inference_mode():
        attention = encoder.encode_tiles_attention(rectangular)
    expected_weights = _capture_rudolfv2_rectangular_fused_attention(
        encoder, rectangular
    )
    expected_attention = expected_weights[:, :, :1, 9:].reshape(
        1, case.num_heads, 28, 29
    )

    assert attention.shape == (1, case.num_heads, 28, 29)
    torch.testing.assert_close(attention, expected_attention, rtol=1e-5, atol=1e-6)


@pytest.mark.heavy
def test_rudolfv2_real_checkpoint_rectangular_prefix_attention_matches_fused_path(
    _real_rudolfv2,
):
    name, encoder = _real_rudolfv2
    rectangular = _deterministic_batch(224, 232)
    case = _PRESETS_BY_NAME[name]

    with torch.inference_mode():
        attention = encoder.encode_tiles_attention(rectangular, include_registers=True)
    expected_weights = _capture_rudolfv2_rectangular_fused_attention(
        encoder, rectangular
    )
    expected_attention = (
        expected_weights[:, :, :9, 9:]
        .permute(0, 2, 1, 3)
        .reshape(1, case.num_heads * 9, 28, 29)
    )

    assert attention.shape == (1, case.num_heads * 9, 28, 29)
    torch.testing.assert_close(
        attention,
        expected_attention,
        rtol=1e-5,
        atol=1e-6,
    )
