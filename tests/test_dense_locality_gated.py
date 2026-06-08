"""Spatial-registration check for dense extraction at the larger (512px) grid.

This is the "spatially correct, not just non-crashing" oracle. It places a
high-contrast block at a known, asymmetric grid cell and verifies the *change*
in the dense feature map (vs a background-only encoding) peaks at that cell.

Why a *difference* probe: register-free ViTs (UNI, Virchow, Lunit, Prost40M)
carry high-norm artifact/outlier tokens (Darcet et al., "Vision Transformers
Need Registers") that dominate any raw distance-from-background score and produce
false misregistration. Encoding background-only and localizing the delta cancels
those artifacts, isolating the stimulus.

Gated on HF_TOKEN (downloads real encoder weights) and skips per-encoder when a
download is unavailable. Heavy on CPU for ViT-giant encoders — intended to be run
deliberately (ideally on GPU), not in the fast offline suite.
"""

from __future__ import annotations

import os

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("timm")
from timm.data import resolve_data_config  # noqa: E402

pytestmark = pytest.mark.skipif(
    not os.environ.get("HF_TOKEN"),
    reason="HF_TOKEN required to download encoder weights",
)


def _encoder_classes() -> dict:
    from slide2vec.encoders.models.hoptimus import H0Mini, HOptimus0, HOptimus1
    from slide2vec.encoders.models.lunit import LunitTileEncoder
    from slide2vec.encoders.models.prost40m import Prost40M
    from slide2vec.encoders.models.uni import UNI, UNI2
    from slide2vec.encoders.models.virchow import Virchow, Virchow2

    return {
        "uni": UNI,
        "uni2": UNI2,
        "virchow": Virchow,
        "virchow2": Virchow2,
        "h0-mini": H0Mini,
        "h-optimus-0": HOptimus0,
        "h-optimus-1": HOptimus1,
        "lunit": LunitTileEncoder,
        "prost40m": Prost40M,
    }


def _registration_failures(enc, *, target_size: int = 512, block_cells: int = 2):
    """Run the difference-probe; return (encoded_size, grid, list_of_failures)."""
    model = enc._model
    cfg = resolve_data_config(getattr(model, "pretrained_cfg", {}) or {}, model=model)
    mean = torch.tensor(cfg["mean"]).view(1, 3, 1, 1)
    std = torch.tensor(cfg["std"]).view(1, 3, 1, 1)

    patch = enc._dense_patch_size()[0]
    encoded = (target_size // patch) * patch  # largest patch multiple <= target
    grid = encoded // patch

    @torch.no_grad()
    def encode(image: torch.Tensor) -> torch.Tensor:
        return enc.encode_tiles_dense((image - mean) / std)[0]  # (d, G, G)

    background = encode(torch.full((1, 3, encoded, encoded), 0.5))

    def stimulus(cell_r: int, cell_c: int) -> torch.Tensor:
        image = torch.full((1, 3, encoded, encoded), 0.5)
        r0, c0 = cell_r * patch, cell_c * patch
        r1, c1 = r0 + block_cells * patch, c0 + block_cells * patch
        image[:, 0, r0:r1, c0:c1] = 1.0  # red block on gray
        image[:, 1, r0:r1, c0:c1] = 0.0
        image[:, 2, r0:r1, c0:c1] = 0.0
        return image

    # Asymmetric targets so a transpose / flip is detectable; one centre sanity check.
    targets = [
        (int(0.15 * grid), int(0.72 * grid)),
        (int(0.07 * grid), int(0.22 * grid)),
        (int(0.85 * grid), int(0.10 * grid)),
        (grid // 2, grid // 2),
    ]
    failures = []
    for cell_r, cell_c in targets:
        delta = (encode(stimulus(cell_r, cell_c)) - background).pow(2).sum(0).sqrt()
        peak_r, peak_c = divmod(int(delta.argmax().item()), grid)
        # Correct if the peak lands inside the stimulus block (+1-cell halo).
        in_block = (
            cell_r - 1 <= peak_r <= cell_r + block_cells
            and cell_c - 1 <= peak_c <= cell_c + block_cells
        )
        if not in_block:
            failures.append(((cell_r, cell_c), (peak_r, peak_c)))
    return encoded, grid, failures


# The encoders whose dynamic_img_size we FLIPPED this session: real hf-hub source
# + the extra timm kwargs slide2vec passes. The offline proxy test verifies the
# no-op property architecturally; this verifies it on the ACTUAL merged configs
# (no_embed_class, patch-arg quirks) before relying on "native-size no-op" as the
# safety guarantee for the global flag change. Native size is read from the model's
# own pretrained_cfg (NOT the registry input_size, which is the larger preprocessing
# tile size the encoder transform downsizes to the model native, e.g. GigaPath
# 256->224).
_FLIPPED_CONFIGS = {
    "h-optimus-0": ("hf-hub:bioptimus/H-optimus-0", {"init_values": 1e-5}),
    "h-optimus-1": ("hf-hub:bioptimus/H-optimus-1", {"init_values": 1e-5}),
    "gigapath": ("hf_hub:prov-gigapath/prov-gigapath", {}),
    "lunit": ("hf_hub:1aurent/vit_small_patch8_224.lunit_dino", {}),
    "prost40m": ("hf-hub:waticlems/Prost40M", {}),
}


@pytest.mark.parametrize("name", sorted(_FLIPPED_CONFIGS))
def test_flipped_encoder_native_size_is_noop_on_real_config(name: str):
    """Flipping dynamic_img_size on the REAL merged config is bit-identical at
    native size — the guarantee that the global flag change does not alter
    existing pooled extraction for these encoders."""
    import timm

    source, extra = _FLIPPED_CONFIGS[name]
    try:
        static = timm.create_model(
            source, pretrained=True, num_classes=0, dynamic_img_size=False, **extra
        ).eval()
        dynamic = timm.create_model(
            source, pretrained=True, num_classes=0, dynamic_img_size=True, **extra
        ).eval()
    except Exception as exc:  # gated weights / network unavailable
        pytest.skip(f"{name} weights unavailable: {type(exc).__name__}: {exc}")
    dynamic.load_state_dict(static.state_dict())  # identical weights, only flag differs
    native = static.pretrained_cfg["input_size"][-1]  # model native size, not tile size
    x = torch.randn(2, 3, native, native)
    with torch.no_grad():
        torch.testing.assert_close(
            static.forward_features(x), dynamic.forward_features(x), rtol=0, atol=1e-6
        )


# H-optimus recommends dynamic_img_size=False; dense extraction opts in explicitly.
_DENSE_CTOR_KWARGS = {
    "h-optimus-0": dict(dynamic_img_size=True, allow_non_recommended_settings=True),
    "h-optimus-1": dict(dynamic_img_size=True, allow_non_recommended_settings=True),
}


@pytest.mark.parametrize("name", sorted(_encoder_classes()))
def test_dense_grid_is_spatially_registered(name: str):
    encoder_cls = _encoder_classes()[name]
    try:
        enc = encoder_cls(**_DENSE_CTOR_KWARGS.get(name, {})).to("cpu")
    except Exception as exc:  # gated weights / network unavailable
        pytest.skip(f"{name} weights unavailable: {type(exc).__name__}: {exc}")
    encoded, grid, failures = _registration_failures(enc)
    assert not failures, (
        f"{name} @ {encoded}px ({grid}x{grid}) dense grid is misregistered: "
        f"(target -> peak) {failures}"
    )
