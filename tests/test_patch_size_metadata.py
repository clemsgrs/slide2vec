"""Static ``patch_size`` registry metadata + the cache-key parity contract.

``patch_size`` is a static architectural constant per encoder. Exposing it as
``register_encoder`` metadata lets a consumer resolve the dense cache key WITHOUT
loading the (multi-GB) encoder. The critical correctness contract: the static
value must serialize byte-identically to the runtime ``encoder.patch_size`` tuple
(a downstream soma dense cache key depends on it). The fast tests here pin the
declared values and prove no-load; the heavy test loads each real encoder and
checks tuple equality against the static value.
"""

from __future__ import annotations

import pytest

from slide2vec.encoders import encoder_registry
from slide2vec.encoders.registry import (
    normalize_patch_size,
    resolve_patch_size,
)


# Every dense-capable encoder (those whose model class exposes a runtime
# ``patch_size`` property) and its declared static patch size.
DENSE_PATCH_SIZES: dict[str, tuple[int, int]] = {
    "uni": (16, 16),
    "uni2": (14, 14),
    "virchow": (14, 14),
    "virchow2": (14, 14),
    "gigapath": (14, 14),
    "h-optimus-0": (14, 14),
    "h-optimus-1": (14, 14),
    "h0-mini": (14, 14),
    "lunit": (8, 8),
    "prost40m": (16, 16),
    "conch": (16, 16),
    "conchv15": (16, 16),
    "phikon": (16, 16),
    "phikonv2": (16, 16),
    "midnight": (14, 14),
    "musk": (16, 16),
    "hibou-b": (14, 14),
    "hibou-l": (14, 14),
}

# Non-dense encoders: no recoverable patch grid, so no declared patch_size.
NON_DENSE_ENCODERS = ["gigapath-slide", "titan", "prism"]


def test_normalize_patch_size_int_and_tuple():
    assert normalize_patch_size(14) == (14, 14)
    assert normalize_patch_size((14, 14)) == (14, 14)
    assert normalize_patch_size((16, 8)) == (16, 8)
    # always plain python ints
    result = normalize_patch_size(14)
    assert all(isinstance(v, int) for v in result)


@pytest.mark.parametrize("name,expected", sorted(DENSE_PATCH_SIZES.items()))
def test_registry_info_carries_patch_size_without_construction(name, expected):
    """info(name) returns the declared patch_size; no model is built."""
    info = encoder_registry.info(name)
    assert "patch_size" in info
    assert normalize_patch_size(info["patch_size"]) == expected


@pytest.mark.parametrize("name,expected", sorted(DENSE_PATCH_SIZES.items()))
def test_resolve_patch_size_returns_declared_tuple(name, expected):
    resolved = resolve_patch_size(name)
    assert resolved == expected
    assert isinstance(resolved, tuple)
    assert all(isinstance(v, int) for v in resolved)


def test_resolve_patch_size_constructs_nothing(monkeypatch):
    """resolve_patch_size must never fetch/instantiate the encoder class.

    Sabotage ``encoder_registry.require`` (the only door to the encoder class /
    weights / CUDA): if resolve_patch_size touched it the call would explode. It
    reads pure metadata, so every dense encoder still resolves.
    """
    def _boom(_name):  # pragma: no cover - must not be called
        raise AssertionError("resolve_patch_size must not construct the encoder")

    monkeypatch.setattr(encoder_registry, "require", _boom)
    for name, expected in DENSE_PATCH_SIZES.items():
        assert resolve_patch_size(name) == expected


@pytest.mark.parametrize("name", NON_DENSE_ENCODERS)
def test_resolve_patch_size_raises_for_non_dense_encoder(name):
    with pytest.raises(ValueError, match="does not declare a patch_size"):
        resolve_patch_size(name)


def test_every_tile_encoder_declares_patch_size():
    """All registered tile-level encoders are dense-capable and declare patch_size."""
    for info in encoder_registry.list_with_metadata():
        if info["level"] != "tile":
            continue
        name = info["name"]
        assert info.get("patch_size") is not None, (
            f"tile encoder '{name}' is dense-capable but declares no patch_size"
        )
        assert name in DENSE_PATCH_SIZES, f"untracked tile encoder '{name}'"


# --------------------------------------------------------------------------- #
# Parity contract: static declared == runtime instance ``.patch_size``.
# Loads real foundation-model weights on CPU (minutes each) -> heavy; excluded
# from the PR suite, run on the scheduled heavy workflow. Skips cleanly when
# weights / optional deps are unavailable so a developer run never hard-fails.
# --------------------------------------------------------------------------- #


def test_load_model_drift_guard_rejects_patch_size_mismatch(monkeypatch):
    """The model-load path fails loud when runtime patch_size drifts from static."""
    from types import SimpleNamespace

    import slide2vec.inference as inference

    class _DriftedEncoder:
        def __init__(self, *, output_variant=None):
            self.device = "cpu"
            self.encode_dim = 8
            self.patch_size = (16, 16)  # runtime DISAGREES with declared (14, 14)

        def get_transform(self):
            return SimpleNamespace()

        def to(self, device):
            self.device = device
            return self

    monkeypatch.setattr(inference, "canonicalize_model_name", lambda name: name)
    monkeypatch.setattr(
        inference.encoder_registry,
        "info",
        lambda name: {"level": "tile", "precision": "fp32", "patch_size": 14},
    )
    monkeypatch.setattr(inference.encoder_registry, "require", lambda name: _DriftedEncoder)
    monkeypatch.delenv("HF_TOKEN", raising=False)

    with pytest.raises(ValueError, match="patch_size"):
        inference.load_model(name="drifted-model", device="cpu")


def test_load_model_drift_guard_passes_when_consistent(monkeypatch):
    from types import SimpleNamespace

    import slide2vec.inference as inference

    class _ConsistentEncoder:
        def __init__(self, *, output_variant=None):
            self.device = "cpu"
            self.encode_dim = 8
            self.patch_size = (14, 14)

        def get_transform(self):
            return SimpleNamespace()

        def to(self, device):
            self.device = device
            return self

    monkeypatch.setattr(inference, "canonicalize_model_name", lambda name: name)
    monkeypatch.setattr(
        inference.encoder_registry,
        "info",
        lambda name: {"level": "tile", "precision": "fp32", "patch_size": (14, 14)},
    )
    monkeypatch.setattr(inference.encoder_registry, "require", lambda name: _ConsistentEncoder)
    monkeypatch.delenv("HF_TOKEN", raising=False)

    loaded = inference.load_model(name="consistent-model", device="cpu")
    assert loaded.name == "consistent-model"


@pytest.mark.heavy
@pytest.mark.parametrize("name,expected", sorted(DENSE_PATCH_SIZES.items()))
def test_static_patch_size_matches_runtime(name, expected):
    from slide2vec.inference import load_model

    try:
        loaded = load_model(name=name, device="cpu")
    except Exception as exc:  # network / weights / optional dep unavailable
        pytest.skip(f"{name} weights unavailable: {type(exc).__name__}: {exc}")
    runtime = loaded.model.patch_size
    static = resolve_patch_size(name)
    assert isinstance(runtime, tuple)
    assert static == runtime == expected  # tuple equality protects the cache key
