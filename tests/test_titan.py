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
    import slide2vec.encoders.models.titan as titan_mod

    monkeypatch.setattr(
        titan_mod,
        "AutoModel",
        SimpleNamespace(from_pretrained=lambda *args, **kwargs: _FakeTitan()),
    )
    encoder = titan_mod.TitanSlideEncoder()

    features = torch.randn(5, 768, dtype=torch.float16)
    coordinates = torch.zeros(5, 2, dtype=torch.int64)
    embedding = encoder.encode_slide(features, coordinates, tile_size_lv0=512)

    assert embedding.shape[-1] == 768
