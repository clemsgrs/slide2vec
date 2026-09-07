"""PRISM accepts persisted features independently of their storage dtype."""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")


class _TinyPrism(torch.nn.Module):
    def __init__(self, dtype):
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=dtype)
        )

    @property
    def dtype(self):
        return self.weight.dtype

    def slide_representations(self, tile_features):
        projected = torch.nn.functional.linear(tile_features, self.weight)
        return {"image_embedding": projected.sum(dim=1)}


@pytest.mark.parametrize(
    ("stored_dtype", "model_dtype"),
    [
        (torch.float16, torch.float32),
        (torch.bfloat16, torch.float32),
        (torch.float32, torch.float32),
        (torch.float32, torch.float16),
    ],
)
def test_prism_aggregates_features_in_model_dtype(monkeypatch, stored_dtype, model_dtype):
    import slide2vec.encoders.models.prism as prism

    monkeypatch.setattr(
        prism,
        "AutoModel",
        SimpleNamespace(from_pretrained=lambda *args, **kwargs: _TinyPrism(model_dtype)),
    )
    encoder = prism.PrismSlideEncoder().to("cpu")
    features = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=stored_dtype)

    embedding = encoder.encode_slide(features)

    torch.testing.assert_close(
        embedding, torch.tensor([16.0, 36.0], dtype=model_dtype), rtol=0, atol=0,
    )
    assert features.dtype == stored_dtype
