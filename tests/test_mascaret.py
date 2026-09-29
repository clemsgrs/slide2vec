"""Public contract tests for the Mascaret tile-encoder preset."""

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")


def test_mascaret_dense_encoding_preserves_the_entire_row_major_patch_layout(
    monkeypatch,
):
    from types import SimpleNamespace

    from slide2vec.encoders.models.waiv import Mascaret

    calls = []
    last_hidden_state = torch.tensor(
        [
            [[-1000.0, -1000.0], [1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]],
            [[-2000.0, -2000.0], [5.0, 50.0], [6.0, 60.0], [7.0, 70.0], [8.0, 80.0]],
        ]
    )
    expected = torch.tensor(
        [
            [[[1.0, 2.0], [3.0, 4.0]], [[10.0, 20.0], [30.0, 40.0]]],
            [[[5.0, 6.0], [7.0, 8.0]], [[50.0, 60.0], [70.0, 80.0]]],
        ]
    )

    class FakeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace()

        def forward(self, *, pixel_values):
            calls.append(pixel_values)
            return SimpleNamespace(last_hidden_state=last_hidden_state)

    monkeypatch.setattr(
        transformers.AutoModel,
        "from_pretrained",
        lambda *args, **kwargs: FakeModel(),
    )
    encoder = Mascaret()
    batch = torch.ones(2, 3, 28, 28)

    output = encoder.encode_tiles_dense(batch)

    assert calls == [batch]
    torch.testing.assert_close(output, expected, rtol=0, atol=0)


@pytest.mark.heavy
def test_mascaret_real_weights_pooled_dense_and_attention_contract():
    from slide2vec.encoders.models.waiv import Mascaret

    transformers_major = int(transformers.__version__.split(".", maxsplit=1)[0])
    if transformers_major < 5:
        pytest.skip(
            "Mascaret real weights require the slide2vec[waiv] Transformers 5 runtime"
        )

    try:
        encoder = Mascaret()
    except (ImportError, OSError) as exc:
        pytest.skip(
            f"Mascaret weights/runtime unavailable: {type(exc).__name__}: {exc}"
        )
    encoder.to("cpu")

    default_pixel_values = encoder.get_transform()(
        torch.zeros(3, 224, 224, dtype=torch.uint8)
    ).unsqueeze(0)
    non_default_pooled_pixel_values = encoder.get_normalization_transform()(
        torch.zeros(3, 238, 238, dtype=torch.uint8)
    ).unsqueeze(0)
    rectangular_dense_pixel_values = encoder.get_normalization_transform()(
        torch.zeros(3, 238, 252, dtype=torch.uint8)
    ).unsqueeze(0)
    with torch.no_grad():
        non_default_pooled = encoder.encode_tiles(non_default_pooled_pixel_values)
        rectangular_dense = encoder.encode_tiles_dense(
            rectangular_dense_pixel_values
        )
        wrapper_output = encoder._model(
            pixel_values=default_pixel_values,
            output_attentions=True,
        )

    assert non_default_pooled.shape == (1, 1536)
    torch.testing.assert_close(
        non_default_pooled.norm(dim=-1),
        torch.ones(1),
        rtol=0,
        atol=1e-5,
    )
    assert rectangular_dense.shape == (1, 1536, 17, 18)
    assert wrapper_output.last_hidden_state.shape == (1, 257, 1536)
    assert wrapper_output.attentions is None
