"""Behavioral checks for the fixed-coordinate inference measurement workflow."""
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("field,value,message", [
    ("batch_size", 0, "batch_size must be positive"),
    ("modes", ["unknown"], "Unknown inference mode"),
    ("cache_policy", "server-cold", "Unknown cache policy"),
])
def test_inference_benchmark_rejects_invalid_workloads(field, value, message):
    from scripts.benchmark_inference import run_inference_benchmark

    args = SimpleNamespace(batch_size=2, tile_size=224, repeat=1, warmup=0,
                           workers=0, threads=1, modes=["cached"], cache_policy="warm")
    setattr(args, field, value)
    with pytest.raises(ValueError, match=message):
        run_inference_benchmark(args)


def test_inference_modes_preserve_coordinates_batches_and_embeddings(tmp_path, monkeypatch):
    import json
    import numpy as np
    import torch
    from torchvision.transforms import Compose, ToTensor
    import slide2vec.inference
    import slide2vec.data.tile_reader
    from slide2vec.runtime.types import LoadedModel
    from scripts.benchmark_inference import run_inference_benchmark

    class Encoder:
        def encode_tiles(self, images):
            return images.mean(dim=(2, 3))

    class Reader:
        def read_region(self, location, level, size):
            return np.full((size[1], size[0], 3), location[0], dtype=np.uint8)

        def close(self):
            pass

    monkeypatch.setattr(slide2vec.inference, "load_model", lambda **kwargs: LoadedModel(
        name="phikonv2", level="tile", model=Encoder(), transforms=Compose([ToTensor()]),
        feature_dim=3, device=torch.device("cpu"),
    ))
    monkeypatch.setattr(slide2vec.data.tile_reader, "_open_wsi_backend", lambda *a: Reader())
    slide = tmp_path / "slide.tif"
    slide.write_bytes(b"mock slide")
    coordinates = tmp_path / "coordinates.npz"
    np.savez(coordinates, x=np.array([255, 0, 255]), y=np.array([0, 0, 224]))
    output = tmp_path / "result.json"
    args = SimpleNamespace(
        model="phikonv2", slide=slide, coordinates=coordinates, tile_size=224,
        batch_size=2, workers=0, backend="openslide", modes=["model-only", "cached", "wsi"],
        repeat=1, warmup=0, threads=1, output=output, profile=False,
        use_supertiles=False, cache_policy="fresh-reader", device="cpu", precision="fp32",
    )
    result = run_inference_benchmark(args)
    assert set(result["modes"]) == {"model-only", "cached", "wsi"}
    expected = torch.tensor([[1., 1., 1.], [0., 0., 0.], [1., 1., 1.]])
    for mode, measured in result["modes"].items():
        assert len(measured["samples_seconds"]) == 1
        assert measured["max_abs_error"] == 0
        torch.testing.assert_close(torch.load(measured["embeddings_path"], weights_only=True), expected)
    assert json.loads(output.read_text())["parameters"]["num_tiles"] == 3
    args.compare = output
    args.output = tmp_path / "after.json"
    comparison = run_inference_benchmark(args)
    for measured in comparison["modes"].values():
        assert measured["baseline_max_abs_error"] == 0
        assert "speedup" in measured
    baseline_payload = (tmp_path / "result-cached.pt").read_bytes()
    args.output = tmp_path / "result.txt"
    with pytest.raises(ValueError, match="preserve the baseline embeddings"):
        run_inference_benchmark(args)
    assert (tmp_path / "result-cached.pt").read_bytes() == baseline_payload


def test_inference_comparison_rejects_different_coordinates_before_model_load(tmp_path):
    import json
    import numpy as np
    from scripts.benchmark_inference import run_inference_benchmark

    coordinates = tmp_path / "coordinates.npz"
    np.savez(coordinates, x=np.array([0]), y=np.array([0]))
    slide = tmp_path / "slide.tif"
    slide.write_bytes(b"mock slide")
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps({"parameters": {"coordinates_sha256": "different"}}))
    with pytest.raises(ValueError, match="different parameters"):
        run_inference_benchmark(SimpleNamespace(
            model="never-load-this-model", slide=slide, coordinates=coordinates, tile_size=224,
            batch_size=2, workers=0, backend="openslide", modes=["wsi"],
            repeat=1, warmup=0, threads=1, output=tmp_path / "after.json", profile=False,
            use_supertiles=False, cache_policy="warm", device="cpu", precision="fp32", compare=baseline,
        ))
