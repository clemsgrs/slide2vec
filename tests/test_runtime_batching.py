from __future__ import annotations

import torch
import pytest
from torchvision.transforms import functional as tvF

from slide2vec.runtime.batching import dataloader_kwargs
from slide2vec.runtime.preprocessing import apply_transforms_itemwise


class ConvertToRgbAndBack:
    def __call__(self, image):
        return tvF.pil_to_tensor(image.convert("RGB"))


def test_apply_transforms_itemwise_converts_tensor_samples_for_pil_only_transforms():
    image = torch.tensor(
        [
            [
                [[0, 10], [20, 30]],
                [[40, 50], [60, 70]],
                [[80, 90], [100, 110]],
            ],
            [
                [[1, 11], [21, 31]],
                [[41, 51], [61, 71]],
                [[81, 91], [101, 111]],
            ],
        ],
        dtype=torch.uint8,
    )

    transformed = apply_transforms_itemwise(image, ConvertToRgbAndBack())

    assert torch.equal(transformed, image)


def test_dataloader_kwargs_uses_spawn_only_when_worker_processes_are_enabled():
    assert dataloader_kwargs(
        device=torch.device("cpu"),
        num_workers=2,
        prefetch_factor=3,
        worker_start_method="spawn",
    ) == {
        "num_workers": 2,
        "prefetch_factor": 3,
        "multiprocessing_context": "spawn",
    }
    assert dataloader_kwargs(
        device=torch.device("cpu"),
        num_workers=0,
        prefetch_factor=3,
        worker_start_method="spawn",
    ) == {"num_workers": 0}


def test_batched_preprocessing_transfers_bytes_before_normalizing(monkeypatch):
    """Do not expand a byte image into a float transfer buffer on the host."""
    from types import SimpleNamespace
    from torchvision import transforms
    from slide2vec.runtime.batching import build_batch_preprocessor_for_tile_images

    transferred_dtypes = []
    original_to = torch.Tensor.to

    def capture_transfer(image, *args, **kwargs):
        if args and args[0] == torch.device('cuda'):
            transferred_dtypes.append(image.dtype)
            return image  # Exercise the transfer boundary without requiring a GPU.
        return original_to(image, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, 'to', capture_transfer)
    loaded = SimpleNamespace(name='test', device=torch.device('cuda'), transforms=transforms.Compose([
        transforms.ToTensor(), transforms.Normalize([0.5] * 3, [0.5] * 3),
    ]))
    preprocess = build_batch_preprocessor_for_tile_images(loaded, requested_tile_size_px=2)
    pixels = torch.tensor([[[[0, 255], [255, 0]]] * 3], dtype=torch.uint8)
    expected = torch.tensor([[[[-1.0, 1.0], [1.0, -1.0]]] * 3])

    assert torch.equal(preprocess(pixels), expected)
    assert transferred_dtypes == [torch.uint8]


def test_cuda_prefetch_keeps_distinct_unpinned_batches_intact():
    import pytest
    from contextlib import nullcontext
    from torchvision import transforms
    from slide2vec.runtime.batching import build_batch_preprocessor_for_tile_images, run_forward_pass
    from slide2vec.runtime.types import LoadedModel

    if not torch.cuda.is_available():
        pytest.skip('requires one CUDA device')

    class Encoder:
        def encode_tiles(self, image):
            return image.mean(dim=(-2, -1))

    loaded = LoadedModel(name='test', level='tile', model=Encoder(), feature_dim=3,
                         device=torch.device('cuda'), transforms=transforms.Compose([
                             transforms.ToTensor(), transforms.Normalize([0.5] * 3, [0.5] * 3),
                         ]))
    batches = [(torch.arange(8), torch.zeros((8, 3, 224, 224), dtype=torch.uint8)),
               (torch.arange(8, 16), torch.full((8, 3, 224, 224), 255, dtype=torch.uint8)),
               (torch.arange(16, 19), torch.zeros((3, 3, 224, 224), dtype=torch.uint8))]
    preprocess = build_batch_preprocessor_for_tile_images(loaded, requested_tile_size_px=224)
    indices, features = run_forward_pass(batches, loaded, nullcontext(), batch_preprocessor=preprocess, total_items=19)
    assert indices.tolist() == list(range(19))
    assert features.tolist() == [[-1.0] * 3] * 8 + [[1.0] * 3] * 8 + [[-1.0] * 3] * 3
    assert batches[0][1].count_nonzero().item() == 0
    assert batches[1][1].min().item() == 255


@pytest.mark.parametrize("device,expected_order", [
    ("cpu", ["load first", "load second", "encode", "encode"]),
    ("cuda", ["load first", "encode", "load second", "encode"]),
])
def test_forward_overlaps_loading_only_on_cuda(device, expected_order):
    from contextlib import nullcontext
    from slide2vec.runtime.batching import run_forward_pass
    from slide2vec.runtime.types import LoadedModel

    import pytest
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('requires one CUDA device')

    operations = []

    def batches():
        operations.append('load first')
        yield torch.tensor([0]), torch.full((1, 3, 2, 2), 2.0)
        operations.append('load second')
        yield torch.tensor([1]), torch.full((1, 3, 2, 2), 3.0)

    class Encoder:
        def encode_tiles(self, image):
            operations.append('encode')
            return image.mean(dim=(-2, -1))

    loaded = LoadedModel(name='test', level='tile', model=Encoder(), feature_dim=3,
                         device=torch.device(device), transforms=None)
    indices, features = run_forward_pass(batches(), loaded, nullcontext(), batch_preprocessor=lambda x: x.to(loaded.device), total_items=2)
    assert indices.tolist() == [0, 1]
    assert features.tolist() == [[2.0, 2.0, 2.0], [3.0, 3.0, 3.0]]
    assert operations == expected_order
