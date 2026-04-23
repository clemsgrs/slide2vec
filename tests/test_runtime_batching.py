from __future__ import annotations

import torch
from torchvision.transforms import functional as tvF

from slide2vec.runtime.batching import apply_transforms_itemwise


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
