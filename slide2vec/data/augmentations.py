from typing import Sequence
from einops import rearrange
from torchvision import transforms

# Use timm's names
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


def make_normalize_transform(
    mean: Sequence[float] = IMAGENET_DEFAULT_MEAN,
    std: Sequence[float] = IMAGENET_DEFAULT_STD,
) -> transforms.Normalize:
    return transforms.Normalize(mean=mean, std=std)


class RegionUnfolding:
    def __init__(self, tile_size):
        self.tile_size = tile_size

    def __call__(self, x):
        # x = [3, region_size, region_size]
        # unfold into tilees and rearrange
        x = x.unfold(1, self.tile_size, self.tile_size).unfold(
            2, self.tile_size, self.tile_size
        )  # [3, ntile, region_size, tile_size] -> [3, ntile, ntile, tile_size, tile_size]
        x = rearrange(
            x, "c p1 p2 w h -> (p1 p2) c w h"
        )  # [num_tilees, 3, tile_size, tile_size]
        return x
