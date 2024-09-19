import cv2
import numpy as np


def find_common_spacings(spacings_1, spacings_2, tolerance: float = 0.05):
    common_spacings = []
    for s1 in spacings_1:
        for s2 in spacings_2:
            # check how far appart these two spacings are
            if abs(s1 - s2) / s1 <= tolerance:
                common_spacings.append((s1, s2))
    return common_spacings


class Contour_Checking_fn(object):
    def __call__(self, pt):
        raise NotImplementedError


class HasEnoughTissue(Contour_Checking_fn):
    def __init__(self, contour, contour_holes, tissue_mask, tile_size, scale, pct=0.01):
        self.cont = contour
        self.holes = contour_holes
        self.mask = tissue_mask // 255
        self.tile_size = tile_size
        self.scale = scale
        self.pct = pct

    def __call__(self, pt):
        # work on downsampled image to compute tissue percentage
        # input tile_size is given for level 0
        downsampled_tile_size = int(self.tile_size * 1 / self.scale[0])
        assert (
            downsampled_tile_size > 0
        ), "downsampled tile_size is equal to zero, aborting ; please consider using a smaller seg_params.downsample parameter"
        downsampled_pt = pt * 1 / self.scale[0]
        x_tile, y_tile = downsampled_pt
        x_tile, y_tile = int(x_tile), int(y_tile)

        # draw white filled contour on black background
        contour_mask = np.zeros_like(self.mask)
        cv2.drawContours(contour_mask, [self.cont], 0, (255, 255, 255), -1)

        # draw black filled holes on white filled contour
        cv2.drawContours(contour_mask, self.holes, 0, (0, 0, 0), -1)

        # apply mask to input image
        mask = cv2.bitwise_and(self.mask, contour_mask)

        # x,y axis inversed
        sub_mask = mask[
            y_tile : y_tile + downsampled_tile_size,
            x_tile : x_tile + downsampled_tile_size,
        ]

        tile_area = downsampled_tile_size**2
        tissue_area = np.sum(sub_mask)
        tissue_pct = round(tissue_area / tile_area, 3)

        if tissue_pct >= self.pct:
            return 1, tissue_pct
        else:
            return 0, tissue_pct
