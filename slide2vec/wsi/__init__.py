from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

from .wsi import (
    FilterParameters,
    SegmentationParameters,
    TilingParameters,
    WholeSlideImage,
)


def sort_coordinates_with_tissue(coordinates, tissue_percentages):
    # mock region filenames
    mocked_filenames = [f"{x}_{y}.jpg" for x, y in coordinates]
    # combine mocked filenames with coordinates and tissue percentages
    combined = list(zip(mocked_filenames, coordinates, tissue_percentages))
    # sort combined list by mocked filenames
    sorted_combined = sorted(combined, key=lambda x: x[0])
    # extract sorted coordinates and tissue percentages
    sorted_coordinates = [coord for _, coord, _ in sorted_combined]
    sorted_tissue_percentages = [tissue for _, _, tissue in sorted_combined]
    return sorted_coordinates, sorted_tissue_percentages


def extract_coordinates(
    *,
    wsi_path: Path,
    mask_path: Path,
    backend: str,
    segment_params: SegmentationParameters,
    tiling_params: TilingParameters,
    filter_params: FilterParameters,
    mask_visu_path: Path | None = None,
    num_workers: int = 1,
):
    wsi = WholeSlideImage(
        path=wsi_path,
        mask_path=mask_path,
        backend=backend,
        segment=True,
        segment_params=segment_params,
    )
    assert (
        tiling_params.spacing >= wsi.spacings[0]
    ), f"Desired spacing ({tiling_params.spacing}) is smaller than the whole-slide image starting spacing ({wsi.spacings[0]})"
    (
        contours,
        holes,
        coordinates,
        tissue_percentages,
        tile_level,
        resize_factor,
        tile_size_lv0,
    ) = wsi.get_tile_coordinates(tiling_params, filter_params, num_workers=num_workers)
    sorted_coordinates, _ = sort_coordinates_with_tissue(
        coordinates, tissue_percentages
    )
    if mask_visu_path is not None:
        wsi.visualize_mask(contours, holes).save(mask_visu_path)
    return (
        sorted_coordinates,
        tile_level,
        resize_factor,
        tile_size_lv0,
    )


def save_coordinates(
    *,
    coordinates: list[tuple[int, int]],
    target_spacing: float,
    tile_level: int,
    tile_size: int,
    resize_factor: float,
    tile_size_lv0: int,
    save_path: Path,
):
    x = [x for x, _ in coordinates]  # defined w.r.t level 0
    y = [y for _, y in coordinates]  # defined w.r.t level 0
    ntile = len(x)
    tile_size_resized = int(tile_size * resize_factor)
    dtype = [
        ("x", int),
        ("y", int),
        ("tile_size_resized", int),
        ("tile_level", int),
        ("resize_factor", float),
        ("tile_size_lv0", int),
        ("target_spacing", float),
    ]
    data = np.zeros(ntile, dtype=dtype)
    for i in range(ntile):
        data[i] = (
            x[i],
            y[i],
            tile_size_resized,
            tile_level,
            resize_factor,
            tile_size_lv0,
            target_spacing,
        )
    data_arr = np.array(data)
    np.save(save_path, data_arr)
    return save_path


def draw_grid(img, coord, shape, thickness=2, color=(0, 0, 0, 255)):
    cv2.rectangle(
        img,
        tuple(np.maximum([0, 0], coord - thickness // 2)),
        tuple(coord - thickness // 2 + np.array(shape)),
        color,
        thickness=thickness,
    )
    return img


def draw_grid_from_coordinates(
    canvas,
    wsi,
    coords,
    tile_size_at_0,
    vis_level: int,
    thickness: int = 2,
    indices: list[int] | None = None,
):
    downsamples = wsi.level_downsamples[vis_level]
    if indices is None:
        indices = np.arange(len(coords))
    total = len(indices)

    tile_size = tuple(
        np.ceil((np.array(tile_size_at_0) / np.array(downsamples))).astype(np.int32)
    )  # defined w.r.t vis_level

    wsi_width_at_0, wsi_height_at_0 = wsi.level_dimensions[
        0
    ]  # retrieve slide dimension at level 0

    for idx in range(total):
        tile_id = indices[idx]
        coord = coords[tile_id]
        x, y = coord
        vis_spacing = wsi.get_level_spacing(vis_level)

        width, height = tile_size
        tile = np.ones((height, width, 3), dtype=np.uint8) * 255  # White background

        # compute valid tile area
        if x + tile_size_at_0[0] > wsi_width_at_0:
            valid_width_at_0 = max(
                0, wsi_width_at_0 - x
            )  # how much of the tile width is inside the wsi
            valid_width = int(valid_width_at_0 / downsamples[0])
        else:
            valid_width = width

        if y + tile_size_at_0[1] > wsi_height_at_0:
            valid_height_at_0 = max(
                0, wsi_height_at_0 - y
            )  # how much of the tile height is inside the wsi
            valid_height = int(valid_height_at_0 / downsamples[1])
        else:
            valid_height = height

        # extract only the valid portion of the tile
        if valid_width > 0 and valid_height > 0:
            valid_tile = wsi.get_tile(
                x, y, valid_width, valid_height, spacing=vis_spacing
            )
            valid_tile = Image.fromarray(valid_tile).convert("RGB")
            valid_tile = np.array(valid_tile)
            # paste the valid part into the white tile
            tile[:valid_height, :valid_width, :] = valid_tile

        coord = np.ceil(
            tuple(coord[i] / downsamples[i] for i in range(len(coord)))
        ).astype(np.int32)
        canvas_crop_shape = canvas[
            coord[1] : coord[1] + tile_size[1],
            coord[0] : coord[0] + tile_size[0],
            :3,
        ].shape[:2]
        canvas[
            coord[1] : coord[1] + tile_size[1],
            coord[0] : coord[0] + tile_size[0],
            :3,
        ] = tile[: canvas_crop_shape[0], : canvas_crop_shape[1], :]
        draw_grid(canvas, coord, tile_size, thickness=thickness)

    return Image.fromarray(canvas)


def pad_to_patch_size(canvas: Image.Image, patch_size: tuple[int, int]) -> Image.Image:
    width, height = canvas.size
    # compute amount of padding required for width and height
    pad_width = (patch_size[0] - (width % patch_size[0])) % patch_size[0]
    pad_height = (patch_size[1] - (height % patch_size[1])) % patch_size[1]
    # apply the padding to canvas
    padded_canvas = ImageOps.expand(
        canvas, (0, 0, pad_width, pad_height), fill=(255, 255, 255)
    )  # white padding
    return padded_canvas


def visualize_coordinates(
    *,
    wsi_path: Path,
    coordinates: list[tuple[int, int]],
    tile_level: int,
    tile_size: int,
    save_dir: Path,
    downsample: int = 64,
    backend: str = "asap",
    grid_thickness: int = 1,
):
    wsi = WholeSlideImage(wsi_path, backend=backend)
    vis_level = wsi.get_best_level_for_downsample_custom(downsample)
    vis_spacing = wsi.spacings[vis_level]

    canvas = wsi.get_slide(spacing=vis_spacing)
    canvas = Image.fromarray(canvas).convert("RGB")
    if len(coordinates) == 0:
        return canvas

    tile_size_at_0 = tuple(
        (np.array((tile_size, tile_size)) * wsi.level_downsamples[tile_level]).astype(
            np.int32
        )
    )  # defined w.r.t level 0

    w, h = wsi.level_dimensions[vis_level]
    if w * h > Image.MAX_IMAGE_PIXELS:
        raise Image.DecompressionBombError(
            f"Visualization downsample ({downsample}) is too large"
        )

    tile_size_at_vis_level = tuple(
        (np.array(tile_size_at_0) / np.array(wsi.level_downsamples[vis_level])).astype(
            np.int32
        )
    )  # defined w.r.t vis_level
    canvas = pad_to_patch_size(canvas, tile_size_at_vis_level)
    canvas = np.array(canvas)
    canvas = draw_grid_from_coordinates(
        canvas,
        wsi,
        coordinates,
        tile_size_at_0,
        vis_level,
        indices=None,
        thickness=grid_thickness,
    )
    wsi_name = wsi_path.stem.replace(" ", "_")
    visu_path = Path(save_dir, f"{wsi_name}.jpg")
    canvas.save(visu_path)
