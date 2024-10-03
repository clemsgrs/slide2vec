import cv2
import numpy as np

from PIL import Image, ImageOps
from pathlib import Path
from typing import Optional, List, Dict, Tuple

from .wsi import WholeSlideImage


def sort_coords_with_tissue(coords, tissue_percentages):
    # mock region filenames
    mocked_filenames = [f"{x}_{y}.jpg" for x, y in coords]
    # combine mocked filenames with coordinates and tissue percentages
    combined = list(zip(mocked_filenames, coords, tissue_percentages))
    # sort combined list by mocked filenames
    sorted_combined = sorted(combined, key=lambda x: x[0])
    # extract sorted coordinates and tissue percentages
    sorted_coords = [coord for _, coord, _ in sorted_combined]
    sorted_tissue_percentages = [tissue for _, _, tissue in sorted_combined]
    return sorted_coords, sorted_tissue_percentages


def extract_coordinates(
    wsi_fp,
    mask_fp,
    spacing,
    tile_size,
    backend,
    tissue_val,
    downsample: int = 64,
    tiling_params: Dict[str, int] = {
        "overlap": 0.0,
        "drop_holes": False,
        "tissue_thresh": 0.01,
        "use_padding": True,
        "ref_tile_size": 16,
        "a_t": 4,
        "a_h": 2,
        "max_n_holes": 8,
    },
    mask_visu_path: Optional[Path] = None,
    num_workers: int = 1,
):
    wsi = WholeSlideImage(
        wsi_fp,
        mask_fp,
        backend=backend,
        tissue_val=tissue_val,
        downsample=downsample,
        segment=True,
    )
    (
        contours,
        holes,
        coordinates,
        tissue_percentages,
        tile_level,
        resize_factor,
    ) = wsi.get_tile_coordinates(
        spacing, tile_size, tiling_params, num_workers=num_workers
    )
    sorted_coordinates, sorted_tissue_percentages = sort_coords_with_tissue(
        coordinates, tissue_percentages
    )
    if mask_visu_path is not None:
        wsi.visualize_mask(contours, holes).save(mask_visu_path)
    return sorted_coordinates, sorted_tissue_percentages, tile_level, resize_factor


def save_coordinates(
    coordinates, target_spacing, level, tile_size, resize_factor, save_path
):
    x = [x for x, _ in coordinates]  # defined w.r.t level 0
    y = [y for _, y in coordinates]  # defined w.r.t level 0
    ntile = len(x)
    tile_size_resized = tile_size * resize_factor
    data = []
    for i in range(ntile):
        data.append(
            [x[i], y[i], tile_size_resized, level, resize_factor, target_spacing]
        )
    data_arr = np.array(data, dtype=int)
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
    indices: Optional[List[int]] = None,
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


def pad_to_patch_size(canvas: Image.Image, patch_size: Tuple[int, int]) -> Image.Image:
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
    wsi_fp,
    coordinates,
    tile_level,
    tile_size,
    resize_factor,
    save_dir,
    downsample: int = 64,
    backend: str = "asap",
    grid_thickness: int = 1,
):
    wsi = WholeSlideImage(wsi_fp, backend=backend)
    vis_level = wsi.get_best_level_for_downsample_custom(downsample)
    vis_spacing = wsi.spacings[vis_level]

    canvas = wsi.get_slide(spacing=vis_spacing)
    canvas = Image.fromarray(canvas).convert("RGB")
    if len(coordinates) == 0:
        return canvas

    tile_size = tile_size * resize_factor  # defined w.r.t tile_level
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
    visu_path = Path(save_dir, f"{wsi_fp.stem}.jpg")
    canvas.save(visu_path)
