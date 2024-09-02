import numpy as np

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


def extract_coordinates(wsi_fp, mask_fp, spacing, region_size, num_workers: int = 1):
    wsi = WholeSlideImage(wsi_fp, mask_fp)
    coordinates, tissue_percentages, patch_level, resize_factor = wsi.get_patch_coordinates(spacing, region_size, num_workers=num_workers)
    sorted_coordinates, sorted_tissue_percentages = sort_coords_with_tissue(coordinates, tissue_percentages)
    return sorted_coordinates, sorted_tissue_percentages, patch_level, resize_factor


def save_coordinates(coordinates, target_spacing, level, tile_size, resize_factor, save_path):
    x = list(coordinates[:, 0])  # defined w.r.t level 0
    y = list(coordinates[:, 1])  # defined w.r.t level 0
    npatch = len(x)
    tile_size_resized = tile_size * resize_factor
    data = []
    for i in range(npatch):
        data.append([x[i], y[i], tile_size_resized, level, resize_factor, target_spacing])
    data_arr = np.array(data, dtype=int)
    np.save(save_path, data_arr)
    return save_path


def visualize_coordinates():
    return NotImplementedError