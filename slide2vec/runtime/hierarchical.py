import numpy as np

from slide2vec.api import PreprocessingConfig
from slide2vec.utils.coordinates import coordinate_arrays

from .types import HierarchicalIndex


def num_tiles(tiling_result) -> int:
    x_values, _y_values = coordinate_arrays(tiling_result)
    return int(len(x_values))


def is_hierarchical_preprocessing(preprocessing: PreprocessingConfig | None) -> bool:
    if preprocessing is None:
        return False
    return preprocessing.region_tile_multiple is not None or preprocessing.requested_region_size_px is not None


def resolve_hierarchical_geometry(preprocessing: PreprocessingConfig, tiling_result) -> dict[str, int]:
    if preprocessing.region_tile_multiple is None:
        raise ValueError("Hierarchical preprocessing requires region_tile_multiple")
    if preprocessing.requested_region_size_px is None:
        raise ValueError("Hierarchical preprocessing requires requested_region_size_px")
    requested_tile_size_px = int(preprocessing.requested_tile_size_px)
    requested_region_size_px = int(preprocessing.requested_region_size_px)
    requested_spacing_um = float(preprocessing.requested_spacing_um)
    multiple = int(preprocessing.region_tile_multiple)
    if requested_region_size_px % multiple != 0:
        raise ValueError("requested_region_size_px must be divisible by region_tile_multiple")
    read_spacing_um = float(getattr(tiling_result, "read_spacing_um"))
    base_spacing_um = float(getattr(tiling_result, "base_spacing_um"))
    if abs(read_spacing_um - requested_spacing_um) / requested_spacing_um <= float(preprocessing.tolerance):
        read_tile_size_px = requested_tile_size_px
    else:
        read_tile_size_px = int(
            round(requested_tile_size_px * requested_spacing_um / read_spacing_um)
        )
    read_region_size_px = read_tile_size_px * multiple
    # Use the actual read geometry that produced the tile crop. When the
    # resolved spacing is considered equivalent to the requested spacing,
    # this keeps the level-0 footprint aligned with the real crop size.
    tile_size_lv0 = int(round(read_tile_size_px * read_spacing_um / base_spacing_um))
    return {
        "region_tile_multiple": multiple,
        "tiles_per_region": multiple * multiple,
        "requested_tile_size_px": requested_tile_size_px,
        "read_tile_size_px": read_tile_size_px,
        "requested_region_size_px": requested_region_size_px,
        "read_region_size_px": read_region_size_px,
        "tile_size_lv0": tile_size_lv0,
    }


def build_hierarchical_index(
    tiling_result,
    *,
    region_tile_multiple: int,
    tile_size_lv0: int | None = None,
) -> HierarchicalIndex:
    x_values, y_values = coordinate_arrays(tiling_result)
    num_regions = int(len(x_values))
    multiple = int(region_tile_multiple)
    if multiple < 2:
        raise ValueError("region_tile_multiple must be at least 2")
    subtile_size_lv0 = (
        int(tile_size_lv0)
        if tile_size_lv0 is not None
        else int(getattr(tiling_result, "tile_size_lv0")) // multiple
    )
    tiles_per_region = multiple * multiple
    if num_regions == 0:
        empty = np.empty(0, dtype=np.int64)
        return HierarchicalIndex(
            flat_index=empty,
            region_index=np.empty(0, dtype=np.int32),
            subtile_index_within_region=np.empty(0, dtype=np.int32),
            subtile_x=empty,
            subtile_y=empty,
            num_regions=0,
            tiles_per_region=tiles_per_region,
        )
    rows, cols = np.divmod(np.arange(tiles_per_region, dtype=np.int32), multiple)
    offsets_x = cols.astype(np.int64) * subtile_size_lv0
    offsets_y = rows.astype(np.int64) * subtile_size_lv0
    region_x = np.asarray(x_values, dtype=np.int64)[:, np.newaxis]
    region_y = np.asarray(y_values, dtype=np.int64)[:, np.newaxis]
    subtile_x = (region_x + offsets_x[np.newaxis, :]).reshape(-1)
    subtile_y = (region_y + offsets_y[np.newaxis, :]).reshape(-1)
    return HierarchicalIndex(
        flat_index=np.arange(num_regions * tiles_per_region, dtype=np.int64),
        region_index=np.repeat(np.arange(num_regions, dtype=np.int32), tiles_per_region),
        subtile_index_within_region=np.tile(np.arange(tiles_per_region, dtype=np.int32), num_regions),
        subtile_x=subtile_x,
        subtile_y=subtile_y,
        num_regions=num_regions,
        tiles_per_region=tiles_per_region,
    )


def num_embedding_items(tiling_result, preprocessing: PreprocessingConfig | None) -> int:
    if not is_hierarchical_preprocessing(preprocessing):
        return num_tiles(tiling_result)
    geometry = resolve_hierarchical_geometry(preprocessing, tiling_result)
    return num_tiles(tiling_result) * int(geometry["tiles_per_region"])
