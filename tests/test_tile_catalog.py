from pathlib import Path

import numpy as np

from slide2vec.data.tile_catalog import ensure_tile_catalogs
from slide2vec.utils.parquet import require_pyarrow


def test_tile_catalog_preserves_npy_row_order(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    input_wsi = (repo_root / "tests" / "fixtures" / "input" / "test-wsi.tif").resolve()
    gt_coords = np.load(
        repo_root / "tests" / "fixtures" / "gt" / "test-wsi.npy",
        allow_pickle=False,
    )

    coordinates_dir = tmp_path / "coordinates"
    coordinates_dir.mkdir(parents=True, exist_ok=True)
    np.save(coordinates_dir / "test-wsi.npy", gt_coords)

    catalog_dir = tmp_path / "tile_catalog"
    mapping = ensure_tile_catalogs(
        slide_mask_pairs=[(input_wsi, None)],
        coordinates_dir=coordinates_dir,
        catalog_dir=catalog_dir,
    )

    _, pq, _ = require_pyarrow()
    table = pq.read_table(str(mapping[str(input_wsi)]))
    columns = table.to_pydict()

    expected_idx = np.arange(len(gt_coords), dtype=np.int64)
    np.testing.assert_array_equal(np.asarray(columns["coord_index"], dtype=np.int64), expected_idx)
    np.testing.assert_array_equal(np.asarray(columns["x"], dtype=np.int64), gt_coords["x"].astype(np.int64))
    np.testing.assert_array_equal(np.asarray(columns["y"], dtype=np.int64), gt_coords["y"].astype(np.int64))

    manifest_path = catalog_dir / "manifest.json"
    assert manifest_path.exists()
