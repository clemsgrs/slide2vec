from __future__ import annotations

import numpy as np


def coordinate_arrays(tiling_result) -> tuple[np.ndarray, np.ndarray]:
    x_values = getattr(tiling_result, "x", None)
    y_values = getattr(tiling_result, "y", None)
    if x_values is None or y_values is None:
        raise ValueError("Tiling result must expose x/y coordinates")
    return np.asarray(x_values), np.asarray(y_values)


def coordinate_matrix(tiling_result) -> np.ndarray:
    return np.column_stack(coordinate_arrays(tiling_result)).astype(int)