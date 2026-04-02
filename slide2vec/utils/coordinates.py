import numpy as np


def coordinate_arrays(tiling_result) -> tuple[np.ndarray, np.ndarray]:
    x_values = getattr(tiling_result, "x", None)
    y_values = getattr(tiling_result, "y", None)
    if x_values is None or y_values is None:
        raise ValueError("Tiling result must expose x/y coordinates")
    x_array = np.asarray(x_values)
    y_array = np.asarray(y_values)
    if x_array.ndim != 1 or y_array.ndim != 1 or x_array.shape != y_array.shape:
        raise ValueError(
            f"x and y must have shape (N,), got {x_array.shape} and {y_array.shape}"
        )
    return x_array, y_array
