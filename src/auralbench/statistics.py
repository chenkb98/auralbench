from __future__ import annotations

import math
import numpy as np
from numpy.typing import ArrayLike, NDArray


def finite_values(values: ArrayLike) -> NDArray[np.float64]:
    """Copy a nonempty, bounded vector of finite real observations."""
    array = np.asarray(values)
    if array.dtype.kind not in "iuf" or array.ndim != 1 or not 1 <= len(array) <= 1000000:
        raise ValueError("observations must be a nonempty bounded numeric vector")
    result = np.array(array, dtype=float, copy=True)
    if not np.isfinite(result).all():
        raise ValueError("observations must be finite")
    return result
