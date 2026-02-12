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


def mean(values: ArrayLike) -> float:
    """Compute an overflow-resistant arithmetic mean via magnitude scaling."""
    array = finite_values(values)
    scale = float(np.max(np.abs(array)))
    return float(np.mean(array / scale) * scale) if scale else 0.0


def median(values: ArrayLike) -> float:
    """Compute a stable sample median without mutating observation order."""
    array = finite_values(values)
    scale = float(np.max(np.abs(array)))
    return float(np.median(array / scale) * scale) if scale else 0.0


def quantile(values: ArrayLike, probability: float) -> float:
    """Linear interpolated sample quantile, explicitly matching NumPy's linear method."""
    array = finite_values(values)
    if not np.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError("probability must be in [0, 1]")
    scale = float(np.max(np.abs(array)))
    return float(np.quantile(array / scale, probability, method="linear") * scale) if scale else 0.0
