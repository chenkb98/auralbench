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


def trimmed_mean(values: ArrayLike, fraction: float = 0.1) -> float:
    """Drop floor(n*fraction) observations from both tails before averaging."""
    array = np.sort(finite_values(values))
    if not np.isfinite(fraction) or not 0 <= fraction < 0.5:
        raise ValueError("trim fraction must be in [0, .5)")
    count = int(len(array) * fraction)
    return mean(array[count : len(array) - count])


def _sampling(samples: int, seed: int, count: int):
    if (
        isinstance(samples, bool)
        or not isinstance(samples, int)
        or not 1 <= samples <= 100000
        or samples * count > 10000000
    ):
        raise ValueError("invalid resample count or allocation")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**63:
        raise ValueError("invalid seed")
    return np.random.default_rng(seed)


def bootstrap_mean(
    values: ArrayLike, confidence: float = 0.95, samples: int = 1000, seed: int = 0
) -> dict:
    """Percentile bootstrap confidence interval for independent observations."""
    array = finite_values(values)
    if not np.isfinite(confidence) or not 0 < confidence < 1:
        raise ValueError("confidence must be in (0, 1)")
    rng = _sampling(samples, seed, len(array))
    indices = rng.integers(0, len(array), (samples, len(array)))
    scale = float(np.max(np.abs(array)))
    means = np.mean(array[indices] / scale, axis=1) * scale if scale else np.zeros(samples)
    tail = (1 - confidence) / 2
    return {
        "mean": mean(array),
        "lower": quantile(means, tail),
        "upper": quantile(means, 1 - tail),
        "confidence": confidence,
        "samples": samples,
        "seed": seed,
        "n": len(array),
    }


def paired_difference(candidate: ArrayLike, baseline: ArrayLike) -> NDArray[np.float64]:
    """Return candidate-minus-baseline differences with strict pairing."""
    a, b = finite_values(candidate), finite_values(baseline)
    if a.shape != b.shape:
        raise ValueError("paired observations must have equal lengths")
    return finite_values(a - b)
