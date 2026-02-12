import numpy as np
import pytest
from auralbench import statistics as m


def test_finite_values_example():
    np.testing.assert_array_equal(m.finite_values([1, 2]), [1.0, 2.0])


def test_finite_values_boundaries():
    for values in [[], [np.nan], [np.inf], [[1]], [True], ["1"], [complex(1)]]:
        with pytest.raises(ValueError):
            m.finite_values(values)


def test_finite_values_invariants():
    x = np.array([1, 2, 3])
    y = m.finite_values(x)
    y[0] = 99
    assert x[0] == 1 and y.dtype == np.float64
    np.testing.assert_array_equal(m.finite_values(m.finite_values(x)), x)


def test_mean_example():
    assert m.mean([1, 2, 6]) == pytest.approx(3)


def test_mean_boundaries():
    assert m.mean([1e308, 1e308]) == 1e308
    assert m.mean([1e308, -1e308]) == 0
    assert m.mean([0]) == 0
    with pytest.raises(ValueError):
        m.mean([])


def test_mean_invariants():
    values = np.array([-3.0, 1.0, 5.0, 9.0])
    assert m.mean(values[::-1]) == pytest.approx(m.mean(values))
    assert m.mean(values + 10) == pytest.approx(m.mean(values) + 10)
    assert m.mean(values * 3) == pytest.approx(m.mean(values) * 3)


def test_median_example():
    assert m.median([9, 1, 5]) == 5
    assert m.median([1, 3, 7, 9]) == 5


def test_median_boundaries():
    assert m.median([1e308, 1e308]) == 1e308
    assert m.median([-1e308, 1e308]) == 0
    with pytest.raises(ValueError):
        m.median([np.nan])


def test_median_invariants():
    x = np.array([5.0, 1.0, 2.0, 7.0, 3.0])
    assert m.median(x) == m.median(x[::-1]) == 3
    assert m.median(x + 10) == pytest.approx(13)
    np.testing.assert_array_equal(x, [5, 1, 2, 7, 3])
