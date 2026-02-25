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


def test_quantile_example():
    assert m.quantile([0, 10, 20, 30], 0.25) == pytest.approx(7.5)


def test_quantile_boundaries():
    for probability in [-1, 2, np.nan, np.inf]:
        with pytest.raises(ValueError):
            m.quantile([1, 2], probability)
    assert m.quantile([5], 0.9) == 5
    assert m.quantile([-1e308, 1e308], 0.5) == 0


def test_quantile_invariants():
    x = [5, 1, 9, 3]
    assert m.quantile(x, 0) == 1 and m.quantile(x, 1) == 9
    assert m.quantile(x, 0.5) == pytest.approx(m.median(x))
    values = [m.quantile(x, q) for q in np.linspace(0, 1, 21)]
    assert all(a <= b for a, b in zip(values, values[1:]))


def test_trimmed_mean_example():
    assert m.trimmed_mean([-100, 1, 2, 3, 100], 0.2) == pytest.approx(2)


def test_trimmed_mean_boundaries():
    for fraction in [-1, 0.5, 1, np.nan]:
        with pytest.raises(ValueError):
            m.trimmed_mean([1, 2], fraction)
    assert m.trimmed_mean([7], 0.49) == 7


def test_trimmed_mean_invariants():
    x = [-9, 1, 2, 3, 20]
    assert m.trimmed_mean(x, 0) == pytest.approx(m.mean(x))
    assert m.trimmed_mean(x, 0.2) == m.trimmed_mean(x[::-1], 0.2)
    assert m.trimmed_mean(np.array(x) + 10, 0.2) == pytest.approx(12)
