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
