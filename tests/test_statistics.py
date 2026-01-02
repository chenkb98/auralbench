import numpy as np
import pytest
from auralbench import statistics as m


def test_finite_values_example():
    np.testing.assert_array_equal(m.finite_values([1, 2]), [1.0, 2.0])
