import numpy as np
import pytest
from auralbench import metrics as m


def test_si_sdr_example():
    assert m.si_sdr([1, -1], [2, -2]) == 120


def test_si_sdr_boundaries():
    assert m.si_sdr([], []) == 120
    assert m.si_sdr([0, 0], [0, 0]) == 120
    assert m.si_sdr([1, -1], [0, 0]) == -120
    assert m.si_sdr([0, 0], [1, -1]) == -120
    with pytest.raises(ValueError):
        m.si_sdr([1], [1, 2])
