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


def test_si_sdr_invariants():
    a, b = np.array([1.0, -1.0, 0.5, -0.5]), np.array([0.9, -0.8, 0.6, -0.7])
    value = m.si_sdr(a, b)
    for scale in [-3, 0.01, 4]:
        assert m.si_sdr(a, b * scale) == pytest.approx(value)
    np.testing.assert_array_equal(a, [1, -1, 0.5, -0.5])
    np.testing.assert_array_equal(b, [0.9, -0.8, 0.6, -0.7])


def test_log_spectral_distance_example():
    assert m.log_spectral_distance([1], [0.1], 8000) == pytest.approx(20)


def test_log_spectral_distance_boundaries():
    assert m.log_spectral_distance([], [], 8000) == 0
    assert m.log_spectral_distance([0], [0], 8000) == 0
    with pytest.raises(ValueError):
        m.log_spectral_distance([1], [1, 2], 8000)
    with pytest.raises(ValueError):
        m.log_spectral_distance([1], [1], 8000, 0)


def test_log_spectral_distance_invariants():
    a, b = [1.0, 0, -1, 0], [0.8, 0.1, -0.7, -0.2]
    assert m.log_spectral_distance(a, a, 8000) == 0
    assert m.log_spectral_distance(a, b, 8000) == pytest.approx(m.log_spectral_distance(b, a, 8000))
    assert m.log_spectral_distance(a, b, 8000) >= 0


def test_duration_error_example():
    assert m.duration_error([0] * 8, 8, [0] * 12, 8) == 0.5
