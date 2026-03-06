import numpy as np
import pytest
from auralbench import metrics as m


def test_si_sdr_example():
    assert m.si_sdr([1, -1], [2, -2]) == 120
