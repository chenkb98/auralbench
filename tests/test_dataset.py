import numpy as np
import pytest
from auralbench import dataset as m


def test_record_example():
    value = m.record({"id": "sample-1"})
    assert value["split"] == "test" and value["reference_audio"] is None


def test_record_boundaries():
    for value in [
        {},
        {"id": "../x"},
        {"id": "x", "extra": 1},
        {"id": "x", "reference_audio": "../a.wav", "candidate_audio": "b.wav"},
        {"id": "x", "reference_audio": "a.wav"},
        {"id": "x", "split": "unknown"},
    ]:
        with pytest.raises(ValueError):
            m.record(value)
