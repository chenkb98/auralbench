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


def test_record_invariants():
    value = m.record({"id": "x", "reference_text": "你好", "hypothesis_text": "你好"})
    assert set(value) == {
        "id",
        "reference_text",
        "hypothesis_text",
        "reference_audio",
        "candidate_audio",
        "speaker",
        "split",
    }
    assert m.record(value) == value
    for path in ["/a.wav", "C:/a.wav", "../a.wav", "a/../../b.wav"]:
        with pytest.raises(ValueError):
            m.record({"id": "x", "reference_audio": path, "candidate_audio": "b.wav"})


def test_records_encode_example():
    assert m.records_encode([{"id": "a"}]).endswith("\n")
