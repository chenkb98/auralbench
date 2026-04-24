import numpy as np
import pytest
from auralbench import runner as m


def test_evaluate_record_example():
    value = m.evaluate_record(
        {"id": "a", "reference_text": "one two", "hypothesis_text": "one red"}
    )
    assert value["text"]["wer"] == 0.5 and value["audio"] is None


def test_evaluate_record_boundaries():
    with pytest.raises(ValueError):
        m.evaluate_record({"id": "a", "reference_audio": "../x", "candidate_audio": "y"})
    cached = m.evaluate_record({"id": "a"})
    with pytest.raises(ValueError):
        m.evaluate_record({"id": "b"}, cached=cached)
    cached["text"]["wer"] = np.nan
    with pytest.raises(ValueError):
        m.evaluate_record({"id": "a"}, cached=cached)
