import numpy as np
import pytest
from auralbench import runner as m


def test_evaluate_record_example():
    value = m.evaluate_record(
        {"id": "a", "reference_text": "one two", "hypothesis_text": "one red"}
    )
    assert value["text"]["wer"] == 0.5 and value["audio"] is None
