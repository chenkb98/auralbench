import numpy as np
import pytest
from auralbench import dataset as m


def test_record_example():
    value = m.record({"id": "sample-1"})
    assert value["split"] == "test" and value["reference_audio"] is None
