import numpy as np
import pytest
from auralbench import text as m


def test_normalize_unicode_example():
    assert m.normalize_unicode("Ａ１２") == "A12"
    assert m.normalize_unicode("e\u0301", "NFC") == "é"


def test_normalize_unicode_boundaries():
    for text in [None, 1, "x" * 100001]:
        with pytest.raises(ValueError):
            m.normalize_unicode(text)
    with pytest.raises(ValueError):
        m.normalize_unicode("text", "bad")
    assert m.normalize_unicode("") == ""


def test_normalize_unicode_invariants():
    for form in ["NFC", "NFD", "NFKC", "NFKD"]:
        for text in ["Ａbc", "é", "你好🎵", "Straße"]:
            value = m.normalize_unicode(text, form)
            assert m.normalize_unicode(value, form) == value
    assert m.normalize_unicode("é", "NFD") == "é"


def test_casefold_example():
    assert m.casefold("Straße ABC") == "strasse abc"
