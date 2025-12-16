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


def test_casefold_boundaries():
    assert m.casefold("") == ""
    assert m.casefold("你好🎵") == "你好🎵"
    with pytest.raises(ValueError):
        m.casefold(None)


def test_casefold_invariants():
    for text in ["ABC", "Straße", "Σσς", "İ", "你好"]:
        assert m.casefold(m.casefold(text)) == m.casefold(text)
    assert m.casefold("Σσς") == "σσσ"


def test_collapse_whitespace_example():
    assert m.collapse_whitespace("  one\t two\nthree ") == "one two three"


def test_collapse_whitespace_boundaries():
    assert m.collapse_whitespace(" \t\n") == ""
    assert m.collapse_whitespace("a b") == "a b"
    with pytest.raises(ValueError):
        m.collapse_whitespace(1)


def test_collapse_whitespace_invariants():
    for text in ["", " a  b ", "你好\n世界", "a b"]:
        value = m.collapse_whitespace(text)
        assert m.collapse_whitespace(value) == value
        assert not value.startswith(" ") and not value.endswith(" ") and "  " not in value


def test_strip_punctuation_example():
    assert m.strip_punctuation("hello-world，你好！") == "hello world 你好 "


def test_strip_punctuation_boundaries():
    assert m.strip_punctuation("") == ""
    assert m.strip_punctuation("a+b 🎵") == "a+b 🎵"
    with pytest.raises(ValueError):
        m.strip_punctuation(None)


def test_strip_punctuation_invariants():
    for text in ["a-b", "你好，世界。", "“quoted”", "one/two"]:
        value = m.strip_punctuation(text)
        assert m.strip_punctuation(value) == value
        assert len(value) == len(text)
    assert m.collapse_whitespace(m.strip_punctuation("one--two")) == "one two"


def test_normalize_text_example():
    assert m.normalize_text(" Ｈello， WORLD! ", punctuation=True) == "hello world"


def test_normalize_text_boundaries():
    with pytest.raises(ValueError):
        m.normalize_text("x", fold_case=1)
    with pytest.raises(ValueError):
        m.normalize_text("x", punctuation="yes")
    assert m.normalize_text("HI!", fold_case=False) == "HI!"


def test_normalize_text_invariants():
    for fold in [False, True]:
        for punctuation in [False, True]:
            value = m.normalize_text(" Ａbc，Straße\n你好 ", fold, punctuation)
            assert m.normalize_text(value, fold, punctuation) == value
    assert m.normalize_text("Hello!") == "hello!"


def test_word_tokens_example():
    assert m.word_tokens(" One   TWO ") == ["one", "two"]


def test_word_tokens_boundaries():
    assert m.word_tokens("") == []
    assert m.word_tokens("你好世界") == ["你好世界"]
    with pytest.raises(ValueError):
        m.word_tokens(None)


def test_word_tokens_invariants():
    for text in ["one two", "你好 世界", "a-b c", "Straße ABC"]:
        tokens = m.word_tokens(text)
        assert " ".join(tokens) == m.normalize_text(text)
        assert m.word_tokens(" ".join(tokens)) == tokens
    assert m.word_tokens("one-two", punctuation=True) == ["one", "two"]


def test_character_tokens_example():
    assert m.character_tokens("你 好") == ["你", "好"]


def test_character_tokens_boundaries():
    assert m.character_tokens("") == []
    assert m.character_tokens("a b", False) == ["a", " ", "b"]
    with pytest.raises(ValueError):
        m.character_tokens(None)
    with pytest.raises(ValueError):
        m.character_tokens("a", 1)


def test_character_tokens_invariants():
    for text in ["你好世界", "é", "ＡＢＣ", "🎵🙂"]:
        tokens = m.character_tokens(text)
        assert "".join(tokens) == m.normalize_text(text).replace(" ", "")
        assert all(len(token) == 1 for token in tokens)


def test_edit_alignment_example():
    value = m.edit_alignment(["one", "two"], ["one", "red"])
    assert value["distance"] == 1 and value["matches"] == 1 and value["substitutions"] == 1
