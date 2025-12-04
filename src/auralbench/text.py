from __future__ import annotations

import math
import numpy as np
from numpy.typing import ArrayLike, NDArray


def _text(value: str) -> str:
    if not isinstance(value, str) or len(value) > 100000:
        raise ValueError("text must be a bounded string")
    return value


def normalize_unicode(text: str, form: str = "NFKC") -> str:
    """Normalize Unicode using an explicitly selected standard form."""
    import unicodedata

    if form not in ["NFC", "NFD", "NFKC", "NFKD"]:
        raise ValueError("unsupported normalization form")
    return unicodedata.normalize(form, _text(text))


def casefold(text: str) -> str:
    """Apply Unicode case folding rather than ASCII-only lowercasing."""
    return _text(text).casefold()


def collapse_whitespace(text: str) -> str:
    """Trim and join runs of Unicode whitespace into single ASCII spaces."""
    return " ".join(_text(text).split())


def strip_punctuation(text: str) -> str:
    """Replace Unicode punctuation with spaces, preserving token boundaries."""
    import unicodedata

    return "".join(
        " " if unicodedata.category(char).startswith("P") else char for char in _text(text)
    )


def normalize_text(text: str, fold_case: bool = True, punctuation: bool = False) -> str:
    """Apply a documented NFKC, optional case/punctuation, whitespace pipeline."""
    if type(fold_case) is not bool or type(punctuation) is not bool:
        raise ValueError("normalization flags must be boolean")
    value = normalize_unicode(text)
    if fold_case:
        value = casefold(value)
    if punctuation:
        value = strip_punctuation(value)
    return collapse_whitespace(value)


def word_tokens(text: str, fold_case: bool = True, punctuation: bool = False) -> list[str]:
    """Split normalized text on whitespace; no language-specific segmentation."""
    return normalize_text(text, fold_case, punctuation).split()
