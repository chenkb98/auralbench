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
