from __future__ import annotations

from typing import Literal

import numpy as np


def _text(value: str) -> str:
    if not isinstance(value, str) or len(value) > 100000:
        raise ValueError("text must be a bounded string")
    return value


def normalize_unicode(text: str, form: Literal["NFC", "NFD", "NFKC", "NFKD"] = "NFKC") -> str:
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


def character_tokens(text: str, remove_spaces: bool = True) -> list[str]:
    """Return normalized Unicode codepoints, optionally excluding spaces."""
    if type(remove_spaces) is not bool:
        raise ValueError("remove_spaces must be boolean")
    value = normalize_text(text)
    return [char for char in value if not remove_spaces or char != " "]


def edit_alignment(reference: list[str], hypothesis: list[str]) -> dict:
    """Compute bounded Levenshtein alignment with deterministic tie breaking."""
    for tokens in [reference, hypothesis]:
        if not isinstance(tokens, (list, tuple)) or any(
            not isinstance(token, str) for token in tokens
        ):
            raise ValueError("alignment needs string token sequences")
    n, k = len(reference), len(hypothesis)
    if (n + 1) * (k + 1) > 1000000:
        raise ValueError("alignment exceeds one million cells; segment long transcripts")
    costs = np.zeros((n + 1, k + 1), dtype=np.int32)
    costs[:, 0], costs[0, :] = np.arange(n + 1), np.arange(k + 1)
    for i in range(1, n + 1):
        for j in range(1, k + 1):
            costs[i, j] = min(
                costs[i - 1, j] + 1,
                costs[i, j - 1] + 1,
                costs[i - 1, j - 1] + int(reference[i - 1] != hypothesis[j - 1]),
            )
    i, j = n, k
    operations: list[dict[str, str | int | None]] = []
    counts = dict(matches=0, substitutions=0, deletions=0, insertions=0)
    while i or j:
        if (
            i
            and j
            and costs[i, j] == costs[i - 1, j - 1] + int(reference[i - 1] != hypothesis[j - 1])
        ):
            same = reference[i - 1] == hypothesis[j - 1]
            counts["matches" if same else "substitutions"] += 1
            operations.append(
                {"op": "match" if same else "substitute", "reference": i - 1, "hypothesis": j - 1}
            )
            i, j = i - 1, j - 1
        elif i and costs[i, j] == costs[i - 1, j] + 1:
            counts["deletions"] += 1
            operations.append({"op": "delete", "reference": i - 1, "hypothesis": None})
            i -= 1
        else:
            counts["insertions"] += 1
            operations.append({"op": "insert", "reference": None, "hypothesis": j - 1})
            j -= 1
    return dict(
        counts,
        distance=int(costs[n, k]),
        reference_length=n,
        hypothesis_length=k,
        operations=list(reversed(operations)),
    )


def word_error_rate(reference: str, hypothesis: str) -> float:
    """WER after the documented default normalization, including empty references."""
    alignment = edit_alignment(word_tokens(reference), word_tokens(hypothesis))
    return alignment["distance"] / max(1, alignment["reference_length"])


def character_error_rate(reference: str, hypothesis: str) -> float:
    """CER over normalized codepoints with whitespace excluded."""
    alignment = edit_alignment(character_tokens(reference), character_tokens(hypothesis))
    return alignment["distance"] / max(1, alignment["reference_length"])
