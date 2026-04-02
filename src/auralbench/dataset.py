from __future__ import annotations

import math
import numpy as np
from numpy.typing import ArrayLike, NDArray


from . import text


def record(value: dict) -> dict:
    """Validate a local paired-record schema with portable relative audio paths."""
    import re
    from pathlib import PurePosixPath

    defaults = dict(
        reference_text="",
        hypothesis_text="",
        reference_audio=None,
        candidate_audio=None,
        speaker="",
        split="test",
    )
    if not isinstance(value, dict) or "id" not in value or set(value) - set(defaults) - {"id"}:
        raise ValueError("invalid record fields")
    result = dict(defaults, **value)
    if (
        not isinstance(result["id"], str)
        or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", result["id"]) is None
    ):
        raise ValueError("invalid record ID")
    for key in ["reference_text", "hypothesis_text", "speaker"]:
        text._text(result[key])
    if result["split"] not in ["train", "validation", "test"]:
        raise ValueError("invalid split")
    for key in ["reference_audio", "candidate_audio"]:
        path = result[key]
        if path is not None:
            if (
                not isinstance(path, str)
                or not path
                or len(path) > 4096
                or chr(92) in path
                or ":" in path
            ):
                raise ValueError("audio paths must be portable relative paths")
            parts = PurePosixPath(path)
            if parts.is_absolute() or ".." in parts.parts or str(parts) == ".":
                raise ValueError("audio paths cannot leave the dataset root")
            result[key] = str(parts)
    if (result["reference_audio"] is None) != (result["candidate_audio"] is None):
        raise ValueError("audio paths must be paired")
    return result
