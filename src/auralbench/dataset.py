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


def _records(values: list[dict]) -> list[dict]:
    if not isinstance(values, list) or len(values) > 100000:
        raise ValueError("dataset must be a bounded record list")
    result = [record(value) for value in values]
    if len({value["id"] for value in result}) != len(result):
        raise ValueError("duplicate record IDs")
    return result


def records_encode(records: list[dict]) -> str:
    """Write normalized, unique records as deterministic JSONL."""
    import json

    values = _records(records)
    result = "".join(
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        + "\n"
        for value in values
    )
    if len(result) > 20000000:
        raise ValueError("manifest exceeds twenty million characters")
    return result


def _json(text: str):
    import json

    def pairs(items):
        value = {}
        for key, item in items:
            if key in value:
                raise ValueError("duplicate JSON field")
            value[key] = item
        return value

    def constant(value):
        raise ValueError("non-finite JSON constant")

    return json.loads(text, object_pairs_hook=pairs, parse_constant=constant)


def records_decode(payload: str) -> list[dict]:
    """Read complete JSONL records with line-numbered validation failures."""
    if not isinstance(payload, str) or len(payload) > 20000000:
        raise ValueError("invalid dataset payload")
    result = []
    for index, line in enumerate(payload.splitlines(), 1):
        if not line.strip():
            continue
        try:
            raw = _json(line)
            value = record(raw)
            if set(raw) != set(value):
                raise ValueError("persisted records require all schema fields")
            result.append(value)
        except (ValueError, TypeError) as error:
            raise ValueError(f"invalid dataset line {index}: {error}") from error
    return _records(result)
