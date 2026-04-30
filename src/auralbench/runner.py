from __future__ import annotations

import math
import numpy as np
from numpy.typing import ArrayLike, NDArray


from . import audio, dataset, metrics, statistics

_TEXT_FIELDS = {
    "wer",
    "cer",
    "matches",
    "substitutions",
    "deletions",
    "insertions",
    "reference_words",
    "hypothesis_words",
}
_AUDIO_FIELDS = {
    "snr_db",
    "si_sdr_db",
    "spectral_distance_db",
    "rmse",
    "reference_peak",
    "estimate_peak",
    "clipping_fraction",
    "duration_seconds",
}


def _digest(value):
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise ValueError("invalid SHA-256")


def _result(value: dict) -> dict:
    import copy

    if not isinstance(value, dict) or set(value) != {
        "id",
        "speaker",
        "split",
        "input_sha256",
        "text",
        "audio",
    }:
        raise ValueError("invalid result fields")
    dataset.record({key: value[key] for key in ["id", "speaker", "split"]})
    _digest(value["input_sha256"])
    for group, fields in [("text", _TEXT_FIELDS), ("audio", _AUDIO_FIELDS)]:
        if group == "audio" and value[group] is None:
            continue
        if not isinstance(value[group], dict) or set(value[group]) != fields:
            raise ValueError("invalid metric fields")
        for key, number in value[group].items():
            if (
                isinstance(number, bool)
                or not isinstance(number, (int, float))
                or not math.isfinite(number)
            ):
                raise ValueError("metric values must be finite")
            if key not in ["snr_db", "si_sdr_db"] and number < 0:
                raise ValueError("negative error/count metric")
    counts = value["text"]
    for key in _TEXT_FIELDS - {"wer", "cer"}:
        if type(counts[key]) is not int:
            raise ValueError("error counts must be integers")
    if (
        counts["matches"] + counts["substitutions"] + counts["deletions"]
        != counts["reference_words"]
        or counts["matches"] + counts["substitutions"] + counts["insertions"]
        != counts["hypothesis_words"]
    ):
        raise ValueError("inconsistent transcript counts")
    return copy.deepcopy(value)


def _inputs(value: dict, root: str):
    from pathlib import Path
    import hashlib

    directory = Path(root).resolve(strict=True)
    digest = hashlib.sha256(dataset.records_encode([value]).encode())
    payloads = []
    for key in ["reference_audio", "candidate_audio"]:
        if value[key] is None:
            continue
        path = (directory / value[key]).resolve(strict=True)
        if not path.is_relative_to(directory):
            raise ValueError("audio symlink escapes the dataset root")
        if path.stat().st_size > 20000000:
            raise ValueError("audio file exceeds twenty megabytes")
        payload = path.read_bytes()
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
        payloads.append(payload)
    return payloads, digest.hexdigest()


def evaluate_record(record: dict, root: str = ".", cached: dict | None = None) -> dict:
    """Evaluate one pair, reusing a validated cache only when all input bytes match."""
    value = dataset.record(record)
    payloads, checksum = _inputs(value, root)
    if cached is not None:
        previous = _result(cached)
        if any(previous[key] != value[key] for key in ["id", "speaker", "split"]):
            raise ValueError("cached identity differs from the record")
        if previous["input_sha256"] == checksum:
            return previous
    signal = None
    if payloads:
        reference, reference_rate = audio.wav_decode(payloads[0])
        candidate, candidate_rate = audio.wav_decode(payloads[1])
        if reference_rate != candidate_rate:
            raise ValueError("audio sample rates differ; resample explicitly")
        signal = metrics.score_audio(reference, candidate, reference_rate)
    return _result(
        {
            "id": value["id"],
            "speaker": value["speaker"],
            "split": value["split"],
            "input_sha256": checksum,
            "text": metrics.score_transcript(value["reference_text"], value["hypothesis_text"]),
            "audio": signal,
        }
    )


def _run(value: dict) -> dict:
    if (
        not isinstance(value, dict)
        or set(value) != {"schema_version", "dataset_sha256", "results"}
        or type(value["schema_version"]) is not int
        or value["schema_version"] != 1
    ):
        raise ValueError("invalid run schema")
    _digest(value["dataset_sha256"])
    if not isinstance(value["results"], list) or len(value["results"]) > 100000:
        raise ValueError("invalid result list")
    results = [_result(item) for item in value["results"]]
    if len({item["id"] for item in results}) != len(results):
        raise ValueError("duplicate result IDs")
    return dict(value, results=results)


def evaluate_dataset(records: list[dict], root: str = ".", resume: dict | None = None) -> dict:
    """Evaluate in manifest order, optionally resuming a validated partial run."""
    values, checksum = dataset._records(records), dataset.dataset_digest(records)
    cached = {}
    if resume is not None:
        prior = _run(resume)
        if prior["dataset_sha256"] != checksum:
            raise ValueError("resume belongs to a different dataset definition")
        cached = {value["id"]: value for value in prior["results"]}
        if set(cached) - {value["id"] for value in values}:
            raise ValueError("resume contains unknown record IDs")
    results = [evaluate_record(value, root, cached.get(value["id"])) for value in values]
    return {"schema_version": 1, "dataset_sha256": checksum, "results": results}
