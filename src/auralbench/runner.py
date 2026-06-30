from __future__ import annotations

import math

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
    expected_wer = (counts["substitutions"] + counts["deletions"] + counts["insertions"]) / max(
        1, counts["reference_words"]
    )
    if not math.isclose(counts["wer"], expected_wer, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError("WER disagrees with transcript counts")
    signal = value["audio"]
    if signal is not None and (
        signal["clipping_fraction"] > 1
        or any(not -120 <= signal[key] <= 120 for key in ["snr_db", "si_sdr_db"])
    ):
        raise ValueError("audio metrics are outside their documented bounds")
    return copy.deepcopy(value)


def _inputs(value: dict, root: str):
    import hashlib
    from pathlib import Path

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


def aggregate_results(run: dict) -> dict:
    """Report corpus and macro text error rates, plus per-metric audio intervals."""
    rows = _run(run)["results"]
    reference_words = sum(row["text"]["reference_words"] for row in rows)
    errors = sum(
        row["text"][key] for row in rows for key in ["substitutions", "deletions", "insertions"]
    )
    text_summary = {
        "corpus_wer": errors / max(1, reference_words),
        "macro_wer": statistics.mean([row["text"]["wer"] for row in rows]) if rows else None,
        "macro_cer": statistics.mean([row["text"]["cer"] for row in rows]) if rows else None,
        "reference_words": reference_words,
        "hypothesis_words": sum(row["text"]["hypothesis_words"] for row in rows),
    }
    signal_rows = [row["audio"] for row in rows if row["audio"] is not None]
    signal = (
        {
            key: statistics.bootstrap_mean([row[key] for row in signal_rows], samples=200, seed=0)
            for key in sorted(_AUDIO_FIELDS)
        }
        if signal_rows
        else {}
    )
    return {"count": len(rows), "text": text_summary, "audio": signal}


def report_json(run: dict) -> str:
    """Serialize an auditable report containing inputs, row scores and summaries."""
    import json

    value = _run(run)
    return (
        json.dumps(
            dict(value, summary=aggregate_results(value)),
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )


def report_csv(run: dict) -> str:
    """Export a compact row comparison table using standard CSV quoting."""
    import csv
    import io

    rows = _run(run)["results"]
    buffer = io.StringIO(newline="")
    fields = ["id", "speaker", "split", "wer", "cer", "snr_db", "si_sdr_db", "rmse"]
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        signal = row["audio"] or {}
        writer.writerow(
            dict(
                id=row["id"],
                speaker=row["speaker"],
                split=row["split"],
                wer=row["text"]["wer"],
                cer=row["text"]["cer"],
                **{key: signal.get(key, "") for key in ["snr_db", "si_sdr_db", "rmse"]},
            )
        )
    return buffer.getvalue()


def report_html(run: dict, title: str = "AuralBench report") -> str:
    """Render a standalone escaped HTML report with no scripts or external requests."""
    import html

    if not isinstance(title, str) or len(title) > 200:
        raise ValueError("title must be a bounded string")
    value = _run(run)
    summary = aggregate_results(value)
    rows = []
    for row in value["results"]:
        cells = [
            row["id"],
            row["speaker"],
            row["split"],
            f"{row['text']['wer']:.4f}",
            f"{row['text']['cer']:.4f}",
        ]
        rows.append(
            "<tr>" + "".join("<td>" + html.escape(str(cell)) + "</td>" for cell in cells) + "</tr>"
        )
    heading = html.escape(title)
    return (
        '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>'
        + heading
        + "</title><style>body{font:16px system-ui;max-width:1000px;margin:48px auto;padding:0 24px;color:#17243b;background:#f7f8fc}h1{font-size:40px}table{border-collapse:collapse;width:100%;background:white}th,td{text-align:left;padding:12px;border-bottom:1px solid #ddd}code{overflow-wrap:anywhere}small{color:#53627a}</style><h1>"
        + heading
        + "</h1><p>Samples: "
        + str(summary["count"])
        + " · Corpus WER: "
        + f"{summary['text']['corpus_wer']:.4f}"
        + "</p><p><small>Dataset SHA-256: <code>"
        + value["dataset_sha256"]
        + "</code></small></p><table><thead><tr><th>ID</th><th>Speaker</th><th>Split</th><th>WER</th><th>CER</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table><p><small>WER/CER use documented normalization. Signal diagnostics and synthetic fixtures do not establish perceptual model quality.</small></p></html>"
    )
