from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike

from . import audio, text


def si_sdr(reference: ArrayLike, estimate: ArrayLike, limit_db: float = 120.0) -> float:
    """Zero-mean scale-invariant SDR on the flattened paired sample vectors."""
    a, b = audio.as_audio(reference), audio.as_audio(estimate)
    if a.shape != b.shape or not np.isfinite(limit_db) or not 0 < limit_db <= 300:
        raise ValueError("invalid paired shapes or dB limit")
    if not a.size:
        return float(limit_db)
    a, b = a.ravel(), b.ravel()
    for value in [a, b]:
        scale = float(np.max(np.abs(value)))
        if scale:
            value /= scale
        value -= value.mean()
    reference_power = float(a @ a)
    if reference_power == 0:
        return float(limit_db if not np.any(b) else -limit_db)
    if not np.any(b):
        return -float(limit_db)
    target = float((a @ b) / reference_power) * a
    signal, error = audio.rms(target), audio.rms(b - target)
    if signal == 0:
        return -float(limit_db)
    if error == 0:
        return float(limit_db)
    return float(np.clip(20 * (math.log10(signal) - math.log10(error)), -limit_db, limit_db))


def log_spectral_distance(
    reference: ArrayLike, estimate: ArrayLike, rate: int, epsilon: float = 1e-8
) -> float:
    """RMS dB difference between paired one-sided amplitude spectra."""
    a, b = audio.as_audio(reference), audio.as_audio(estimate)
    if a.shape != b.shape or not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("invalid paired shapes or spectral floor")
    _, left = audio.spectrum(a, rate)
    _, right = audio.spectrum(b, rate)
    if not left.size:
        return 0.0
    difference = 20 * (np.log10(np.maximum(left, epsilon)) - np.log10(np.maximum(right, epsilon)))
    return audio.rms(difference)


def duration_error(
    reference: ArrayLike, reference_rate: int, estimate: ArrayLike, estimate_rate: int
) -> float:
    """Signed duration difference in seconds: estimate minus reference."""
    return audio.duration(estimate, estimate_rate) - audio.duration(reference, reference_rate)


def score_transcript(reference: str, hypothesis: str) -> dict:
    """Report WER/CER alongside the complete compact word error counts."""
    alignment = text.edit_alignment(text.word_tokens(reference), text.word_tokens(hypothesis))
    return {
        "wer": alignment["distance"] / max(1, alignment["reference_length"]),
        "cer": text.character_error_rate(reference, hypothesis),
        "matches": alignment["matches"],
        "substitutions": alignment["substitutions"],
        "deletions": alignment["deletions"],
        "insertions": alignment["insertions"],
        "reference_words": alignment["reference_length"],
        "hypothesis_words": alignment["hypothesis_length"],
    }


def score_audio(reference: ArrayLike, estimate: ArrayLike, rate: int) -> dict:
    """Compute paired signal diagnostics without automatic alignment or resampling."""
    a, b = audio.as_audio(reference), audio.as_audio(estimate)
    rate = audio.sample_rate(rate)
    if a.shape != b.shape:
        raise ValueError("paired audio shapes must match")
    return {
        "snr_db": audio.snr(a, b),
        "si_sdr_db": si_sdr(a, b),
        "spectral_distance_db": log_spectral_distance(a, b, rate),
        "rmse": audio.rms(a - b),
        "reference_peak": audio.peak(a),
        "estimate_peak": audio.peak(b),
        "clipping_fraction": audio.clipping_fraction(b),
        "duration_seconds": audio.duration(a, rate),
    }
