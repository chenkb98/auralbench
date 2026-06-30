from pathlib import Path

import numpy as np

from auralbench.audio import wav_encode
from auralbench.runner import evaluate_dataset, report_html

output = Path("outputs/audio")
output.mkdir(parents=True, exist_ok=True)
signal = 0.25 * np.sin(2 * np.pi * 440 * np.arange(800) / 16000)
candidate = signal + np.random.default_rng(7).normal(0, 0.01, len(signal))
(output / "reference.wav").write_bytes(wav_encode(signal, 16000))
(output / "candidate.wav").write_bytes(wav_encode(candidate, 16000))
run = evaluate_dataset(
    [
        {
            "id": "synthetic-tone",
            "reference_audio": "reference.wav",
            "candidate_audio": "candidate.wav",
        }
    ],
    str(output),
)
(output / "report.html").write_text(report_html(run, "Synthetic signal diagnostics"))
print(output / "report.html")
