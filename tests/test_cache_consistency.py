import copy

import pytest

from auralbench import audio, runner


def test_resume_and_all_reports_reject_inconsistent_word_error_rates():
    records = [{"id": "a", "reference_text": "one two", "hypothesis_text": "one"}]
    run = runner.evaluate_dataset(records)
    changed = copy.deepcopy(run)
    changed["results"][0]["text"]["wer"] = 0.25
    for output in [
        runner.aggregate_results,
        runner.report_json,
        runner.report_csv,
        runner.report_html,
    ]:
        with pytest.raises(ValueError):
            output(changed)
    with pytest.raises(ValueError):
        runner.evaluate_dataset(records, resume=changed)


@pytest.mark.parametrize(
    "key, value",
    [
        ("clipping_fraction", 1.01),
        ("snr_db", -121),
        ("snr_db", 121),
        ("si_sdr_db", -121),
        ("si_sdr_db", 121),
    ],
)
def test_cached_audio_metrics_respect_all_documented_bounds(tmp_path, key, value):
    (tmp_path / "a.wav").write_bytes(audio.wav_encode([0.1, -0.1], 16000))
    run = runner.evaluate_dataset(
        [{"id": "a", "reference_audio": "a.wav", "candidate_audio": "a.wav"}], str(tmp_path)
    )
    run["results"][0]["audio"][key] = value
    with pytest.raises(ValueError):
        runner.report_json(run)
