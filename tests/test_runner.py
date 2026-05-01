import numpy as np
import pytest
from auralbench import runner as m


def test_evaluate_record_example():
    value = m.evaluate_record(
        {"id": "a", "reference_text": "one two", "hypothesis_text": "one red"}
    )
    assert value["text"]["wer"] == 0.5 and value["audio"] is None


def test_evaluate_record_boundaries():
    with pytest.raises(ValueError):
        m.evaluate_record({"id": "a", "reference_audio": "../x", "candidate_audio": "y"})
    cached = m.evaluate_record({"id": "a"})
    with pytest.raises(ValueError):
        m.evaluate_record({"id": "b"}, cached=cached)
    cached["text"]["wer"] = np.nan
    with pytest.raises(ValueError):
        m.evaluate_record({"id": "a"}, cached=cached)


def test_evaluate_record_invariants():
    import tempfile
    from pathlib import Path
    from auralbench import audio

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        (root / "a.wav").write_bytes(audio.wav_encode([0.5, -0.5], 8000))
        (root / "b.wav").write_bytes(audio.wav_encode([0.5, -0.5], 8000))
        record = {"id": "pair", "reference_audio": "a.wav", "candidate_audio": "b.wav"}
        first = m.evaluate_record(record, temporary)
        assert first == m.evaluate_record(record, temporary, first)
        (root / "b.wav").write_bytes(audio.wav_encode([0.25, -0.25], 8000))
        second = m.evaluate_record(record, temporary, first)
        assert second["input_sha256"] != first["input_sha256"] and second["audio"]["rmse"] > 0


def test_evaluate_dataset_example():
    value = m.evaluate_dataset(
        [{"id": "a"}, {"id": "b", "reference_text": "x", "hypothesis_text": "y"}]
    )
    assert [row["id"] for row in value["results"]] == ["a", "b"]


def test_evaluate_dataset_boundaries():
    with pytest.raises(ValueError):
        m.evaluate_dataset([{"id": "a"}, {"id": "a"}])
    run = m.evaluate_dataset([{"id": "a"}])
    with pytest.raises(ValueError):
        m.evaluate_dataset([{"id": "b"}], resume=run)
    assert m.evaluate_dataset([])["results"] == []


def test_evaluate_dataset_invariants():
    values = [{"id": "a"}, {"id": "b"}, {"id": "c"}]
    full = m.evaluate_dataset(values)
    partial = dict(full, results=full["results"][:1])
    assert m.evaluate_dataset(values, resume=partial) == full
    reversed_run = m.evaluate_dataset(values[::-1], resume=full)
    assert [row["id"] for row in reversed_run["results"]] == ["c", "b", "a"]
    assert set(full) == {"schema_version", "dataset_sha256", "results"}


def test_aggregate_results_example():
    run = m.evaluate_dataset(
        [
            {"id": "a", "reference_text": "one two", "hypothesis_text": "one"},
            {"id": "b", "reference_text": "one", "hypothesis_text": ""},
        ]
    )
    value = m.aggregate_results(run)
    assert (
        value["text"]["corpus_wer"] == pytest.approx(2 / 3) and value["text"]["macro_wer"] == 0.75
    )


def test_aggregate_results_boundaries():
    empty = m.aggregate_results(m.evaluate_dataset([]))
    assert empty["count"] == 0 and empty["text"]["macro_wer"] is None and empty["audio"] == {}
    with pytest.raises(ValueError):
        m.aggregate_results({})


def test_aggregate_results_invariants():
    values = [
        {"id": "a", "reference_text": "a b", "hypothesis_text": "a"},
        {"id": "b", "reference_text": "a", "hypothesis_text": "a"},
    ]
    summary = m.aggregate_results(m.evaluate_dataset(values))
    assert summary == m.aggregate_results(m.evaluate_dataset(values[::-1]))
    assert set(summary) == {"count", "text", "audio"}
    assert summary["text"]["reference_words"] == 3
