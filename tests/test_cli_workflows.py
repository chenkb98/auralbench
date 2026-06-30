import json

import pytest

from auralbench import cli, dataset


@pytest.fixture
def command(capsys):
    def invoke(*arguments):
        assert cli.main(list(arguments)) == 0
        return json.loads(capsys.readouterr().out)

    return invoke


def test_demo_report_roundtrip_and_no_overwrite(command, tmp_path):
    target, copied = tmp_path / "demo", tmp_path / "copied"
    result = command("demo", "--output", str(target))
    assert result["count"] == 2
    assert result["files"] == ["checkpoint.json", "report.csv", "report.html", "report.json"]
    report = json.loads((target / "report.json").read_text())
    assert report["summary"]["text"]["corpus_wer"] == pytest.approx(0.2)
    command("report", str(target / "checkpoint.json"), "--output", str(copied))
    for name in result["files"]:
        assert (target / name).read_bytes() == (copied / name).read_bytes()
    with pytest.raises(SystemExit) as error:
        cli.main(["demo", "--output", str(target)])
    assert error.value.code == 2


def test_evaluate_and_resume_produce_identical_reports(command, tmp_path):
    manifest = tmp_path / "data.jsonl"
    manifest.write_text(
        dataset.records_encode(
            [{"id": "one", "reference_text": "one", "hypothesis_text": "one two"}]
        )
    )
    target, resumed = tmp_path / "run", tmp_path / "resumed"
    result = command("evaluate", str(manifest), "--root", str(tmp_path), "--output", str(target))
    assert result["count"] == 1
    command(
        "evaluate",
        str(manifest),
        "--root",
        str(tmp_path),
        "--resume",
        str(target / "checkpoint.json"),
        "--output",
        str(resumed),
    )
    assert (target / "report.json").read_bytes() == (resumed / "report.json").read_bytes()


def test_invalid_manifest_does_not_create_output(tmp_path):
    manifest, target = tmp_path / "bad.jsonl", tmp_path / "run"
    manifest.write_text("{invalid}")
    with pytest.raises(SystemExit) as error:
        cli.main(["evaluate", str(manifest), "--output", str(target)])
    assert error.value.code == 2 and not target.exists()
