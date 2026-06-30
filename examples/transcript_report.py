from pathlib import Path

from auralbench.runner import evaluate_dataset, report_csv, report_html, report_json

run = evaluate_dataset(
    [
        {"id": "synthetic-cn", "reference_text": "你好 世界", "hypothesis_text": "你好 世间"},
        {"id": "synthetic-en", "reference_text": "one two three", "hypothesis_text": "one three"},
    ]
)
output = Path("outputs")
output.mkdir(exist_ok=True)
for name, contents in [
    ("report.html", report_html(run)),
    ("report.json", report_json(run)),
    ("report.csv", report_csv(run)),
]:
    (output / name).write_text(contents)
print(output / "report.html")
