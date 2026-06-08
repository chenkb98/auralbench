from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import __version__, dataset, runner


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="auralbench")
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)
    evaluate = commands.add_parser("evaluate", help="Evaluate a local paired JSONL manifest")
    evaluate.add_argument("manifest")
    evaluate.add_argument("--root", default=".")
    evaluate.add_argument("--resume")
    evaluate.add_argument("--output", required=True)
    report = commands.add_parser("report", help="Render an existing run checkpoint")
    report.add_argument("checkpoint")
    report.add_argument("--output", required=True)
    demo = commands.add_parser("demo", help="Write a synthetic transcript evaluation report")
    demo.add_argument("--output", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "evaluate":
            records = dataset.records_decode(Path(args.manifest).read_text())
            resume = dataset._json(Path(args.resume).read_text()) if args.resume else None
            run = runner.evaluate_dataset(records, args.root, resume)
        elif args.command == "report":
            run = runner._run(dataset._json(Path(args.checkpoint).read_text()))
        else:
            run = runner.evaluate_dataset(
                [
                    {
                        "id": "synthetic-1",
                        "reference_text": "你好 世界",
                        "hypothesis_text": "你好 世界",
                    },
                    {
                        "id": "synthetic-2",
                        "reference_text": "one two three",
                        "hypothesis_text": "one three",
                    },
                ]
            )
        artifacts = {
            "checkpoint.json": json.dumps(run, ensure_ascii=False, indent=2, allow_nan=False)
            + "\n",
            "report.json": runner.report_json(run),
            "report.csv": runner.report_csv(run),
            "report.html": runner.report_html(run),
        }
        output = Path(args.output)
        output.mkdir(parents=True, exist_ok=False)
        for name, contents in artifacts.items():
            (output / name).write_text(contents)
        print(
            json.dumps(
                {"output": str(output), "count": len(run["results"]), "files": sorted(artifacts)},
                ensure_ascii=False,
            )
        )
        return 0
    except (ValueError, OSError) as error:
        parser.error(str(error))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
