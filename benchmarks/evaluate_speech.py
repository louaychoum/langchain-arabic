"""Offline exact-output checks for synthetic Arabic speech-formatting fixtures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from langchain_arabic import ArabicSpeechNormalizer

_DATA = Path(__file__).with_name("data") / "speech.json"
_RESULTS = Path(__file__).with_name("speech-results.md")


def report() -> tuple[str, bool]:
    """Return measured fixture counts and whether every expected string matched."""
    counts: dict[str, list[int]] = {}
    failed: list[str] = []
    for case in json.loads(_DATA.read_text(encoding="utf-8")):
        category = counts.setdefault(case["category"], [0, 0])
        predicted = ArabicSpeechNormalizer(**case.get("options", {})).process(case["input"])
        category[1] += 1
        category[0] += predicted == case["expected"]
        if predicted != case["expected"]:
            failed.append(case["input"])
    lines = [
        "# Speech normalization fixture results",
        "",
        "Repository-authored synthetic MIT-licensed fixtures; no external corpus or independent",
        "native-speaker review. Exact text matches include intentionally retained structures.",
        "These are regression checks, not speech quality, ASR, or general accuracy scores.",
        "",
        "| Fixture category | Exact matches |",
        "|---|---:|",
    ]
    for name, (correct, total) in counts.items():
        lines.append(f"| {name} | {correct}/{total} |")
    lines += [
        f"| Total | {sum(c[0] for c in counts.values())}/{sum(c[1] for c in counts.values())} |",
        "",
        "Reproduce: `python benchmarks/evaluate_speech.py --check`.",
        "See [scope and limits](../docs/speech-normalization.md) and [data license](data/README.md).",
        "",
    ]
    if failed:
        lines += ["Failed inputs:", "", *(f"- `{text}`" for text in failed), ""]
    return "\n".join(lines), not failed


def main() -> None:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--output", type=Path, help="Write Markdown results")
    cli.add_argument("--check", action="store_true", help="Check committed offline results")
    args = cli.parse_args()
    if args.check and args.output:
        cli.error("--check cannot be combined with --output")
    result, passed = report()
    if args.check:
        if result != _RESULTS.read_text(encoding="utf-8"):
            raise SystemExit(
                "Speech results changed; rerun with --output benchmarks/speech-results.md"
            )
        print("Committed speech fixture results match.")
    elif args.output:
        args.output.write_text(result, encoding="utf-8")
    else:
        print(result, end="")
    if not passed:
        raise SystemExit("Speech fixture mismatch")


if __name__ == "__main__":
    main()
