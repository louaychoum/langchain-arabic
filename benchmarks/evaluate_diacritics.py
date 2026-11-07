"""Reproduce the offline smoke evaluation; optional CATT runs require --catt."""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path

from langchain_arabic import ArabicTextOutputParser, apply_diacritics, convert_numbers_in_text

_DATA = Path(__file__).with_name("data")
_RESULTS = Path(__file__).with_name("results.md")
DICTIONARY = {
    "شركة": "شَرِكَةٌ",
    "تقنية": "تِقْنِيَّةٌ",
    "كبيرة": "كَبِيرَةٌ",
    "علم الحاسوب": "عِلْمُ الْحَاسُوبِ",
}
_DIACRITICS = set("\u064b\u064c\u064d\u064e\u064f\u0650\u0651\u0652\u0670")


def _get_diacritic_pairs(text: str) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for char in unicodedata.normalize("NFC", text):
        if char in _DIACRITICS:
            if not pairs:
                raise ValueError("Diacritic without a base character")
            pairs[-1] = (pairs[-1][0], pairs[-1][1] + char)
        else:
            pairs.append((char, ""))
    return pairs


def _is_arabic_letter(char: str) -> bool:
    return unicodedata.category(char).startswith("L") and "ARABIC" in unicodedata.name(char, "")


def compute_der(predicted: str, reference: str) -> float:
    """Wrong mark sequences / Arabic letters, including reference case endings.

    Canonically equivalent combining marks compare equal. Base text must match
    exactly; deletion or alteration raises instead of silently improving DER.
    """
    pred_pairs = _get_diacritic_pairs(predicted)
    ref_pairs = _get_diacritic_pairs(reference)
    if [p[0] for p in pred_pairs] != [r[0] for r in ref_pairs]:
        raise ValueError("DER requires identical base text; prediction changed or lost content")
    eligible = [(p, r) for p, r in zip(pred_pairs, ref_pairs) if _is_arabic_letter(r[0])]
    return sum(p[1] != r[1] for p, r in eligible) / len(eligible) if eligible else 0.0


def compute_wer(predicted: str, reference: str) -> float:
    """Word edit distance / reference words; insertions can make WER exceed 1."""
    predicted_words = unicodedata.normalize("NFC", predicted).split()
    reference_words = unicodedata.normalize("NFC", reference).split()
    row = list(range(len(reference_words) + 1))
    for i, word in enumerate(predicted_words, 1):
        new = [i]
        for j, ref in enumerate(reference_words, 1):
            new.append(min(new[-1] + 1, row[j] + 1, row[j - 1] + (word != ref)))
        row = new
    return row[-1] / max(1, len(reference_words))


def evaluate(mode: str) -> tuple[float, float]:
    corpus = json.loads((_DATA / "diacritics.json").read_text(encoding="utf-8"))
    parser = None
    if mode != "dictionary":
        parser = ArabicTextOutputParser(backend="catt", catt_model=mode, convert_numbers=False)
    errors, letters, word_errors, words = 0.0, 0, 0.0, 0
    for text, reference in corpus:
        assert re.sub("[" + "".join(_DIACRITICS) + "]", "", reference) == text
        predicted = parser.parse(text) if parser else apply_diacritics(text, DICTIONARY)
        letter_count = sum(_is_arabic_letter(c) for c in text)
        word_count = len(reference.split())
        errors += compute_der(predicted, reference) * letter_count
        letters += letter_count
        word_errors += compute_wer(predicted, reference) * word_count
        words += word_count
    return errors / letters, word_errors / words


def report(catt: bool = False) -> str:
    der, wer = evaluate("dictionary")
    lines = [
        "# Smoke evaluation results",
        "",
        "Hand-authored MIT-licensed examples: 12 diacritization pairs and 24 numeric cases.",
        "These scores describe this small set only; they are not a general accuracy claim.",
        "",
        "| Diacritization mode | DER | WER |",
        "|---|---:|---:|",
        f"| Dictionary (4 entries) | {der:.2%} | {wer:.2%} |",
    ]
    if catt:
        for model in ("encoder_only", "encoder_decoder"):
            der, wer = evaluate(model)
            lines.append(f"| CATT {model} | {der:.2%} | {wer:.2%} |")
    else:
        lines += ["", "CATT was not run. No neural accuracy or speed is claimed."]
    cases = json.loads((_DATA / "numbers.json").read_text(encoding="utf-8"))
    correct = 0
    for case in cases:
        prediction = convert_numbers_in_text(
            case["input"], case.get("language", "ar"), dialect=case.get("dialect", "msa")
        )
        correct += prediction == case["expected"]
    lines += [
        "",
        "| Number conversion | Exact matches | Accuracy |",
        "|---|---:|---:|",
        f"| Hand-checked cases | {correct}/{len(cases)} | {correct / len(cases):.2%} |",
        "",
        "See [methodology](README.md) and [data provenance/license](data/README.md).",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--catt", action="store_true", help="Run neural models (may download weights)")
    cli.add_argument("--output", type=Path, help="Write Markdown results")
    cli.add_argument("--check", action="store_true", help="Check committed offline results")
    args = cli.parse_args()
    if args.check and (args.catt or args.output):
        cli.error("--check must run offline without --catt or --output")
    result = report(args.catt)
    if args.check:
        if result != _RESULTS.read_text(encoding="utf-8"):
            raise SystemExit("Benchmark results changed; rerun with --output benchmarks/results.md")
        print("Committed benchmark results match.")
    elif args.output:
        args.output.write_text(result, encoding="utf-8")
    else:
        print(result, end="")


if __name__ == "__main__":
    main()
