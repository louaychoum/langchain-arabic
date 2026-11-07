"""Evaluate diacritization accuracy across different modes.

Computes Diacritization Error Rate (DER) and Word Error Rate (WER) on a small
curated Arabic test set.

Run with:
    python benchmarks/evaluate_diacritics.py

CATT benchmarks are skipped gracefully if catt-tashkeel is not installed.
"""

from __future__ import annotations

import time

from langchain_arabic import ArabicTextOutputParser, apply_diacritics

# ---------------------------------------------------------------------------
# Test corpus: (undiacritized input, expected fully-diacritized output)
#
# These are manually curated examples — not a substitute for a full academic
# benchmark (e.g. Tashkeela corpus), but sufficient to demonstrate the
# evaluation methodology and compare modes.
# ---------------------------------------------------------------------------
TEST_CORPUS = [
    (
        "شركة تقنية كبيرة",
        "شَرِكَة تِقْنِيَة كَبِيرَة",
    ),
    (
        "الطالب يدرس في الجامعة",
        "الطَّالِبُ يَدْرُسُ فِي الجَامِعَةِ",
    ),
    (
        "المملكة العربية السعودية",
        "المَمْلَكَة العَرَبِيَّة السُّعُودِيَّة",
    ),
    (
        "بسم الله الرحمن الرحيم",
        "بِسْمِ اللَّهِ الرَّحْمَنِ الرَّحِيمِ",
    ),
    (
        "الحمد لله رب العالمين",
        "الحَمْدُ لِلَّهِ رَبِّ العَالَمِينَ",
    ),
    (
        "علم الحاسوب مهم جدا",
        "عِلْمُ الحَاسُوبِ مُهِمٌّ جِدًّا",
    ),
    (
        "الذكاء الاصطناعي يتطور",
        "الذَّكَاءُ الاصْطِنَاعِيُّ يَتَطَوَّرُ",
    ),
    (
        "تعلم اللغة العربية",
        "تَعَلُّمُ اللُّغَةِ العَرَبِيَّةِ",
    ),
]

# Dictionary for dictionary-only mode
DICTIONARY = {
    "شركة": "شَرِكَة",
    "تقنية": "تِقْنِيَة",
    "علم الحاسوب": "عِلْمُ الحَاسُوبِ",
    "كبيرة": "كَبِيرَة",
}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

_DIACRITICS = set("\u064b\u064c\u064d\u064e\u064f\u0650\u0651\u0652\u0670")


def _get_diacritic_pairs(text: str) -> list[tuple[str, str]]:
    """Return (base_char, diacritic_or_empty) pairs for DER calculation."""
    pairs: list[tuple[str, str]] = []
    base = None
    for char in text:
        if char in _DIACRITICS:
            if base is not None:
                pairs[-1] = (pairs[-1][0], pairs[-1][1] + char)
        else:
            pairs.append((char, ""))
            base = char
    return pairs


def compute_der(predicted: str, reference: str) -> float:
    """Diacritization Error Rate: fraction of characters with wrong diacritics.

    Only compares positions where the base characters match (skips alignment
    issues).  Returns 0.0 for perfect match, 1.0 for all wrong.
    """
    pred_pairs = _get_diacritic_pairs(predicted)
    ref_pairs = _get_diacritic_pairs(reference)

    total = 0
    errors = 0
    for (pc, pd), (rc, rd) in zip(pred_pairs, ref_pairs):
        if pc != rc:
            continue  # skip misaligned characters
        total += 1
        if pd != rd:
            errors += 1

    return errors / total if total > 0 else 0.0


def compute_wer(predicted: str, reference: str) -> float:
    """Word Error Rate: fraction of words where diacritization differs."""
    pred_words = predicted.split()
    ref_words = reference.split()

    total = max(len(pred_words), len(ref_words))
    if total == 0:
        return 0.0

    errors = 0
    for pw, rw in zip(pred_words, ref_words):
        if pw != rw:
            errors += 1
    errors += abs(len(pred_words) - len(ref_words))

    return errors / total


# ---------------------------------------------------------------------------
# Evaluation runners
# ---------------------------------------------------------------------------


def evaluate_dictionary_mode() -> tuple[float, float, float]:
    """Evaluate dictionary-only diacritization."""
    ders, wers = [], []
    start = time.perf_counter()

    for undiacritized, reference in TEST_CORPUS:
        predicted = apply_diacritics(undiacritized, DICTIONARY)
        ders.append(compute_der(predicted, reference))
        wers.append(compute_wer(predicted, reference))

    elapsed = time.perf_counter() - start
    return sum(ders) / len(ders), sum(wers) / len(wers), elapsed


def evaluate_catt_mode(model: str) -> tuple[float, float, float] | None:
    """Evaluate CATT auto-diacritization. Returns None if CATT not installed."""
    parser = ArabicTextOutputParser(
        backend="catt",
        catt_model=model,
        convert_numbers=False,
    )

    ders, wers = [], []
    start = time.perf_counter()

    try:
        for undiacritized, reference in TEST_CORPUS:
            predicted = parser.parse(undiacritized)
            ders.append(compute_der(predicted, reference))
            wers.append(compute_wer(predicted, reference))
    except ImportError:
        return None

    elapsed = time.perf_counter() - start
    return sum(ders) / len(ders), sum(wers) / len(wers), elapsed


def evaluate_hybrid_mode(model: str) -> tuple[float, float, float] | None:
    """Evaluate CATT + dictionary overrides. Returns None if CATT not installed."""
    parser = ArabicTextOutputParser(
        backend="catt",
        catt_model=model,
        diacritics_map=DICTIONARY,
        convert_numbers=False,
    )

    ders, wers = [], []
    start = time.perf_counter()

    try:
        for undiacritized, reference in TEST_CORPUS:
            predicted = parser.parse(undiacritized)
            ders.append(compute_der(predicted, reference))
            wers.append(compute_wer(predicted, reference))
    except ImportError:
        return None

    elapsed = time.perf_counter() - start
    return sum(ders) / len(ders), sum(wers) / len(wers), elapsed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print(f"Corpus: {len(TEST_CORPUS)} sentence pairs")
    print()

    header = f"{'Mode':<30} {'DER':>8} {'WER':>8} {'Time':>10}"
    print(header)
    print("-" * len(header))

    # Dictionary mode (always available)
    der, wer, elapsed = evaluate_dictionary_mode()
    print(f"{'Dictionary only':<30} {der:>7.1%} {wer:>7.1%} {elapsed:>9.3f}s")

    # CATT modes (optional)
    for model in ("encoder_only", "encoder_decoder"):
        result = evaluate_catt_mode(model)
        label = f"CATT {model}"
        if result is None:
            print(f"{label:<30} {'(not installed)':>28}")
        else:
            der, wer, elapsed = result
            print(f"{label:<30} {der:>7.1%} {wer:>7.1%} {elapsed:>9.3f}s")

    # Hybrid modes (optional)
    for model in ("encoder_only", "encoder_decoder"):
        result = evaluate_hybrid_mode(model)
        label = f"Hybrid (CATT {model} + dict)"
        if result is None:
            print(f"{label:<30} {'(not installed)':>28}")
        else:
            der, wer, elapsed = result
            print(f"{label:<30} {der:>7.1%} {wer:>7.1%} {elapsed:>9.3f}s")

    print()
    print("DER = Diacritization Error Rate (lower is better)")
    print("WER = Word Error Rate (lower is better)")


if __name__ == "__main__":
    main()
