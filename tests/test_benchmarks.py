"""Regression checks for the metrics, independent of package output fixtures."""

import pytest

from benchmarks.evaluate_diacritics import compute_der, compute_wer


@pytest.mark.parametrize("prediction", ["", "تُ", "بِ تَ"])
def test_der_rejects_changed_or_missing_base_text(prediction):
    with pytest.raises(ValueError, match="base text"):
        compute_der(prediction, "بِ")


def test_der_counts_arabic_letters_and_normalizes_marks():
    assert compute_der("بَ تِ", "بِ تِ") == 0.5
    assert compute_der("بَّ", "بَّ") == 0
    assert compute_der("English! بِ", "English! بِ") == 0
    assert compute_der("", "") == 0


def test_wer_uses_edit_distance_and_reference_length():
    assert compute_wer("أ ب ج", "ب ج") == 0.5
    assert compute_wer("أ", "أ ب") == 0.5
    assert compute_wer("بَّ", "بَّ") == 0
    assert compute_wer("", "") == 0
    assert compute_wer("أ ب ج", "أ") == 2
