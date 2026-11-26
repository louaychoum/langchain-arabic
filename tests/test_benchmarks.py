"""Regression checks for the metrics, independent of package output fixtures."""

import json

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


def test_speech_report_counts_mismatches_as_failures(tmp_path, monkeypatch):
    from benchmarks import evaluate_speech

    data = tmp_path / "speech.json"
    data.write_text(
        '[{"category":"test","input":"5","expected":"wrong"},'
        '{"category":"test","input":"","expected":""}]',
        encoding="utf-8",
    )
    monkeypatch.setattr(evaluate_speech, "_DATA", data)
    report, passed = evaluate_speech.report()
    assert not passed
    assert "| test | 1/2 |" in report
    assert "Failed inputs:" in report


@pytest.mark.parametrize(
    ("expected", "transcript", "found"),
    [
        ("001234", "رقم الطلب 001234", True),
        ("001234", "رقم الطلب 1234", False),  # leading zeros are part of an identifier
        ("0501234567", "اتصل على ٠٥٠ ١٢٣ ٤٥٦٧", True),  # Eastern digits, grouped
        ("1250000", "1,250,000 زائر", True),
        ("1250.75", "المبلغ 1,250.75 ريال", True),
        ("12.50", "12.5 ريال", True),  # amounts compare by value
        ("9", "الساعة 09:30", True),
        ("2026", "عام 20260", False),
        ("نوفمبر|تشرين الثاني|11", "العاشر من تشرين الثاني", True),
        ("نوفمبر|11", "10/11/2026", True),
        ("%|بالمئة", "خمسة عشر بالمئة", True),
        ("ريال", "رِيَال", True),  # vowel marks ignored for words
        ("%|بالمئة", "15٪", True),  # Arabic percent sign
        ("9|التاسعة", "الساعة التاسعة والنصف", True),  # ordinal hour
        ("9|التاسعة", "الساعة التاسعة عشرة", False),  # 19, not 9
        ("الثالث والعشرين", "الثالث و العشرين", True),  # spacing after و
        ("10", "10 11 2026", True),  # spaced date parts stay separate numbers
        ("0501234567", "050-123-4567", True),  # hyphen-grouped phone
        ("8", "بين 3 و5 أيام", False),  # plain numbers are never summed
        ("101000", "101 ألف شخص", True),  # digits with a scale word
        ("101000", "100 ألف وألف", True),
        ("1250000", "مليون و250 ألف زائر", True),
        ("49.99", "49 دولار و99 سنت", True),  # main unit and subunit
        ("!ناقص|سالب", "2026 ناقص 11 ناقص 10", False),  # a date read as subtraction
        ("!ناقص|سالب", "العاشر من نوفمبر", True),
    ],
)
def test_roundtrip_entity_matching(expected, transcript, found):
    from benchmarks.tts_roundtrip import entity_found

    assert entity_found(expected, transcript) is found


def test_roundtrip_dry_run_writes_report_and_caches(tmp_path, capsys):
    from benchmarks import tts_roundtrip

    tts_roundtrip.main(["--dry-run", "--cases", "order-id,scale-101", "--out", str(tmp_path)])
    report = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "Dry run" in report and "| echo | echo | 2/2 | 0/2 | 2/2 | 0/2 |" in report
    results = json.loads((tmp_path / "results.json").read_text(encoding="utf-8"))
    assert {r["condition"] for r in results} == {"raw", "normalized"}
    assert len(list(tmp_path.rglob("*.wav"))) == 4 and len(list(tmp_path.rglob("*.echo.txt"))) == 4
    assert capsys.readouterr().out == report


def test_roundtrip_requires_explicit_paid_providers(tmp_path):
    from benchmarks import tts_roundtrip

    with pytest.raises(SystemExit):
        tts_roundtrip.main(["--out", str(tmp_path)])
    with pytest.raises(SystemExit):
        tts_roundtrip.main(["--tts", "nope", "--asr", "scribe", "--out", str(tmp_path)])
    with pytest.raises(SystemExit):  # echo scores would read as real results
        tts_roundtrip.main(["--tts", "echo", "--asr", "echo", "--out", str(tmp_path)])


def test_azure_asr_retries_errors_and_records_no_match(monkeypatch):
    from benchmarks import tts_roundtrip

    monkeypatch.setenv("AZURE_SPEECH_KEY", "key")
    monkeypatch.setenv("AZURE_SPEECH_ENDPOINT", "https://example.invalid")
    monkeypatch.setattr(tts_roundtrip, "_post", lambda *a: b'{"RecognitionStatus": "Error"}')
    with pytest.raises(RuntimeError):  # raising keeps the error out of the cache
        tts_roundtrip.asr_azure(b"")
    monkeypatch.setattr(tts_roundtrip, "_post", lambda *a: b'{"RecognitionStatus": "NoMatch"}')
    assert tts_roundtrip.asr_azure(b"") == "<NoMatch>"
