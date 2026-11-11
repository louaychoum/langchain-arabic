"""Meaning-preserving structured readings, separate from legacy conversion."""

import asyncio
import json
from pathlib import Path

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

from langchain_arabic import ArabicSpeechNormalizer, ArabicTextOutputParser


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("", ""),
        ("نص عَرَبِيّ، API؛", "نص عَرَبِيّ، API؛"),
        ("الموعد 2026-11-10.", "الموعد العاشر من نوفمبر ألفان و ستة و عشرون."),
        ("٢٠٢٤/٠٢/٢٩", "التاسع والعشرون من فبراير ألفان و أربعة و عشرون"),
        ("03/04/2026", "03/04/2026"),
        ("2026-02-29", "2026-02-29"),
        ("2026/11-10", "2026/11-10"),
        ("2026-13-01", "2026-13-01"),
        ("0000-01-01", "0000-01-01"),
        ("2026-11-10-11", "2026-11-10-11"),
        ("2026-11-10T09:30:00Z", "2026-11-10T09:30:00Z"),
        ("الساعة ٠٩:٣٠.", "الساعة التاسعة و ثلاثون دقيقة."),
        ("00:00", "صفر تماما"),
        ("12:00", "الثانية عشرة تماما"),
        ("23:59:05", "الثالثة والعشرون و تسع و خمسون دقيقة و خمس ثوان"),
        ("01:01:02", "الأولى و دقيقة واحدة و ثانيتان"),
        ("02:02:01", "الثانية و دقيقتان و ثانية واحدة"),
        ("10:10", "العاشرة و عشر دقائق"),
        ("11:11", "الحادية عشرة و إحدى عشرة دقيقة"),
        ("12:12", "الثانية عشرة و اثنتا عشرة دقيقة"),
        ("13:13", "الثالثة عشرة و ثلاث عشرة دقيقة"),
        ("21:21", "الحادية والعشرون و إحدى و عشرون دقيقة"),
        ("24:00 09:60 09:10:60", "24:00 09:60 09:10:60"),
        ("12:30 PM 09:30am", "12:30 PM 09:30am"),
        ("09:3 1:2:3 09:30:00:01", "09:3 1:2:3 09:30:00:01"),
        ("رقم الطلب 001234.", "رقم الطلب صفر صفر واحد اثنان ثلاثة أربعة."),
        ("رقم الحجز: ١٢٣", "رقم الحجز: واحد اثنان ثلاثة"),
        ("رَقْم الطَّلَب 123", "رَقْم الطَّلَب واحد اثنان ثلاثة"),
        ("order id: 1234", "order id: واحد اثنان ثلاثة أربعة"),
        ("العدد 1234", "العدد ألف و مئتان و أربعة و ثلاثون"),
        ("هَاتِف: 1234567", "هَاتِف: واحد اثنان ثلاثة أربعة خمسة ستة سبعة"),
        ("+1234567", "زائد واحد اثنان ثلاثة أربعة خمسة ستة سبعة"),
        ("السعر ١٢٫٥٠ SAR،", "السعر اثنا عشر فاصلة خمسة صفر ريال سعودي،"),
        ("USD -0.050", "سالب صفر فاصلة صفر خمسة صفر دولار أمريكي"),
        ("المبلغ 12.50 ريال", "المبلغ اثنا عشر فاصلة خمسة صفر ريال"),
        ("SAR 12 USD", "SAR 12 USD"),
        ("نسبة ٩٥٪؛", "نسبة خمسة و تسعون بالمائة؛"),
        ("-12.50 +5", "سالب اثنا عشر فاصلة خمسة صفر موجب خمسة"),
        ("1,000,000", "مليون"),
        ("001 ٠٠٢", "صفر صفر واحد صفر صفر اثنان"),
        ("AB-123 منتج-123 v2.0 192.168.1.2", "AB-123 منتج-123 v2.0 192.168.1.2"),
        (
            "https://example.com/12?q=34 user1@example.com `95%`",
            "https://example.com/12?q=34 user1@example.com `95%`",
        ),
        ("3-5 1/2", "3-5 1/2"),
    ],
)
def test_readings(text, expected):
    assert ArabicSpeechNormalizer().process(text) == expected


@pytest.mark.parametrize(
    ("order", "expected"),
    [("dmy", "الثالث من أبريل"), ("mdy", "الرابع من مارس")],
)
def test_explicit_date_order(order, expected):
    normalizer = ArabicSpeechNormalizer(date_order=order)
    assert normalizer.process("03/04/2026") == expected + " ألفان و ستة و عشرون"
    assert normalizer.process("2026-11-10").startswith("العاشر من نوفمبر")
    assert normalizer.process("13/31/2026") == "13/31/2026"


def test_protected_literals_precede_all_normalization_and_are_copied():
    terms = ["001234", "سعر 12", "سعر 12.50"]
    normalizer = ArabicSpeechNormalizer(protected_terms=terms)
    terms.clear()
    result = normalizer.explain("رقم الطلب 001234، سعر 12.50؛ 95%")
    assert result.text == "رقم الطلب 001234، سعر 12.50؛ خمسة و تسعون بالمائة"
    assert [change.kind for change in result.changes] == ["protected", "protected", "number"]


def test_explanation_offsets_reconstruct_output_without_rescanning():
    text = "الموعد 2026-11-10؛ الساعة 09:30؛ رقم الطلب 001234؛ 03/04/2026"
    result = ArabicSpeechNormalizer().explain(text)
    reconstructed = text
    for change in reversed(result.changes):
        assert text[change.start : change.end] == change.original
        reconstructed = (
            reconstructed[: change.start] + change.replacement + reconstructed[change.end :]
        )
    assert reconstructed == result.text
    assert [change.kind for change in result.changes] == [
        "date",
        "time",
        "identifier",
        "retained_date",
    ]


def test_literal_protection_cannot_be_swallowed_by_an_earlier_numeric_match():
    normalizer = ArabicSpeechNormalizer(protected_terms=["234 to 56", "11"])
    assert normalizer.process("1234 to 567; 2026-11-10") == "1234 to 567; 2026-11-10"


def test_grouped_ids_decimal_ranges_and_short_explicit_phones():
    normalizer = ArabicSpeechNormalizer()
    for text in ["رقم الطلب 1,234", "ID: -123", "phone: 1,234", "12.50-13.50"]:
        assert normalizer.process(text) == text
    assert normalizer.process("هاتف: 112") == "هاتف: واحد واحد اثنان"
    assert normalizer.process("هاتف: +112") == "هاتف: زائد واحد واحد اثنان"


@pytest.mark.parametrize(
    "text",
    ["192.168.1.2:8080", "السعر 12,50 SAR", "SAR 12,50", "12٬50", "12,34,567", "12–15", "12—15"],
)
def test_unsupported_structures_are_not_partially_normalized(text):
    assert ArabicSpeechNormalizer().process(text) == text


def test_signed_large_amount_is_a_quantity_not_an_international_phone():
    assert ArabicSpeechNormalizer().process("USD +1000000") == "موجب مليون دولار أمريكي"


@pytest.mark.parametrize("terms", ["123", [""], [12]])
def test_invalid_protected_terms(terms):
    with pytest.raises(ValueError, match="protected_terms"):
        ArabicSpeechNormalizer(protected_terms=terms)


def test_invalid_date_order_and_text():
    with pytest.raises(ValueError, match="date_order"):
        ArabicSpeechNormalizer(date_order="guess")
    with pytest.raises(TypeError, match="text"):
        ArabicSpeechNormalizer().process(None)


def test_unsupported_precision_and_code_blocks_are_not_partially_converted():
    text = "09:30:00.500\n```python\nx = 95\ny = 12.5\n```\n95%"
    assert ArabicSpeechNormalizer().process(text) == text.replace("\n95%", "\nخمسة و تسعون بالمائة")
    assert ArabicSpeechNormalizer().process("```python\nx = 95") == "```python\nx = 95"


def test_unsupported_magnitude_is_retained_and_recorded():
    text = "9" * 100
    with pytest.warns(UserWarning, match="retained unchanged"):
        result = ArabicSpeechNormalizer().explain(text)
    assert result.text == text
    assert result.changes[0].kind == "retained_number"


def test_lcel_sync_async_batch_and_buffered_streams():
    normalizer = ArabicSpeechNormalizer()
    raw = "رقم الطلب 001234."
    expected = "رقم الطلب صفر صفر واحد اثنان ثلاثة أربعة."
    chain = (
        FakeListChatModel(responses=[raw]) | StrOutputParser() | RunnableLambda(normalizer.process)
    )
    assert chain.invoke("test") == expected
    assert chain.batch(["a", "b"]) == [expected, expected]
    assert list(chain.stream("test")) == [expected]

    async def check():
        assert await chain.ainvoke("test") == expected
        assert await chain.abatch(["a", "b"]) == [expected, expected]
        assert [chunk async for chunk in chain.astream("test")] == [expected]

    asyncio.run(check())


def test_legacy_parser_behavior_is_unchanged():
    assert (
        ArabicTextOutputParser().parse("رقم الطلب 001234.")
        == "رقم الطلب ألف و مئتان و أربعة و ثلاثون."
    )


def test_checked_in_speech_fixtures():
    corpus = Path(__file__).parents[1] / "benchmarks" / "data" / "speech.json"
    for case in json.loads(corpus.read_text(encoding="utf-8")):
        normalizer = ArabicSpeechNormalizer(**case.get("options", {}))
        assert normalizer.process(case["input"]) == case["expected"], case["input"]
