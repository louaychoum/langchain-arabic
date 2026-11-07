"""Tests for langchain_arabic.numbers module."""

import pytest

from langchain_arabic.numbers import NumbersProcessor, convert_numbers_in_text


class TestConvertNumbersArabic:
    def test_plain_number(self):
        result = convert_numbers_in_text("العدد 5", language="ar")
        assert "5" not in result
        assert "خمسة" in result

    def test_percentage(self):
        result = convert_numbers_in_text("نسبة 95%", language="ar")
        assert "95" not in result
        assert "بالمائة" in result

    def test_percentage_with_space(self):
        result = convert_numbers_in_text("نسبة 95 %", language="ar")
        assert "95" not in result
        assert "بالمائة" in result

    def test_currency_riyal(self):
        result = convert_numbers_in_text("المبلغ 500 ريال", language="ar")
        assert "500" not in result
        assert "خمسمائة" in result
        assert "ريال" in result

    def test_currency_dirham(self):
        result = convert_numbers_in_text("السعر 100 درهم", language="ar")
        assert "100" not in result
        assert "درهم" in result

    def test_phone_digit_by_digit(self):
        result = convert_numbers_in_text("اتصل على 920000247", language="ar")
        assert "920000247" not in result
        # Phone numbers are digit-by-digit; verify individual digit words
        assert "تسعة" in result  # 9
        assert "صفر" in result  # 0

    def test_no_numbers(self):
        text = "مرحبا بالعالم"
        assert convert_numbers_in_text(text, language="ar") == text

    def test_empty_text(self):
        assert convert_numbers_in_text("", language="ar") == ""

    def test_year_context(self):
        result = convert_numbers_in_text("عام 2030", language="ar")
        assert "2030" not in result


class TestConvertNumbersEnglish:
    def test_plain_number(self):
        result = convert_numbers_in_text("the year 2030", language="en")
        assert "two thousand and thirty" in result

    def test_percentage(self):
        result = convert_numbers_in_text("about 95%", language="en")
        assert "percent" in result
        assert "95" not in result

    def test_currency(self):
        result = convert_numbers_in_text("costs 500 SAR", language="en")
        assert "500" not in result


class TestContextFiltering:
    def test_only_percentage(self):
        result = convert_numbers_in_text(
            "95% of 500 items",
            language="en",
            contexts={"percentage"},
        )
        assert "percent" in result
        # 500 should remain as digits (plain not enabled)
        assert "500" in result

    def test_only_plain(self):
        result = convert_numbers_in_text(
            "95% of 500 items",
            language="en",
            contexts={"plain"},
        )
        # percentage pattern not active, so 95% stays (but 95 digits get replaced)
        assert "500" not in result


class TestNumbersProcessor:
    def test_arabic_default(self):
        proc = NumbersProcessor(language="ar")
        result = proc.process("العدد 5")
        assert "خمسة" in result

    def test_english(self):
        proc = NumbersProcessor(language="en")
        result = proc.process("the number 42")
        assert "forty-two" in result

    def test_language_property(self):
        proc = NumbersProcessor(language="ar")
        assert proc.language == "ar"

    def test_with_contexts(self):
        proc = NumbersProcessor(language="ar", contexts={"percentage"})
        result = proc.process("نسبة 95%")
        assert "بالمائة" in result


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("٠ ١ ٢ ٣ ٤ ٥ ٦ ٧ ٨ ٩", "صفر واحد اثنان ثلاثة أربعة خمسة ستة سبعة ثمانية تسعة"),
        ("نسبة ٩٥٪", "نسبة خمسة و تسعون بالمائة"),
        ("١٢٫٥٠ ريال،", "اثنا عشر فاصلة خمسة صفر ريال،"),
        ("١٬٠٠٠", "ألف"),
        ("-12", "سالب اثنا عشر"),
        ("−12", "سالب اثنا عشر"),
        ("+12", "موجب اثنا عشر"),
        ("0.001", "صفر فاصلة صفر صفر واحد"),
        ("12.3456", "اثنا عشر فاصلة ثلاثة أربعة خمسة ستة"),
        ("1000000", "مليون"),
        ("1,000,000", "مليون"),
        ("5, next.", "خمسة, next."),
        ("5. Next", "خمسة. Next"),
        ("X5 v2.0", "X5 v2.0"),
        ("5، 6؛ 7؟", "خمسة، ستة؛ سبعة؟"),
    ],
)
def test_numeric_edges(text, expected):
    assert convert_numbers_in_text(text) == expected


def test_english_decimal_preserves_fraction_and_sign():
    assert convert_numbers_in_text("-0.050", "en") == "minus zero point zero five zero"


def test_phone_requires_cue_or_explicit_context():
    digits = "واحد صفر صفر صفر صفر صفر صفر"
    assert convert_numbers_in_text("phone: 1000000") == "phone: " + digits
    assert convert_numbers_in_text("+1000000") == digits
    assert convert_numbers_in_text("1000000", contexts={"phone"}) == digits
    assert convert_numbers_in_text("1000000", contexts=set()) == "1000000"


def test_gulf_changes_only_generated_numbers():
    assert convert_numbers_in_text("مائة و 100%", dialect="gulf") == "مائة و مية بالمية"


@pytest.mark.parametrize("options", [{"language": "xx"}, {"contexts": {"typo"}}, {"dialect": "xx"}])
def test_invalid_configuration(options):
    with pytest.raises(ValueError):
        convert_numbers_in_text("", **options)


def test_unsupported_magnitude_is_preserved():
    text = "9" * 100
    with pytest.warns(UserWarning, match="retained unchanged"):
        assert convert_numbers_in_text(text) == text
