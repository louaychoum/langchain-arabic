"""langchain-arabic quickstart — demonstrates all three diacritization modes.

Run with:
    python examples/quickstart.py
"""

from langchain_arabic import ArabicTextOutputParser, apply_diacritics, convert_numbers_in_text

SAMPLE_TEXT = "شركة تقنية في علم الحاسوب تأسست عام 2030 وحققت نسبة 95% نمو"

DIACRITICS_MAP = {
    "تقنية": "تِقْنِيَة",
    "شركة": "شَرِكَة",
    "علم الحاسوب": "عِلْمُ الحَاسُوبِ",
}


def demo_dictionary_mode():
    """Dictionary-based diacritics (no PyTorch needed)."""
    print("=" * 60)
    print("1. DICTIONARY MODE")
    print("=" * 60)
    print(f"Input:  {SAMPLE_TEXT}")

    result = apply_diacritics(SAMPLE_TEXT, DIACRITICS_MAP)
    print(f"Output: {result}")
    print()


def demo_numbers():
    """Number-to-word conversion with context detection."""
    print("=" * 60)
    print("2. NUMBER CONVERSION")
    print("=" * 60)

    examples = [
        ("نسبة 95%", "ar"),
        ("المبلغ 500 ريال", "ar"),
        ("اتصل على 920000247", "ar"),
        ("about 95%", "en"),
    ]
    for text, lang in examples:
        result = convert_numbers_in_text(text, language=lang)
        print(f"  [{lang}] {text}")
        print(f"     -> {result}")
    print()


def demo_catt_mode():
    """Neural auto-diacritization with CATT (requires pip install langchain-arabic[catt])."""
    print("=" * 60)
    print("3. CATT AUTO-DIACRITIZATION")
    print("=" * 60)

    try:
        parser = ArabicTextOutputParser(
            backend="catt",
            catt_model="encoder_only",
            convert_numbers=True,
            language="ar",
        )
        result = parser.parse(SAMPLE_TEXT)
        print(f"Input:  {SAMPLE_TEXT}")
        print(f"Output: {result}")
    except ImportError as exc:
        print(f"CATT not installed: {exc}")
        print("Install with: pip install langchain-arabic[catt]")
    print()


def demo_hybrid_mode():
    """CATT + dictionary overrides for proper nouns."""
    print("=" * 60)
    print("4. HYBRID MODE (CATT + dictionary overrides)")
    print("=" * 60)

    try:
        parser = ArabicTextOutputParser(
            backend="catt",
            catt_model="encoder_only",
            diacritics_map=DIACRITICS_MAP,  # overrides applied AFTER CATT
            convert_numbers=True,
            language="ar",
        )
        result = parser.parse(SAMPLE_TEXT)
        print(f"Input:  {SAMPLE_TEXT}")
        print(f"Output: {result}")
        print("(Dictionary overrides applied on top of CATT output)")
    except ImportError:
        print("CATT not installed — skipping hybrid demo.")
        print("Install with: pip install langchain-arabic[catt]")
    print()


def demo_output_parser():
    """ArabicTextOutputParser in dictionary mode (no LangChain LLM needed)."""
    print("=" * 60)
    print("5. OUTPUT PARSER (dictionary mode, standalone)")
    print("=" * 60)

    parser = ArabicTextOutputParser(
        diacritics_map=DIACRITICS_MAP,
        convert_numbers=True,
        language="ar",
    )
    result = parser.invoke(SAMPLE_TEXT)
    print(f"Input:  {SAMPLE_TEXT}")
    print(f"Output: {result}")
    print()


if __name__ == "__main__":
    demo_dictionary_mode()
    demo_numbers()
    demo_catt_mode()
    demo_hybrid_mode()
    demo_output_parser()
