"""langchain-arabic LangChain LCEL chain integration example.

This example shows how to use ArabicTextOutputParser in a LangChain chain.
Replace the FakeLLM with a real LLM (e.g. ChatOpenAI) in production.

Run with:
    python examples/langchain_chain.py
"""

from langchain_core.language_models import FakeListLLM
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from langchain_arabic import ArabicTextOutputParser

# ── Simulated LLM output (undiacritized Arabic with numbers) ─────────────
FAKE_RESPONSE = "شركة تقنية تأسست عام 2030 وحققت نسبة 95% نمو في السوق السعودي"

# ── Dictionary overrides for domain-specific terms ───────────────────────
DIACRITICS_MAP = {
    "تقنية": "تِقْنِيَة",
    "شركة": "شَرِكَة",
}


def demo_dictionary_chain():
    """Dictionary mode — no CATT needed."""
    print("=" * 60)
    print("DICTIONARY MODE CHAIN")
    print("=" * 60)

    llm = FakeListLLM(responses=[FAKE_RESPONSE])
    prompt = ChatPromptTemplate.from_template("{input}")
    parser = ArabicTextOutputParser(
        diacritics_map=DIACRITICS_MAP,
        convert_numbers=True,
        language="ar",
    )

    chain = prompt | llm | StrOutputParser() | parser
    result = chain.invoke({"input": "أخبرني عن الشركة"})

    print(f"Raw LLM:   {FAKE_RESPONSE}")
    print(f"Processed: {result}")
    print()


def demo_catt_chain():
    """CATT hybrid mode chain (requires pip install langchain-arabic[catt])."""
    print("=" * 60)
    print("CATT HYBRID MODE CHAIN")
    print("=" * 60)

    try:
        llm = FakeListLLM(responses=[FAKE_RESPONSE])
        prompt = ChatPromptTemplate.from_template("{input}")
        parser = ArabicTextOutputParser(
            backend="catt",
            catt_model="encoder_only",
            diacritics_map=DIACRITICS_MAP,
            convert_numbers=True,
            language="ar",
        )

        chain = prompt | llm | StrOutputParser() | parser
        result = chain.invoke({"input": "أخبرني عن الشركة"})

        print(f"Raw LLM:   {FAKE_RESPONSE}")
        print(f"Processed: {result}")
    except ImportError:
        print("CATT not installed — skipping.")
        print("Install with: pip install langchain-arabic[catt]")
    print()


if __name__ == "__main__":
    demo_dictionary_chain()
    demo_catt_chain()
