"""Key-free chat-model chain. Run with: python examples/langchain_chain.py."""

from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.prompts import ChatPromptTemplate

from langchain_arabic import ArabicTextOutputParser

FAKE_RESPONSE = "شركة تقنية حققت نسبة 95% نمو"
DIACRITICS_MAP = {"شركة": "شَرِكَة", "تقنية": "تِقْنِيَّة"}


def demo_dictionary_chain() -> None:
    """Process a real LCEL chat-model response without keys or network calls."""
    llm = FakeListChatModel(responses=[FAKE_RESPONSE])
    prompt = ChatPromptTemplate.from_template("أجب بالعربية: {question}")
    parser = ArabicTextOutputParser(diacritics_map=DIACRITICS_MAP)
    chain = prompt | llm | parser
    result = chain.invoke({"question": "كيف كان النمو؟"})
    print(f"Raw:       {FAKE_RESPONSE}")
    print(f"Processed: {result}")
    assert result == "شَرِكَة تِقْنِيَّة حققت نسبة خمسة و تسعون بالمائة نمو"
    assert list(chain.stream({"question": "كيف كان النمو؟"})) == [result]


def demo_catt_chain() -> None:
    """Explicit optional neural demo; may download model weights."""
    llm = FakeListChatModel(responses=[FAKE_RESPONSE])
    parser = ArabicTextOutputParser(backend="catt", diacritics_map=DIACRITICS_MAP)
    print((llm | parser).invoke("كيف كان النمو؟"))


if __name__ == "__main__":
    import argparse

    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--catt", action="store_true", help="Also run CATT (may download weights)")
    args = cli.parse_args()
    demo_dictionary_chain()
    if args.catt:
        demo_catt_chain()
