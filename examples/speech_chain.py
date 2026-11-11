"""Key-free speech-input normalization. Run from an editable repository install."""

from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

from langchain_arabic import ArabicSpeechNormalizer


def main() -> None:
    raw = "الموعد 2026-11-10 الساعة 09:30. رقم الطلب 001234. المبلغ 12.50 SAR."
    normalizer = ArabicSpeechNormalizer()
    chain = (
        FakeListChatModel(responses=[raw]) | StrOutputParser() | RunnableLambda(normalizer.process)
    )
    expected = (
        "الموعد العاشر من نوفمبر ألفان و ستة و عشرون الساعة التاسعة و ثلاثون دقيقة. "
        "رقم الطلب صفر صفر واحد اثنان ثلاثة أربعة. المبلغ اثنا عشر فاصلة خمسة صفر ريال سعودي."
    )
    assert chain.invoke("أعطني تفاصيل الحجز") == expected
    assert list(chain.stream("أعطني تفاصيل الحجز")) == [expected]
    print(f"Raw: {raw}")
    print(f"Speech input: {expected}")
    for change in normalizer.explain(raw).changes:
        print(
            f"{change.kind} [{change.start}:{change.end}]: {change.original} → {change.replacement}"
        )


if __name__ == "__main__":
    main()
