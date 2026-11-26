"""Framework-free contracts for the Pipecat and LiveKit TTS hooks."""

import asyncio
import random
from enum import Enum

import pytest

from langchain_arabic import ArabicSpeechNormalizer
from langchain_arabic.voice import livekit_text_transform, pipecat_text_transform

TEXTS = [
    "رقم الطلب 001234. المبلغ 12.50 SAR. الموعد 2026-11-10 الساعة 09:30.",
    "اتصل على 0501234567 أو اتصل على رقم 911؟ بين 5 و10 أيام بخصم 15%.\nSAR 250 فقط",
    "الغرفة رقم 15! رقم الحجز: ١٢٣ و v2.0 و https://example.com/12 و user1@example.com",
    "نص طويل بدون أرقام " * 12 + "ثم رقم الطلب 4821 والمبلغ USD -0.050.",
    "بدون أي رقم إطلاقا. " * 3,
    # cue and number first, then more than 128 characters of prose in the same sentence
    "رقم الطلب 4821 " + "وسيتم توصيله خلال ثلاثة أيام عمل " * 6 + "والمبلغ المستحق SAR 250.",
    # a cue at the end of a line applies to a number on the next line
    "للدعم اتصل على\n920012345 في أي وقت.\nتفاصيل طلبك:\nرقم الطلب:\n4821\nرقم الطلب:\n-5 جاهز.",
]


async def _stream(chunks):
    for chunk in chunks:
        yield chunk


def _collect(transform, chunks):
    async def run():
        return [piece async for piece in transform(_stream(chunks))]

    return asyncio.run(run())


def _chunkings(text, rng):
    yield [text]
    yield list(text)
    for _ in range(25):
        cuts = sorted(rng.sample(range(1, len(text)), k=min(len(text) - 1, rng.randint(1, 12))))
        yield [text[a:b] for a, b in zip([0, *cuts], [*cuts, len(text)])]


@pytest.mark.parametrize("text", TEXTS)
def test_livekit_stream_matches_whole_text_for_any_chunking(text):
    normalizer = ArabicSpeechNormalizer()
    expected = normalizer.process(text)
    transform = livekit_text_transform(normalizer)
    for chunks in _chunkings(text, random.Random(text)):
        assert "".join(_collect(transform, chunks)) == expected


def test_livekit_releases_digit_free_prose_before_the_sentence_ends():
    prose = "هذا نص عربي طويل لا يحتوي على أي أرقام " * 10
    pieces = _collect(livekit_text_transform(), [w + " " for w in prose.split()])
    assert len(pieces) > 1 and pieces[0] and not any(ch.isdigit() for ch in pieces[0])
    assert "".join(pieces) == prose


def test_livekit_holds_cues_until_the_number_arrives():
    chunks = ["رقم الطلب ", "12", "34", " جاهز."]
    assert "".join(_collect(livekit_text_transform(), chunks)) == (
        "رقم الطلب واحد اثنان ثلاثة أربعة جاهز."
    )


def test_pipecat_transform_normalizes_sentences_and_skips_tokens():
    class TextType(str, Enum):  # Pipecat's TextType is a str enum
        SENTENCE = "sentence"
        TOKEN = "token"

    transform = pipecat_text_transform(ArabicSpeechNormalizer(dialect="gulf"))
    assert asyncio.run(transform("الخصم 100%.", TextType.SENTENCE)) == "الخصم مية بالمية."
    assert asyncio.run(transform("1", TextType.TOKEN)) == "1"
    assert asyncio.run(transform("رقم الطلب 0012", "*")) == "رقم الطلب صفر صفر واحد اثنان"


def test_errors_fall_back_to_the_original_text(monkeypatch, caplog):
    normalizer = ArabicSpeechNormalizer()

    def broken(text):
        raise RuntimeError("boom")

    monkeypatch.setattr(normalizer, "process", broken)
    assert asyncio.run(pipecat_text_transform(normalizer)("سري 12", "sentence")) == "سري 12"
    assert "".join(_collect(livekit_text_transform(normalizer), ["سري", " 12."])) == "سري 12."
    assert "سري" not in caplog.text  # customer text stays out of logs
