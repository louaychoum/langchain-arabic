"""Text-to-speech hooks for voice-agent frameworks, without importing them.

Both helpers wrap :class:`ArabicSpeechNormalizer` and only rewrite text sent to
the TTS engine. The conversation keeps the original text on Pipecat >= 1.5, and on
LiveKit unless ``use_tts_aligned_transcript=True``; see ``docs/voice-agents.md``.

Pipecat (``pipecat-ai`` >= 1.5)::

    tts = SomeTTSService(..., text_transforms=[("*", pipecat_text_transform())])

LiveKit Agents (``livekit-agents`` >= 1.4.6)::

    AgentSession(tts_text_transforms=["filter_markdown", "filter_emoji", livekit_text_transform()])
"""

from __future__ import annotations

import logging
import re
from typing import AsyncIterable, AsyncIterator, Awaitable, Callable, Optional

from langchain_arabic.speech import ArabicSpeechNormalizer

logger = logging.getLogger(__name__)

# A sentence ends at . ! ? ؟ followed by whitespace, or at a newline that does not
# start a number: a cue at the end of one line still applies to a number on the next.
_SENTENCE_END = re.compile(r"[.!?؟]\s|\n(?=[^\d\s+−-])")
# Digits and code spans are read with their surroundings, so text near them is held.
_HOLD = re.compile(r"[\d`]")
# Cue matching looks back at most 64 characters after removing vowel marks
# (<= 128 raw characters); text farther before the first held character is safe to send.
_LOOKBACK = 128


def _safe_process(normalizer: ArabicSpeechNormalizer, text: str) -> str:
    try:
        return normalizer.process(text)
    except Exception as exc:  # speech must not stop because of a normalization bug
        # Customer text is deliberately not logged.
        logger.warning("Arabic speech normalization failed (%s); sending original text", exc)
        return text


def pipecat_text_transform(
    normalizer: Optional[ArabicSpeechNormalizer] = None,
) -> Callable[[str, str], Awaitable[str]]:
    """Return a Pipecat ``TextTransform`` for ``TTSService(text_transforms=...)``.

    Register it for ``"*"`` with the default sentence aggregation. Token-level
    text (``text_type == "token"``) passes through unchanged because a number can
    be split across tokens. On an internal error the original text is spoken;
    Pipecat would otherwise drop that sentence's audio.
    """
    active = normalizer or ArabicSpeechNormalizer()

    async def transform(text: str, text_type: str) -> str:
        if text_type == "token":  # also matches Pipecat's TextType.TOKEN (a str enum)
            return text
        return _safe_process(active, text)

    return transform


def livekit_text_transform(
    normalizer: Optional[ArabicSpeechNormalizer] = None,
) -> Callable[[AsyncIterable[str]], AsyncIterator[str]]:
    """Return a LiveKit ``tts_text_transforms`` callable over streamed LLM text.

    Text is normalized one sentence at a time so cues (``رقم الطلب``), currency
    codes and decimals see their neighbours. Text far enough before the first digit
    or backtick still buffered is released early, so ordinary prose is not delayed
    until the sentence ends.
    """
    active = normalizer or ArabicSpeechNormalizer()

    async def transform(text: AsyncIterable[str]) -> AsyncIterator[str]:
        buffer = ""
        async for chunk in text:
            searched = max(0, len(buffer) - 1)  # only the last character can start a new end
            buffer += chunk
            ends = [m.end() for m in _SENTENCE_END.finditer(buffer, searched)]
            if ends:
                yield _safe_process(active, buffer[: ends[-1]])
                buffer = buffer[ends[-1] :]
            hold = _HOLD.search(buffer)
            limit = (hold.start() if hold else len(buffer)) - _LOOKBACK
            cut = buffer.rfind(" ", 0, max(0, limit) + 1)
            if cut > 0:
                yield buffer[:cut]
                buffer = buffer[cut:]
        if buffer:
            yield _safe_process(active, buffer)

    return transform
