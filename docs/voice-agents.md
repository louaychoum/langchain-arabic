# Voice agents: Pipecat and LiveKit

`langchain_arabic.voice` gives ready-made text hooks for the two most used open
voice-agent frameworks. They only rewrite the text sent to the TTS engine, and
with the versions and settings below the conversation context keeps the original
LLM text. Neither framework is a dependency of this package; install the one you use.

Both hooks wrap `ArabicSpeechNormalizer`. Pass your own instance for
`date_order`, `protected_terms` or `dialect="gulf"`:

```python
from langchain_arabic import ArabicSpeechNormalizer

normalizer = ArabicSpeechNormalizer(dialect="gulf", protected_terms=["Model 12"])
```

## Pipecat (`pipecat-ai` 1.5 or newer)

```python
from langchain_arabic.voice import pipecat_text_transform

tts = ElevenLabsTTSService(  # any Pipecat TTS service
    ...,
    text_transforms=[("*", pipecat_text_transform(normalizer))],
)
```

- [`examples/pipecat_bot.py`](../examples/pipecat_bot.py) runs the hook in a real
  pipeline without API keys (install `examples/requirements-pipecat.txt`).
- Before 1.5, with TTS services that report word timestamps (Cartesia,
  ElevenLabs and similar), Pipecat saves the words the TTS spoke, so the context
  stores `أربعة ثمانية اثنان واحد` instead of `4821`. `text_transforms` exists
  from 0.0.96, but only 1.5+ maps rewritten words back to the original text.
- Keep Pipecat's default sentence aggregation. In token mode each LLM token is
  its own chunk, so the hook passes `text_type="token"` text through unchanged
  rather than reading `4` and `2` as two numbers.
- If normalization fails, the original text is spoken and a warning is logged
  without the text. Pipecat would otherwise drop that sentence's audio.
- Do not add tashkeel (for example with `ArabicTextOutputParser` and CATT) inside
  a Pipecat **TTS** transform when the TTS service reports word timestamps
  (Cartesia, ElevenLabs and similar). In Pipecat 1.17 the added marks shift its
  word-to-context alignment and can split words in the saved context. Number
  normalization does not have this problem: Pipecat 1.5+ maps changed word counts
  back to the original text. Checked against `pipecat-ai` 1.17.0.
- Do not combine it with Pipecat's `VoiceFormatter` or `expand_numbers`; those
  are English-only and also turn `٤٢` into `forty-two`.

## LiveKit Agents (`livekit-agents` 1.4.6 or newer)

```python
from livekit.agents import AgentSession
from langchain_arabic.voice import livekit_text_transform

session = AgentSession(
    # ...stt, llm, tts...
    tts_text_transforms=["filter_markdown", "filter_emoji", livekit_text_transform(normalizer)],
)
```

- Passing a list **replaces** LiveKit's defaults, so list `filter_markdown` and
  `filter_emoji` again, before the normalizer (code fences that span lines are
  not protected once the stream is split into sentences).
- LiveKit streams raw LLM token deltas. The hook normalizes one sentence at a
  time so cues such as `رقم الطلب`, currency codes and decimals see their
  neighbours. Text more than 128 characters before the next digit or backtick is
  released early; from there to the end of the sentence, text is held until the
  sentence (or a line not followed by a number) ends, which can add a little time
  before its audio starts.
- Leave `use_tts_aligned_transcript` off (the default). When it is on, LiveKit
  builds both the visible transcript and the saved chat context from what the
  TTS spoke, so the LLM sees `أربعة ثمانية اثنان واحد` instead of `4821` in later
  turns and can pass the wrong value to tools.
- Checked against `livekit-agents` 1.8.13, including the real transform chain
  with `filter_markdown` and `filter_emoji`.

## What this does and does not establish

The hooks apply the same bounded rules as the [speech normalizer](speech-normalization.md).
They do not prove better audio for a given provider: test your provider and
voice with your own inputs, and keep the original text for display and storage.
