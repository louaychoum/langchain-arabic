# Changelog

## Unreleased (planned 0.6.0)

### Changed (breaking)

- `ArabicTextOutputParser` now reads Arabic numbers with the `ArabicSpeechNormalizer`
  rules when `number_contexts` is `None` (the default). Pass all five
  `number_contexts` (or `language="en"`) for the previous converter. Differences:
  - Identifiers and phones keep their digits and leading zeros: `هاتفي 0501234567`
    lost its zero and was read as a wrong cardinal. Bare leading-zero numbers and
    numbers after `order`/`id`/`otp`/`رمز` cues read digit by digit; `+` phones
    are read with `زائد`.
  - ISO dates, 24-hour times and `SAR`/`USD`/`AED`/`QAR` amounts are read.
  - Ranges, scores, fractions, slash dates (the parser cannot set `date_order`),
    comma decimals, URLs, emails, code spans and alphanumeric codes are kept as
    written instead of being converted piecewise (`فاز 3-1` stays as is).
- Phone cues read digit by digit only for numbers of three or more digits
  (`اتصل 10 مرات` is a count; `اتصل على 911` stays digit by digit).
- After a bare `رقم`, whole numbers under four digits without a leading zero read
  as quantities (`الغرفة رقم 15` → `خمسة عشر`), unless a phone verb precedes it
  (`اتصل على رقم 911`). Qualified cues (`رقم الطلب`) are unchanged.
- `convert_numbers_in_text` reads 7–15 digit numbers with a leading zero as phones.
- `backend="catt"` raises the missing-install `ImportError` when the parser is
  created instead of on the first parse.

### Fixed

- Work around a num2words 0.5.14 Arabic bug that repeated the scale word for
  groups 101–901 (`101000` read as `مائة و ألف ألف`, i.e. a wrong value); now
  `مائة ألف و ألف`, including millions, billions and trillions.
- Serialize num2words calls: its Arabic converter keeps per-call state on a shared
  instance, so concurrent `batch()`/threads could return another call's number.
- Read numbers with an attached conjunction (`و10`, `ف5`) or tatweel-marked
  preposition (`بـ50`, `وبـ3`); they were left as raw digits. A bare `ب`/`ك`/`ل`
  before digits stays a code (`فيتامين ب12`), and ranges after a clitic stay protected.
- Gulf substitutions now also cover the construct form `مئتا` (200, 1,200, 200,000).
- Keep speech normalization linear in input length. Crafted or number-dense inputs
  at the CLI's 64 KB limit took 4–60 s (cue rescans of the whole prefix, email/code
  and comma-number patterns restarting inside tokens, pairwise protected-span
  checks); they now take well under a second. A cue farther than about 60
  characters before its number is ignored, and an email or code that starts in the
  middle of an earlier matched token is no longer protected.

### Added

- `langchain_arabic.voice.pipecat_text_transform()` and `livekit_text_transform()`:
  TTS text hooks for Pipecat (`text_transforms`) and LiveKit Agents
  (`tts_text_transforms`) without depending on either framework. The LiveKit hook
  buffers streamed tokens per sentence and releases text well before the next
  number early; both fall back to the original text on errors. The context keeps
  the original text on Pipecat 1.5+ and on LiveKit with the default
  `use_tts_aligned_transcript=False`. Checked against `pipecat-ai` 1.17.0 and
  `livekit-agents` 1.8.13. See `docs/voice-agents.md`.
- `ArabicSpeechNormalizer(dialect="gulf")` applies Gulf number words to generated
  words only.
- Compiled dictionary patterns are cached; a 5,000-entry map dropped from about
  111 ms to 30 ms per parse.

## 0.5.0 — 2026-11-12

- Add a UTF-8 JSON stdin/stdout command for inspectable speech normalization,
  with input validation, protected terms, date policy and retained-span notices.
- Add a repository-distributed integration skill for coding assistants and a
  key-free LangGraph booking example that keeps original data separate from
  speech text and routes recognized ambiguous dates to clarification.
- Replace the awkward Arabic growth sentence in the README, notebook and chain
  example with a grammatical sentence and a separate percentage label.

## 0.4.0 — 2026-11-11

- Add opt-in `ArabicSpeechNormalizer` for validated Gregorian dates, explicit
  date ordering, 24-hour times, digit-preserving identifiers, decimal amounts
  and Arabic labels for SAR/USD/AED/QAR. Existing conversion APIs are unchanged.
- Add exact protected terms and conservative retention of structured tokens,
  plus source-offset decisions through `SpeechResult` and `SpeechChange`.
- Add a key-free LCEL speech example, sync/async integration tests, policy docs
  and 30 synthetic text fixtures with a reproducible offline report. These are
  text regression checks; no new audio-quality or dialect claim is made.

## 0.3.1 — 2026-11-07

### Fixed

- Apply dictionary entries to complete words/phrases, ignoring existing vowel
  marks and processing replacements only once. Plain dictionary keys now override
  neural output correctly. Existing partial marks are replaced rather than doubled.
- Keep mapping-file parsing on one line so an incomplete entry cannot consume the
  next entry. Invalid dictionary keys/values now raise a clear `ValueError`.
- Preserve numeric signs, fractional digits/trailing zeros, and adjacent punctuation.
  Accept Eastern Arabic digits and Arabic decimal/thousands separators.
- Read bare long integers as cardinal numbers. Digit-by-digit automatic phone
  reading now requires a leading `+` or phone cue; explicit `contexts={"phone"}`
  retains bare-phone reading. This change was approved during the release audit.
- Apply Gulf substitutions only to generated number words; existing prose is retained.
- Extract text content blocks in streams and run async stream processing in an
  executor. Both stream variants still buffer and emit one final result.
- Resolve changed parser dictionaries/dialects when processing instead of retaining
  a stale internal map.
- Match phone cues as whole words even after dictionary diacritization, and keep
  vowel marks from allowing a partial dictionary match inside a longer word.
- Load CATT lazily and once per cached model under concurrent initialization.
  Transfer vowel marks without deleting original non-Arabic content or formatting;
  raise if predictions change Arabic base letters or return an invalid batch count.

### Changed

- Require `langchain-core>=0.3.0,<2` and Pydantic 2 explicitly; the old 0.2 minimum
  installed but failed at runtime. Python 3.9 support remains through LangChain 0.3.
- Bound runtime dependency major/minor families and use the existing CATT 1.x API.
  The optional upstream GPU dependency still does not resolve on macOS.
- Replace unsupported README claims and sample neural scores with documented limits
  and generated smoke results. The external LangChain integration page remains a draft.

### Added

- Public package typing marker, strict source type checks, and focused edge-case,
  async/stream, concurrency, notebook, and benchmark-metric regression tests.
- Hand-authored MIT-licensed offline evaluation data and reproducible Markdown
  results. DER rejects changed base text, WER uses edit distance, and equivalent
  Unicode mark orders compare equally. No CATT accuracy is claimed.
- A key-free fake chat-model example and Colab-ready notebook.
- Contribution/conduct guidance, issue/PR templates, eight starter issue drafts,
  release instructions, CI for Python 3.9–3.13 and dependency floors, and a tag-driven
  PyPI trusted-publishing workflow.

## 0.3.0 — 2026-03-11

- Add `msa`/`gulf` dialect selection, Gulf number substitutions, dialect-keyed
  dictionary selection and MSA fallback (`cf8708e`).
- Export dialect helpers and document the dialect and targeted TTS workflows.

## 0.2.0 — 2026-03-06

- Initial published package: dictionary diacritics, Arabic/English contextual
  number conversion, LangChain output parser, optional CATT/hybrid backend,
  examples, tests and preliminary evaluation (`abd1bfb`).
- Add automatic mapping-format detection for arrows, tabs and colons, input
  warnings, and targeted TTS fixes (`3a71a8c`).
- Correct Pydantic private attribute handling and backend-cache initialization
  locking (`5edc2c3`).

Historical dates above are PyPI upload dates. Both git tags `v0.2.0` and `v0.3.0`
point to `5edc2c3`, whose source version is 0.2.0; the actual 0.3.0 PyPI wheel
matches `cf8708e`. These entries were reconstructed from that history and the
published metadata. Existing tags were not rewritten.
