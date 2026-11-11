# Changelog

## Unreleased

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
