# Arabic speech input milestone

This is an **unreleased**, opt-in API. Install this checkout with
`python -m pip install -e ".[dev]"`; the published 0.3.1 package does not include it.
It leaves the existing `ArabicTextOutputParser`, `NumbersProcessor` and
`convert_numbers_in_text` behavior unchanged. No additional dependency or model
is required.

## Purpose

Give developers explicit, inspectable Arabic readings for structured text before
TTS. This is useful when a date, identifier or amount needs a specific reading.
The prior ElevenLabs pilot did not establish a general benefit from preprocessing
ordinary text. Passing these text checks does not establish better audio.

```python
from langchain_arabic import ArabicSpeechNormalizer

normalizer = ArabicSpeechNormalizer()
raw = "الموعد 2026-11-10 الساعة 09:30. رقم الطلب 001234. المبلغ 12.50 SAR."
result = normalizer.explain(raw)
print(result.text)
# الموعد العاشر من نوفمبر ألفان و ستة و عشرون الساعة التاسعة و ثلاثون دقيقة.
# رقم الطلب صفر صفر واحد اثنان ثلاثة أربعة. المبلغ اثنا عشر فاصلة خمسة صفر ريال سعودي.

for change in result.changes:
    assert raw[change.start:change.end] == change.original
    print(change.kind, change.original, change.replacement)
```

`process(text)` returns only a string. `explain(text)` returns a frozen
`SpeechResult` containing that same string and a tuple of frozen `SpeechChange`
records. Each record has `start`, `end` (Python string offsets into the original
input), `original`, `replacement`, and `kind`. Decisions also include matched
spans that are deliberately retained: `protected`, `retained_date`,
`retained_time`, and `retained_number`. Ordinary prose is not logged. These are
rule categories, not model confidence scores or a guarantee of semantic safety.
Generated words are never matched again in the same pass.

## Supported policy

- **Dates:** Gregorian year/month/day with a four-digit year first, `-` or `/`
  separators, and Western or Eastern Arabic decimal digits. Calendar validation
  rejects impossible dates. Day/month/year and month/day/year are retained by
  default, even if one interpretation looks likely. Set `date_order="dmy"` or
  `"mdy"` explicitly to enable them. Two-digit years, mixed separators, Hijri
  dates and ISO timestamps with time zones are not interpreted.
- **Times:** `HH:MM` and `HH:MM:SS` are interpreted as 24-hour clock times, with
  hour 0–23 and minute/second 0–59. Hours remain in 24-hour form: `23:00` becomes
  `الثالثة والعشرون تماما`. Zero hours read `صفر`. AM/PM inputs, invalid times,
  durations, ratios and subsecond timestamps are not interpreted. A colon form
  such as `01:30` is ambiguous between a clock time and a duration: protect it
  explicitly when the application means a duration.
- **Identifiers:** bare integers beginning with zero, or immediately following
  the configured built-in cues (`رقم`, `رقم الطلب`, `رقم الحجز`, `رقم الحساب`,
  `رمز`, `رمز التحقق`, `معرف`, `order`, `order id`, `booking`, `booking id`,
  `otp`, `id`), read digit by digit. Cue matching ignores Arabic vowel marks and
  English case. A bare `1234` without a cue still reads as a quantity; arbitrary
  identifiers cannot be inferred reliably. Grouped values directly after an ID
  cue, hyphenated numeric chains and mixed letter/digit codes are retained.
  Phones directly after the existing phone cues read digit by digit, including
  short emergency numbers; a leading `+` without a cue requires 7–15 digits.
  Multi-part identifiers/phones separated by spaces need explicit protection.
- **Amounts:** decimal fractional digits retain their precision and trailing
  zeros. `SAR`, `USD`, `AED`, `QAR` become Arabic unit labels. Supported Arabic
  currency labels, percentages, signed numbers and grouped thousands reuse the
  existing converter. Currency codes must be uppercase. Amounts use decimal
  readings, **not** major/minor currency decomposition or rounding; no currency
  agreement, noun gender or syntactic case is inferred. Conflicting prefix and
  suffix currency labels and unsupported comma-decimal/grouping forms (such as
  `12,50 SAR`) are retained. Commas denote thousands only in valid three-digit
  groups; decimal separators are `.` or `٫`.
- **Protection:** URLs beginning with `http://`, `https://` or `www.`, email
  addresses, inline/fenced Markdown code, mixed alphanumeric tokens, numeric
  ranges/fractions, dotted IPv4 addresses with optional ports and unrecognized
  numeric chains bypass conversion. This is a
  bounded recognizer, not a parser for every address, file format or identifier.
  Use `protected_terms` for application-specific values.

Literal protection has highest priority, uses exact case-sensitive substring
matching, and chooses the longest match at the same position. For example:

```python
normalizer = ArabicSpeechNormalizer(
    date_order="dmy",
    protected_terms=["001234", "09:30", "Model 12"],
)
assert normalizer.process("رقم الطلب 001234") == "رقم الطلب 001234"
```

If a protected substring overlaps a larger numeric span, that whole numeric span
is retained. This prevents a date or amount match from swallowing the protection.

Retaining text prevents this package from changing it. It **does not** force the
TTS provider to pronounce the retained text correctly. Speech strings are for
TTS input; keep original application data separately for display and storage.

## LangChain and pronunciation overrides

Run the normalizer after string extraction, and disable the old parser's number
conversion if using it for pronunciation fixes. Otherwise it could destroy an
identifier before the new normalizer sees it.

```python
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.runnables import RunnableLambda
from langchain_arabic import ArabicSpeechNormalizer, ArabicTextOutputParser

normalizer = ArabicSpeechNormalizer()
chain = (
    FakeListChatModel(responses=["رقم الطلب 001234. هذا عداد جديد."])
    | ArabicTextOutputParser(
        convert_numbers=False, diacritics_map={"عداد": "عَدَّاد"}
    )
    | RunnableLambda(normalizer.process)
)
assert chain.invoke("test") == "رقم الطلب صفر صفر واحد اثنان ثلاثة أربعة. هذا عَدَّاد جديد."
```

If no pronunciation fixes are needed, use `StrOutputParser()` instead of
`ArabicTextOutputParser`. `RunnableLambda` accepts the complete string, so sync
and async streams buffer before yielding one final normalized string. There is
no incremental low-latency streaming in this milestone. Run the key-free
[example](../examples/speech_chain.py) for the date/amount/identifier path.

Protection applies within this normalizer only. An earlier pronunciation parser
can still alter a protected term; keep those mappings restricted to the words
you intend to change. CATT is an optional separate stage and was not evaluated
as part of this milestone.

## Verification

```bash
pytest tests/test_speech.py
python benchmarks/evaluate_speech.py --check
python examples/speech_chain.py
```

The [checked-in results](../benchmarks/speech-results.md) use synthetic,
repository-authored expected strings under MIT. They include retained controls
and are regression evidence only. No external corpus, independent native-speaker
review, dialect model, accent control, or audio-quality metric is included in
these fixtures. A separate [targeted ElevenLabs check](speech-audio-check.md)
records three synthetic audio cases and their diagnostic transcripts. Evaluate
the final provider audio on your own intended inputs.

### Local verification, November 11, 2026

- Full suite: 236 passed on each of Python 3.9.6, 3.10.19, 3.11.14, 3.12.12 and
  3.13.9, plus the Python 3.10 dependency-floor environment (`langchain-core`
  0.3.0, Pydantic 2.7.4, `num2words` 0.5.14). The Python 3.13 environment used
  `langchain-core` 1.7.6 and Pydantic 2.14.3. These were existing local environments,
  not fresh installations or hosted Actions runs.
- Python 3.13 line/branch coverage: 99% package aggregate; the new speech module
  has no uncovered statements or branches. Coverage does not establish linguistic
  correctness beyond the assertions.
- Ruff 0.17.4 lint/format, strict mypy, both offline evaluation checks, both
  key-free examples and `git diff --check` passed.
- Source distribution and wheel built; `twine check` passed. Importing directly
  from the built wheel exercised the identifier path, and archive checks verified
  the typing marker, new fixture/docs and exclusion of private `tasks/`/`work/`.
  The build deliberately retains source version 0.3.1 and is a local packaging
  smoke test only. It must not be uploaded as a replacement for published 0.3.1;
  a new version and fresh build are still required for a release.
- The targeted audio check generated four new clips and reused two raw clips.
  No independent native-speaker evaluation, CATT inference, hosted CI, tag or
  PyPI publication is included in this local verification.
