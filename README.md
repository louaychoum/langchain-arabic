# langchain-arabic

[![PyPI](https://img.shields.io/pypi/v/langchain-arabic.svg)](https://pypi.org/project/langchain-arabic/)
[![PyPI Downloads](https://static.pepy.tech/personalized-badge/langchain-arabic?period=total&units=INTERNATIONAL_SYSTEM&left_color=GREY&right_color=RED&left_text=downloads)](https://pepy.tech/projects/langchain-arabic)
[![CI](https://github.com/louaychoum/langchain-arabic/actions/workflows/ci.yml/badge.svg)](https://github.com/louaychoum/langchain-arabic/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Prepare Arabic text for speech with inspectable readings for dates, times and identifiers, plus chosen diacritics and LangChain integration.

```text
Raw LLM:   هذه شركة تقنية. نسبة النمو: 95%.
Processed: هذه شَرِكَة تِقْنِيَّة. نسبة النمو: خمسة و تسعون بالمائة.
```

Only the words supplied in the dictionary receive diacritics in this example.
The package does not supply a general-purpose dictionary or infer grammatical
case, noun agreement, or full dialect morphology.

## Install

Python 3.9+; the current LangChain 1.x line requires Python 3.10+.
On Python 3.9, dependency resolution selects the compatible LangChain 0.3 line.

```bash
pip install langchain-arabic
```

Dictionary and number processing run locally without model downloads or API keys.

## Quick start with LCEL

This runnable example uses a fake **chat model** from `langchain-core`:

```python
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_arabic import ArabicTextOutputParser

llm = FakeListChatModel(responses=["هذه شركة تقنية. نسبة النمو: 95%."])
parser = ArabicTextOutputParser(
    diacritics_map={"شركة": "شَرِكَة", "تقنية": "تِقْنِيَّة"},
)
chain = llm | parser
print(chain.invoke("كيف كان النمو؟"))
# هذه شَرِكَة تِقْنِيَّة. نسبة النمو: خمسة و تسعون بالمائة.
```

Replace the fake chat model with your provider's model to process actual LLM
output. Provider packages and credentials are configured separately.
See the [LangChain integration docs](https://docs.langchain.com/oss/python/integrations/providers/overview)
and [output-parser reference](https://reference.langchain.com/python/langchain_core/output_parsers/).
A package-specific page is [drafted in this repo](docs-pr/providers/langchain_arabic.mdx);
it is not currently published in the LangChain docs.

Run [`examples/langchain_chain.py`](examples/langchain_chain.py) for a prompt,
chat model, and parser chain with assertions for invocation and streaming.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/louaychoum/langchain-arabic/blob/v0.4.0/examples/quickstart.ipynb)

The [notebook](examples/quickstart.ipynb) installs the package, runs without keys,
and covers speech preparation, dialect substitutions, decimals, and async use.
The Colab link uses the `v0.4.0` tag.

## Speech preparation in 0.4.0

Use `ArabicSpeechNormalizer` when a date, amount or identifier needs an explicit
reading. Keep the original text for display and storage; send the derived string
to your speech provider.

```python
from langchain_arabic import ArabicSpeechNormalizer

normalizer = ArabicSpeechNormalizer()
result = normalizer.explain("رقم الطلب 001234. المبلغ ١٢٫٥٠ SAR.")
print(result.text)
# رقم الطلب صفر صفر واحد اثنان ثلاثة أربعة. المبلغ اثنا عشر فاصلة خمسة صفر ريال سعودي.
for change in result.changes:
    print(change.kind, change.original, change.replacement)
```

The opt-in API recognizes supported Gregorian dates and 24-hour times, preserves
identifier digits and decimal precision, and records converted or deliberately
retained spans. Ambiguous dates stay unchanged unless you set `date_order`.
Use `protected_terms` for values your application must leave untouched.

See [policy, examples and limits](docs/speech-normalization.md) and the
[key-free LCEL example](examples/speech_chain.py). When combining it with
`ArabicTextOutputParser` for pronunciation fixes, set `convert_numbers=False`
on that parser so identifiers reach the normalizer intact. These rules do not
establish better audio or a Gulf accent; the [targeted audio check](docs/speech-audio-check.md)
found that raw ElevenLabs input already handled all three examples correctly.

## Features and limits

| Component | What it does | Limit |
|---|---|---|
| Speech normalizer | Supported dates/times, digit-preserving IDs, decimal amounts, protection and source-offset decisions | Opt-in; bounded rules; ambiguous formats retained; no incremental streaming |
| Dictionary diacritics | Longest whole-word/phrase match; replaces existing vowel marks | Requires your mappings; no context disambiguation |
| Mapping input | Dictionary, existing UTF-8 file path, or text with `→`, `->`, tabs, or `:` | First successful delimiter format wins |
| Numbers | Arabic/English cardinal words, signed decimals, percentages, currency labels | No noun/gender/case agreement or ordinal inference |
| Arabic digits | Western `0–9` and Eastern `٠–٩`; `.`/`٫` decimals and grouped `,`/`٬` thousands | Not a locale-aware date, version, or identifier parser |
| Phones | Digit-by-digit for `+` or preceding phone cues; explicit phone-only context | Automatic detection requires 7–15 ungrouped digits |
| Dialects | `msa` and four `gulf` number substitutions; supplied dialect dictionaries | No Egyptian/Levantine support or comprehensive Gulf grammar |
| LangChain | `invoke`, `batch`, async equivalents, sync/async streams | Streams buffer the full answer and yield one final string |
| Optional CATT | Neural diacritization; dictionary overrides; original content retained | Upstream GPU installation has platform restrictions; no accuracy claim here |

### Standalone usage

```python
from langchain_arabic import (
    DiacriticsProcessor, NumbersProcessor, apply_diacritics,
    convert_numbers_in_text, parse_diacritics_map,
)

mapping = parse_diacritics_map("تقنية -> تِقْنِيَّة")
assert apply_diacritics("تَقْنِيَة", mapping) == "تِقْنِيَّة"
assert DiacriticsProcessor(mapping).process("تقنية") == "تِقْنِيَّة"
assert convert_numbers_in_text("١٢٫٥٠") == "اثنا عشر فاصلة خمسة صفر"
assert NumbersProcessor(language="en").process("-0.5") == "minus zero point five"
```

Decimals are read digit by digit after `فاصلة`/`point` to retain precision and
trailing zeros. Leading minus/plus signs become spoken words. Values beyond
`num2words`' supported magnitude are retained with a warning.

Available number contexts are `percentage`, `currency_ar`, `currency_en`,
`phone`, and `plain`. `None` enables all; an empty set disables conversion.
For example, `number_contexts={"percentage"}` leaves unrelated numbers alone.
When `plain` is enabled, it can convert digits in otherwise disabled contexts
while retaining their symbols/labels.

Bare long integers now read as cardinal numbers. To read a known phone number
without a cue, use `convert_numbers_in_text("1234567", contexts={"phone"})`.
Automatic cues include `اتصل على`, `هاتف`, `جوال`, `phone`, `tel`, and `call`.

### Dialect and TTS fixes

```python
from langchain_arabic import ArabicTextOutputParser

parser = ArabicTextOutputParser.for_tts_fixes(
    {"عداد": "عَدَّاد"}, dialect="gulf",
)
assert parser.parse("عداد 100%") == "عَدَّاد مية بالمية"

parser = ArabicTextOutputParser(
    dialect="gulf", convert_numbers=False,
    diacritics_map={"msa": {"تقنية": "تِقْنِيَة"}, "gulf": {"تقنية": "تَقْنِيَّة"}},
)
```

Dialect-keyed maps select the requested dialect, then fall back to `msa`, then
an empty map. Flat maps work for either dialect. `SUPPORTED_DIALECTS`,
`resolve_dialect_diacritics_map`, and `apply_dialect_number_overrides` are also
exported. `language="en"` controls number words, not Arabic dictionary matching.

### Async and streaming

```python
# In an async function or a notebook cell:
result = await chain.ainvoke("كيف كان النمو؟")
chunks = [chunk async for chunk in chain.astream("كيف كان النمو؟")]
```

Words and numeric tokens may span chunks. Both `.stream()` and `.astream()`
buffer the complete answer and yield once; they do not stream processed tokens.
Async stream post-processing runs in an executor, keeping it off the event loop.
Text content blocks are extracted; image/reasoning blocks are not output as text.

### Optional neural backend

```bash
# On a platform supported by upstream onnxruntime-gpu (Linux/Windows):
pip install "langchain-arabic[catt]"
```

```python
parser = ArabicTextOutputParser(
    backend="catt", catt_model="encoder_only",  # or "encoder_decoder"
    diacritics_map={"تقنية": "تِقْنِيَّة"},
)
# First parse may download model weights.
result = parser.parse("هذه شركة تقنية. نسبة النمو: 95%.")
```

CATT models load on first use. Number conversion happens before CATT; dictionary
overrides happen afterward. The wrapper transfers vowel marks while preserving
original text, including English, digits, punctuation, and whitespace. If a model
changes Arabic base letters, it raises `ValueError` rather than dropping content.
`CATTBackend` also exposes `.tashkeel()` and `.tashkeel_batch()`.

Current `catt-tashkeel` uses ONNX Runtime and depends on `onnxruntime-gpu`.
The extra cannot currently resolve on macOS. Both models passed a post-release
smoke check using a manually installed CPU ONNX Runtime on macOS; this does not
validate the GPU extra. See [verification and reproduction](docs/release-verification.md).
Default tests use mocks. Model choice is not an accuracy guarantee.

## Measured smoke results

Generated by [`benchmarks/evaluate_diacritics.py`](benchmarks/evaluate_diacritics.py)
on a small hand-authored set under the [MIT license](LICENSE):

| Check | Result |
|---|---:|
| Dictionary diacritization, 12 pairs / 4 dictionary entries | DER 63.39%; WER 80.77% |
| Number conversion, 24 hand-checked cases | 24/24 exact matches (100.00%) |
| Speech normalization, 30 synthetic fixtures including retained controls | 30/30 exact matches |

These are narrow smoke results. They do not estimate general Arabic accuracy.
The dictionary overlaps the evaluation set and leaves unknown words unchanged.
See [offline results](benchmarks/results.md), [methodology](benchmarks/README.md),
and [data provenance](benchmarks/data/README.md). Separate
[post-release CATT smoke results](benchmarks/catt-results.md) were produced with
CPU ONNX Runtime; they carry the same small-set limitations.
Speech results come from [`benchmarks/evaluate_speech.py`](benchmarks/evaluate_speech.py);
see the [report and scope](benchmarks/speech-results.md). None of these text
scores measure audio quality.

```bash
python benchmarks/evaluate_diacritics.py --check
python benchmarks/evaluate_speech.py --check
```

## Roadmap

The next release's [JSON CLI, coding-assistant skill and LangGraph example](docs/agent-integrations.md)
are available in this checkout and are not included in PyPI 0.4.0. See the
[integration research](docs/agent-integrations-research.md) for design rationale.

- Evaluate on a versioned external corpus with verified redistribution terms.
- Add dialect rules with native-speaker examples and regression tests.
- Cover ordinal/currency grammar explicitly before advertising support.
- Validate the GPU extra on supported platforms and pursue CPU installation upstream.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md), the [starter issues](docs/good-first-issues.md),
and the [code of conduct](CODE_OF_CONDUCT.md). Release changes are listed in
[CHANGELOG.md](CHANGELOG.md). Contributions need runnable examples and tests for
observable behavior; benchmark claims need reproducible data.

## License

[MIT](LICENSE).
