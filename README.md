# langchain-arabic

[![PyPI](https://img.shields.io/pypi/v/langchain-arabic.svg)](https://pypi.org/project/langchain-arabic/)
[![PyPI Downloads](https://static.pepy.tech/personalized-badge/langchain-arabic?period=total&units=INTERNATIONAL_SYSTEM&left_color=GREY&right_color=RED&left_text=downloads)](https://pepy.tech/projects/langchain-arabic)
[![CI](https://github.com/louaychoum/langchain-arabic/actions/workflows/ci.yml/badge.svg)](https://github.com/louaychoum/langchain-arabic/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Add chosen Arabic diacritics and turn digits into spoken words, standalone or at the end of a LangChain chain.

```text
Raw LLM:   شركة تقنية حققت نسبة 95% نمو
Processed: شَرِكَة تِقْنِيَّة حققت نسبة خمسة و تسعون بالمائة نمو
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

llm = FakeListChatModel(responses=["شركة تقنية حققت نسبة 95% نمو"])
parser = ArabicTextOutputParser(
    diacritics_map={"شركة": "شَرِكَة", "تقنية": "تِقْنِيَّة"},
)
chain = llm | parser
print(chain.invoke("كيف كان النمو؟"))
# شَرِكَة تِقْنِيَّة حققت نسبة خمسة و تسعون بالمائة نمو
```

Replace the fake chat model with your provider's model to process actual LLM
output. Provider packages and credentials are configured separately.
See the [LangChain integration docs](https://docs.langchain.com/oss/python/integrations/providers/overview)
and [output-parser reference](https://reference.langchain.com/python/langchain_core/output_parsers/).
A package-specific page is [drafted in this repo](docs-pr/providers/langchain_arabic.mdx);
it is not currently published in the LangChain docs.

Run [`examples/langchain_chain.py`](examples/langchain_chain.py) for a prompt,
chat model, and parser chain with assertions for invocation and streaming.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/louaychoum/langchain-arabic/blob/v0.3.1/examples/quickstart.ipynb)

The [notebook](examples/quickstart.ipynb) installs the package, runs without keys,
and covers dialects, decimals, and async use. The Colab link uses the published
`v0.3.1` tag.

## Features and limits

| Component | What it does | Limit |
|---|---|---|
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
result = parser.parse("شركة تقنية حققت نسبة 95% نمو")
```

CATT models load on first use. Number conversion happens before CATT; dictionary
overrides happen afterward. The wrapper transfers vowel marks while preserving
original text, including English, digits, punctuation, and whitespace. If a model
changes Arabic base letters, it raises `ValueError` rather than dropping content.
`CATTBackend` also exposes `.tashkeel()` and `.tashkeel_batch()`.

Current `catt-tashkeel` uses ONNX Runtime and depends on `onnxruntime-gpu`.
The extra cannot currently resolve on macOS. Neural inference was not validated
for this release; its tests use mocks. Model choice is not an accuracy guarantee.

## Measured smoke results

Generated by [`benchmarks/evaluate_diacritics.py`](benchmarks/evaluate_diacritics.py)
on a small hand-authored set under the [MIT license](LICENSE):

| Check | Result |
|---|---:|
| Dictionary diacritization, 12 pairs / 4 dictionary entries | DER 63.39%; WER 80.77% |
| Number conversion, 24 hand-checked cases | 24/24 exact matches (100.00%) |

These are narrow smoke results. They do not estimate general Arabic accuracy.
The dictionary overlaps the evaluation set and leaves unknown words unchanged.
CATT was not run. See [results](benchmarks/results.md), [methodology](benchmarks/README.md),
and [data provenance](benchmarks/data/README.md).

```bash
python benchmarks/evaluate_diacritics.py --check
```

## Roadmap

- Evaluate on a versioned external corpus with verified redistribution terms.
- Add dialect rules with native-speaker examples and regression tests.
- Cover ordinal/currency grammar explicitly before advertising support.
- Validate real CATT inference on a supported platform and pursue CPU portability upstream.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md), the [starter issues](docs/good-first-issues.md),
and the [code of conduct](CODE_OF_CONDUCT.md). Release changes are listed in
[CHANGELOG.md](CHANGELOG.md). Contributions need runnable examples and tests for
observable behavior; benchmark claims need reproducible data.

## License

[MIT](LICENSE).
