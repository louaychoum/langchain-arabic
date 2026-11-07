# langchain-arabic

[![PyPI version](https://img.shields.io/pypi/v/langchain-arabic.svg)](https://pypi.org/project/langchain-arabic/)
[![PyPI Downloads](https://static.pepy.tech/personalized-badge/langchain-arabic?period=total&units=INTERNATIONAL_SYSTEM&left_color=BLACK&right_color=GREEN&left_text=downloads)](https://pepy.tech/projects/langchain-arabic)
[![Python versions](https://img.shields.io/pypi/pyversions/langchain-arabic.svg)](https://pypi.org/project/langchain-arabic/)
[![CI](https://github.com/louaychoum/langchain-arabic/actions/workflows/ci.yml/badge.svg)](https://github.com/louaychoum/langchain-arabic/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Arabic text post-processing for LLM outputs. Diacritics (tashkeel) restoration, number-to-word conversion, and native LangChain integration. Supports both dictionary-based and neural auto-diacritization via [CATT](https://github.com/abjadai/catt).

## Problem

LLMs produce Arabic text **without diacritics** (tashkeel) 60-70% of the time. This causes mispronunciation in text-to-speech (TTS) pipelines. Numbers in digit form (e.g. `2030`, `95%`) are also read incorrectly by TTS engines.

**langchain-arabic** fixes both issues as a post-processing step on LLM output.

## Installation

```bash
# Dictionary mode only (lightweight, no PyTorch)
pip install langchain-arabic

# With neural auto-diacritization (installs catt-tashkeel + PyTorch)
pip install langchain-arabic[catt]
```

## Quick Start

### Dictionary-Based Diacritics

Provide a mapping of plain Arabic words to their diacritized forms. The library applies longest-first replacement to avoid partial matches.

```python
from langchain_arabic import apply_diacritics, parse_diacritics_map

# From a dictionary
diacritics_map = {
    "تقنية": "تِقْنِيَة",
    "شركة": "شَرِكَة",
    "علم الحاسوب": "عِلْمُ الحَاسُوبِ",
}

text = "شركة تقنية في علم الحاسوب"
result = apply_diacritics(text, diacritics_map)
# -> "شَرِكَة تِقْنِيَة في عِلْمُ الحَاسُوبِ"
```

You can also parse mappings from a markdown file (useful for persona/prompt files):

```python
# Parse from markdown with "- WORD -> DIACRITIZED" format
diacritics_map = parse_diacritics_map("path/to/persona.md")

# Or from a markdown string
markdown = """
- تقنية → تِقْنِيَة
- شركة → شَرِكَة
"""
diacritics_map = parse_diacritics_map(markdown)
```

### Auto-Diacritization with CATT

For neural auto-diacritization (no manual dictionary needed), use the CATT backend. CATT is a state-of-the-art character-level transformer that outperforms GPT-4-turbo on Arabic diacritization benchmarks.

```bash
pip install langchain-arabic[catt]
```

```python
from langchain_arabic import ArabicTextOutputParser

parser = ArabicTextOutputParser(
    backend="catt",
    catt_model="encoder_only",   # faster; or "encoder_decoder" for higher accuracy
    convert_numbers=True,
)

result = parser.parse("شركة تقنية في علم الحاسوب")
# CATT auto-diacritizes the entire text
```

### Hybrid Mode: CATT + Dictionary Overrides

The most powerful setup: let CATT handle general text, then override domain-specific terms (proper nouns, brand names) with your dictionary:

```python
parser = ArabicTextOutputParser(
    backend="catt",
    catt_model="encoder_decoder",  # higher accuracy
    diacritics_map={
        "علم الحاسوب": "عِلْمُ الحَاسُوبِ",  # domain term override
    },
    convert_numbers=True,
)

chain = prompt | llm | parser
```

CATT runs first, then dictionary overrides are applied on top.

### Number-to-Word Conversion

Automatically detects and converts numbers based on context (percentages, currency, phone numbers, plain numbers).

```python
from langchain_arabic import convert_numbers_in_text

# Arabic
convert_numbers_in_text("نسبة 95%", language="ar")
# -> "نسبة خمسة و تسعون بالمائة"

convert_numbers_in_text("المبلغ 500 ريال", language="ar")
# -> "المبلغ خمسمائة ريال"

convert_numbers_in_text("اتصل على 920000247", language="ar")
# -> "اتصل على تسعة اثنان صفر صفر صفر صفر اثنان أربعة سبعة"

# English
convert_numbers_in_text("about 95%", language="en")
# -> "about ninety-five percent"
```

### Dialect Support

Use `dialect="gulf"` for Gulf Arabic (Khaleeji) number phrasing and dialect-keyed diacritics maps:

```python
from langchain_arabic import ArabicTextOutputParser

# Gulf dialect — numbers use Gulf phrasing
parser = ArabicTextOutputParser(dialect="gulf", convert_numbers=True)
parser.parse("النسبة 100%")
# -> "النسبة مية بالمية" (not "مائة بالمائة")

# Dialect-keyed diacritics maps — different vocalization per dialect
parser = ArabicTextOutputParser(
    dialect="gulf",
    diacritics_map={
        "msa": {"تقنية": "تِقْنِيَة"},
        "gulf": {"تقنية": "تَقْنِيَّة"},
    },
    convert_numbers=False,
)
```

Supported dialects: `"msa"` (Modern Standard Arabic, default), `"gulf"`.

### With LangChain

Use `ArabicTextOutputParser` as a drop-in replacement for `StrOutputParser` in any LCEL chain:

```python
from langchain_arabic import ArabicTextOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini")
prompt = ChatPromptTemplate.from_template("أجب بالعربية: {question}")

parser = ArabicTextOutputParser(
    diacritics_map={"تقنية": "تِقْنِيَة", "شركة": "شَرِكَة"},
    convert_numbers=True,
    language="ar",
)

chain = prompt | llm | parser
result = chain.invoke({"question": "ما هي التقنية؟"})
# Output has diacritics restored and numbers converted to words
```

> **Streaming note:** `ArabicTextOutputParser` buffers all chunks before
> processing because diacritics and number conversion require complete words.
> When using `chain.stream()`, the processed result is yielded as a single
> chunk once the LLM finishes generating.

## API Reference

### Diacritics

| Function / Class | Description |
|---|---|
| `parse_diacritics_map(source)` | Parse mappings from dict, markdown string, or file path |
| `apply_diacritics(text, diacritics_map)` | Apply longest-first replacement |
| `DiacriticsProcessor(source)` | Stateful wrapper with `.process(text)` method |

### Numbers

| Function / Class | Description |
|---|---|
| `convert_numbers_in_text(text, language, contexts)` | Convert digits to words in context |
| `NumbersProcessor(language, contexts)` | Stateful wrapper with `.process(text)` method |

**Supported contexts**: `"percentage"`, `"currency_ar"`, `"currency_en"`, `"phone"`, `"plain"`

### LangChain Integration

| Class | Description |
|---|---|
| `ArabicTextOutputParser` | LangChain `Runnable` combining diacritics + numbers |

**Parameters**:

| Parameter | Default | Description |
|---|---|---|
| `backend` | `"dictionary"` | `"dictionary"` or `"catt"` |
| `catt_model` | `"encoder_only"` | `"encoder_only"` (faster) or `"encoder_decoder"` (more accurate) |
| `dialect` | `"msa"` | `"msa"` or `"gulf"` — affects number phrasing and dialect-keyed map resolution |
| `diacritics_map` | `{}` | Plain -> diacritized mapping, or dialect-keyed `{"msa": {...}, "gulf": {...}}` |
| `convert_numbers` | `True` | Convert digit sequences to words |
| `language` | `"ar"` | `"ar"` or `"en"` |
| `number_contexts` | `None` (all) | Set of contexts to enable |

### Dialects

| Function / Constant | Description |
|---|---|
| `SUPPORTED_DIALECTS` | `frozenset` of supported dialect codes |
| `apply_dialect_number_overrides(text, dialect)` | Apply dialect-specific number word substitutions |
| `resolve_dialect_diacritics_map(map, dialect)` | Resolve flat or dialect-keyed map for a given dialect |

### CATT Backend

| Class | Description |
|---|---|
| `CATTBackend(model)` | Direct access to CATT auto-diacritization |

Requires `pip install langchain-arabic[catt]`.

## Examples

See the [`examples/`](https://github.com/louaychoum/langchain-arabic/tree/main/examples) directory for runnable scripts:

- [`quickstart.py`](https://github.com/louaychoum/langchain-arabic/blob/main/examples/quickstart.py) — Dictionary mode, CATT mode, hybrid mode, number conversion
- [`langchain_chain.py`](https://github.com/louaychoum/langchain-arabic/blob/main/examples/langchain_chain.py) — Full LangChain LCEL chain integration

## Benchmarks

See [`benchmarks/`](https://github.com/louaychoum/langchain-arabic/tree/main/benchmarks) for DER/WER evaluation of different diacritization modes.

## Development

```bash
git clone https://github.com/louaychoum/langchain-arabic.git
cd langchain-arabic
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest --cov=langchain_arabic
ruff check src/
```

## License

MIT
