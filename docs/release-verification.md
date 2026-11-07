# 0.3.1 post-release verification

Checked on 2026-11-07 against the published `langchain-arabic==0.3.1` package.
The published tag remains at `5ac5883`; integration into `master` did not change
the released package source.

## Hosted notebook

Opened the [tagged notebook in Google Colab](https://colab.research.google.com/github/louaychoum/langchain-arabic/blob/v0.3.1/examples/quickstart.ipynb)
and selected **Run all** on a new hosted runtime. All five code cells completed:

- `%pip` installed 0.3.1 from PyPI.
- The fake chat-model chain produced the expected dictionary/percentage output.
- Eastern digits, precise decimals, negatives and existing diacritics passed assertions.
- Gulf substitutions and selected number contexts passed assertions.
- Sync streaming, async invocation and async streaming passed assertions.

No API keys, model weights or Google Drive copy were needed for this notebook.

## Real CATT inference

Both `encoder_only` and `encoder_decoder` passed single-input, batch and parser
integration checks with actual upstream v2 ONNX weights. Six inputs per model
covered ordinary Arabic, already vocalized Arabic, mixed Arabic/English/digits,
punctuation, newlines, an empty string and an English-only string. Assertions
checked that removing vowel marks recovered the original content. Parser checks
also covered number conversion followed by a dictionary override.

This was a **manual CPU installation**, not validation of the declared GPU extra:

| Setting | Value |
|---|---|
| Platform | macOS 26.6.2, ARM64 |
| Python | 3.13.9 |
| langchain-arabic | 0.3.1, installed from PyPI |
| catt-tashkeel | 1.0.2 |
| onnxruntime | 1.31.1, `CPUExecutionProvider` |
| numpy / tqdm | 2.5.4 / 4.70.1 |
| Model downloads | [upstream v2 release](https://github.com/abjadai/catt/releases/tag/v2), `eo_model_onnx.zip` and `ed_model_onnx.zip` |

Reproduce the [measured smoke results](../benchmarks/catt-results.md) in a fresh
environment, from the repository root:

```bash
python3.13 -m venv /tmp/langchain-arabic-catt-check
source /tmp/langchain-arabic-catt-check/bin/activate
python -m pip install "langchain-arabic==0.3.1" "onnxruntime==1.31.1" "numpy==2.5.4" "tqdm==4.70.1"
python -m pip install --no-deps "catt-tashkeel==1.0.2"
python benchmarks/evaluate_diacritics.py --catt --output /tmp/catt-results.md
```

The single/batch preservation and combined-parser checks can be repeated in that
environment with:

```python
import re
from langchain_arabic import ArabicTextOutputParser, CATTBackend

def base(text):
    return re.sub(r"[\u064b-\u0652\u0670]", "", text)

texts = ["مرحبا بكم", "شركة تقنية", "السَّلَامُ عَلَيْكُمْ",
         "Hello، شركة ١٢٫٥٠!  \nتقنية؟", "", "English 123"]
for model in ("encoder_only", "encoder_decoder"):
    backend = CATTBackend(model=model)
    singles = [backend.tashkeel(text) for text in texts]
    batch = backend.tashkeel_batch(texts)
    assert len(batch) == len(texts)
    assert [base(text) for text in singles] == [base(text) for text in texts]
    assert [base(text) for text in batch] == [base(text) for text in texts]
    assert singles[-2:] == batch[-2:] == texts[-2:]
    assert any(re.search(r"[\u064b-\u0652]", text) for text in singles)
    parser = ArabicTextOutputParser(
        backend="catt", catt_model=model, diacritics_map={"تقنية": "تِقْنِيَّة"},
    )
    result = parser.parse("Hello، شركة تقنية ١٢٫٥٠ ريال!")
    assert "تِقْنِيَّة" in result
    assert base(result) == "Hello، شركة تقنية اثنا عشر فاصلة خمسة صفر ريال!"
```

`--no-deps` deliberately bypasses upstream's `onnxruntime-gpu` requirement and
uses the separately installed CPU runtime. This isolated experiment does not
make `pip install "langchain-arabic[catt]"` work on macOS; `pip check` will still
report upstream's missing GPU dependency. Do not use this setup as evidence that
the GPU installation or inference works.

The existing evaluation script produced the committed table on the same 12
hand-authored Arabic pairs and 24 numeric cases. No new corpus, general accuracy
or speed claim is made. Single and batch encoder-only predictions differed in
some vowel marks, so this check does not promise prediction parity across batch
sizes. Ordinary unit tests remain offline and mock neural inference.
