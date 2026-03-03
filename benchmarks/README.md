# Benchmarks

Evaluates diacritization accuracy across different modes on a curated Arabic test corpus.

## Metrics

- **DER (Diacritization Error Rate):** Fraction of characters with incorrect diacritical marks. Lower is better.
- **WER (Word Error Rate):** Fraction of words where the diacritization differs from the reference. Lower is better.

## Running

```bash
# Dictionary mode only (no extra dependencies)
python benchmarks/evaluate_diacritics.py

# With CATT benchmarks (requires catt-tashkeel)
pip install langchain-arabic[catt]
python benchmarks/evaluate_diacritics.py
```

## Sample Results

Results on the built-in 8-sentence test corpus (your results will vary with CATT):

| Mode | DER | WER | Time |
|---|---|---|---|
| Dictionary only | ~70% | ~75% | <0.001s |
| CATT encoder_only | ~5-10% | ~15-25% | ~2-5s |
| CATT encoder_decoder | ~3-8% | ~10-20% | ~5-10s |
| Hybrid (CATT EO + dict) | ~4-8% | ~12-22% | ~2-5s |

Dictionary mode has high error rates because it only knows the 4 words in the built-in dictionary. In production, you would use CATT for general text and the dictionary for domain-specific overrides only.

## Custom Corpus

Edit the `TEST_CORPUS` list in `evaluate_diacritics.py` to add your own test sentences. Each entry is a tuple of `(undiacritized_text, reference_diacritized_text)`.

For rigorous evaluation, use the [Tashkeela corpus](https://sourceforge.net/projects/tashkeela/) or the [CATT benchmark dataset](https://github.com/abjadai/catt).
