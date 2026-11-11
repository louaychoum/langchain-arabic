# Speech normalization fixture results

Repository-authored synthetic MIT-licensed fixtures; no external corpus or independent
native-speaker review. Exact text matches include intentionally retained structures.
These are regression checks, not speech quality, ASR, or general accuracy scores.

| Fixture category | Exact matches |
|---|---:|
| dates | 7/7 |
| times | 6/6 |
| identifiers | 5/5 |
| amounts and quantities | 6/6 |
| protected structures | 5/5 |
| mixed text | 1/1 |
| Total | 30/30 |

Reproduce: `python benchmarks/evaluate_speech.py --check`.
See [scope and limits](../docs/speech-normalization.md) and [data license](data/README.md).
