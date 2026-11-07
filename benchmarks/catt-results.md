# Smoke evaluation results

Hand-authored MIT-licensed examples: 12 diacritization pairs and 24 numeric cases.
These scores describe this small set only; they are not a general accuracy claim.

| Diacritization mode | DER | WER |
|---|---:|---:|
| Dictionary (4 entries) | 63.39% | 80.77% |
| CATT encoder_only | 6.25% | 26.92% |
| CATT encoder_decoder | 8.04% | 30.77% |

| Number conversion | Exact matches | Accuracy |
|---|---:|---:|
| Hand-checked cases | 24/24 | 100.00% |

See [methodology](README.md) and [data provenance/license](data/README.md).
