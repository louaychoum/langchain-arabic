# Evaluation data

These 12 diacritization sentence pairs and 24 number cases were authored and
hand-checked for langchain-arabic v0.3.1. They are distributed under the
[repository's MIT license](../../LICENSE). They were not copied from an external
corpus. No licensed public dataset was available in the checkout for offline
reproduction, so this release uses the small hand-checked fallback.

These are synthetic smoke examples, not a representative sample of Arabic or an
independent assessment of model quality. The numeric cases deliberately exercise
supported formatting rules; they do not measure agreement with an external
corpus. The dictionary contains four entries, overlaps the sentence pairs, and
is printed in the evaluation script. The references include case endings.

A later external-corpus evaluation must record the exact source revision,
license, preprocessing, split and dictionary provenance before publishing scores.
