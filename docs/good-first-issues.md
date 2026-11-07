# Starter issue drafts

These are proposals, not implemented features or GitHub issues. Each can be
posted separately. Confirm the desired linguistic reading before changing code;
new behavior should be opt-in if it changes established output.

## 1. Add a small Egyptian number substitution set

Start in `src/langchain_arabic/dialects.py`: add `egyptian` to the supported set
and a narrowly reviewed set for 100, 200 and percentages. Supply native-speaker
examples for standalone and compound forms, add tests in `tests/test_dialects.py`,
and update the README's dialect scope. Acceptance: existing MSA/Gulf output stays
unchanged; every advertised Egyptian rule has an exact expected-output test.

## 2. Add a small Levantine number substitution set

Use the same dialect extension points. Limit the first change to an explicitly
named regional convention for hundreds/percentages; do not claim all Levantine
Arabic. Provide examples and linguistic review before choosing forms. Acceptance:
map resolution/fallback works and both standalone and compound forms are tested.

## 3. Recognize inflected Arabic currency labels

`numbers.py` currently recognizes only `ريال`, `درهم`, and `دينار`. Add tests and
recognition for a small agreed set of plural/dual labels without rewriting their
grammar. Exercise `contexts={"currency_ar"}` so the test proves recognition rather
than plain-number fallback. Acceptance: original labels and punctuation survive,
and word fragments are not treated as currencies.

## 4. Support a bounded grouped-phone format

Phone detection currently requires 7–15 contiguous digits. Propose one format,
such as `phone: +966 50 123 4567`, with an explicit phone cue. Change the shared
numeric tokenizer, not just the LangChain parser. Acceptance: digits read in order,
ordinary lists/dates/negative numbers do not become phones, and selective contexts
still work. Agree handling of the leading plus sign first.

## 5. Design an explicit ordinal-number context

There is no ordinal support. Investigate `num2words` Arabic/English ordinal output,
record supported bounds, and propose an opt-in `ordinal` context with five exact
examples and a documented masculine/feminine policy. Keep this issue to the
contract and a small tested implementation after review. Acceptance: defaults
and existing context names retain their behavior; unsupported forms are explicit.

## 6. Explain Arabic gender and grammatical agreement limits

Add a short `docs/number-grammar.md` using reproducible examples for 1, 2, 3–10,
11–19 and currency labels. Show which forms the current package emits and what
surrounding nouns/case would require. Do not change output in this documentation
issue. Acceptance: examples run, sources are cited, and the README links the note.

## 7. Add a licensed external diacritization subset

The current benchmark is synthetic. Identify one corpus with explicit compatible
redistribution terms, record its source revision/license, and propose a small fixed
subset plus normalization/case-ending policy. Extend the existing script rather
than adding another metric implementation. Acceptance: offline reproduction is
possible, provenance is checked in, and changed base text cannot receive a score.
Do not publish CATT results without actually running it.

## 8. Add an opt-in real CATT integration check

Unit tests mock CATT. On a supported Linux/Windows environment, add an opt-in check
that records dependency/model versions and proves real inference preserves mixed
English/Arabic, digits, whitespace and RTL punctuation through the wrapper.
Keep downloads out of default pytest. Acceptance: the command and resource needs
are documented, both model variants are considered, and failures retain original
content rather than silently accepting changed base letters.
