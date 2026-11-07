# Contributing

Start with a small, reproducible bug report or one of the
[starter issues](docs/good-first-issues.md). Discuss public API changes before
implementing them. Keep fixes focused and preserve existing public names.

## Development

```bash
git clone https://github.com/louaychoum/langchain-arabic.git
cd langchain-arabic
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install -e ".[dev]"
```

Python 3.9 remains supported. CI checks 3.9–3.13; current LangChain 1.x runs on
3.10+, and 3.9 resolves the compatible 0.3 line. The declared minimum dependencies
are also tested on Python 3.10.

## Checks

```bash
ruff check src tests examples benchmarks
ruff format --check src tests examples benchmarks
mypy
pytest --cov=langchain_arabic --cov-branch --cov-report=term-missing
python benchmarks/evaluate_diacritics.py --check
python examples/langchain_chain.py
```

Use `ruff format src tests examples benchmarks` to format changed Python files.
Source type checking is strict. Public APIs ship annotations and `py.typed`.
Add a focused regression for a bug or new nontrivial rule, including the exact
Arabic input and expected output. Keep fixtures UTF-8. Explain linguistic choices,
including case endings, decimal reading, dialect and ambiguity.

Default tests and examples require no API keys, GPU, paid model calls or model
weights. CATT tests use mocks. Real inference must be reported separately with
platform, dependency/model versions and preserved base-text checks.

## Evaluation changes

Document data origin, license, expected forms and scope. Add tests for metric
changes. If supported output intentionally changes, regenerate results and explain
why; do not update expected values merely to make a failure disappear.

```bash
python benchmarks/evaluate_diacritics.py --output benchmarks/results.md
```

## Pull requests

Describe the failing input, expected behavior, fix and checks run. Use
`<type>: <description>` commit messages and keep unrelated cleanup separate.
Do not commit credentials, model weights or `dist/`. Do not claim external users,
accuracy, speed or a published integration without supporting evidence.

Follow the [code of conduct](CODE_OF_CONDUCT.md). Maintainer release instructions
are in [docs/releasing.md](docs/releasing.md).
