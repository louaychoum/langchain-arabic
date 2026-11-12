# Coding assistants and LangGraph

The JSON CLI is available from **0.5.0**. Install it with
`python -m pip install "langchain-arabic>=0.5.0,<0.6"`.
The integration skill and LangGraph example are distributed in the repository;
use the `v0.5.0` tag for a matching checkout. The `ArabicSpeechNormalizer` Python
API remains available from 0.4.0.

## JSON command

```bash
python -m langchain_arabic normalize --json < request.json
```

`request.json` is one UTF-8 JSON object:

```json
{"text":"رقم الطلب 001234. الموعد 03/04/2026.","date_order":null,"protected_terms":[]}
```

`text` is required. `date_order` is optional (`null`, `"dmy"` or `"mdy"`), and
`protected_terms` is an optional array of nonempty literal strings. Unknown
fields and malformed input fail with exit code 2, a diagnostic on stderr and
no JSON on stdout. Input is limited to 65,536 bytes, including JSON syntax.
This is one complete request per process, not JSON Lines or token streaming.

On success, exit code 0 returns one UTF-8 JSON object with:

| Field | Meaning |
|---|---|
| `display_text` | Original input, unchanged |
| `speech_text` | Derived speech reading |
| `changes` | Existing source-offset records: `start`, `end`, `original`, `replacement`, `kind` |
| `warnings` | Normalizer warnings and notices for retained date/time/number spans |

Offsets count Python Unicode characters, not UTF-8 bytes or JavaScript UTF-16
code units. A `protected` decision may represent literal protection or a
recognized unsupported structure; warning absence does not prove correctness.
See the [bounded policy](speech-normalization.md). No network, file lookup or
model invocation is performed by the command. It returns input text, so callers
should avoid logging sensitive responses.

Pass untrusted text as JSON stdin, not interpolated shell syntax:

```python
import json
import subprocess
import sys

completed = subprocess.run(
    [sys.executable, "-m", "langchain_arabic", "normalize", "--json"],
    input=json.dumps({"text": "رقم الطلب 001234"}).encode("utf-8"),
    capture_output=True, check=True,
)
result = json.loads(completed.stdout)
assert result["speech_text"] == "رقم الطلب صفر صفر واحد اثنان ثلاثة أربعة"
```

## One skill for Codex and Claude Code

The source is [skills/arabic-speech/SKILL.md](../skills/arabic-speech/SKILL.md).
Copy that folder into the **application** repository where the assistant is
integrating TTS:

- Codex: `.agents/skills/arabic-speech/SKILL.md`.
- Claude Code: `.claude/skills/arabic-speech/SKILL.md`.

Restart or refresh the host's skill discovery, then explicitly ask it to use
`arabic-speech` to integrate the application's Arabic TTS path. The skill keeps
the application's provider, preserves source/display text, checks date policy
and adds a focused regression. It grants no additional permissions and does not
automatically install dependencies or call paid services.

See [Codex skills](https://learn.chatgpt.com/docs/build-skills) and
[Claude Code skills](https://code.claude.com/docs/en/skills) for host-specific
discovery and invocation. Format validation does not prove automatic activation;
fresh-host adoption or productivity improvements have not been measured.

## LangGraph booking example

Requires Python 3.10+ and an optional dependency, kept out of the base package:

```bash
python -m pip install -r examples/requirements-langgraph.txt
python examples/langgraph_speech.py
pytest tests/test_langgraph_example.py
```

The example uses a real `StateGraph` and a fake chat model:

```text
source facts → compose display text → prepare speech → ready
                                          └────────→ clarify retained date/time
```

`facts`, `display_text`, `speech_text` and `changes` are separate state fields.
The speech node is mandatory. The source date must be recognized and validated
as one complete date. Otherwise, or when a date/time in the reply is retained,
the node produces an empty speech string and a clarification request. The application chooses how
to deliver that question; it should send a booking reading to TTS only when
`status == "ready"`.

This is a bounded date-policy example, not a validator for every
booking field or an arbitrary model reply. The fake model deterministically
echoes synthetic facts; there is no real booking write, TTS call, checkpoint
resume or paid service. Repeated invocation is tested to preserve input and
produce the same output. A real application must validate its source facts and
preserve its own idempotency controls around external side effects.
