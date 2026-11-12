# Coding assistants and Arabic voice agents: next milestone

Researched November 11, 2026. This is a product recommendation, not a claim that
these integrations are implemented, that demand is proven, or that the package
outperforms the projects below. Sources are official docs, upstream repositories
and research-paper abstracts. No competitor accuracy/latency comparison was run.

Implementation update: the JSON CLI, source skill and LangGraph example are now
implemented in 0.5.0; see [usage and limits](agent-integrations.md).
The proposals below preserve the original rationale. Fresh-host skill activation
and the docs-only comparison have not been measured.

## Recommendation

Make the package an inspectable Arabic speech-preparation step that developers
can add to an existing voice agent. Prioritize a stable JSON command and one
reusable integration skill, then a tested LangGraph example. Keep original
display/business data alongside derived speech text. The library should run at
the application's output boundary every time; an LLM should not have to decide
whether to call it.

Adding an autonomous agent, a dialect prompt wrapper, or an MCP server alone
does not create that value. The actual value must come from correct readings,
explicit ambiguity handling, reproducible checks and less integration work.

## Two different meanings of “agent integration”

**Codex/Claude Code integration is a developer experience.** A skill teaches a
coding assistant to install the correct released version, choose conservative
policies, wire the application, and run a regression check. It helps adoption
and correct usage; it does not run inside the resulting application's calls.

**LangGraph/voice-agent integration is runtime behavior.** A deterministic
function takes final text plus application policy, returns speech text and
decisions, and leaves the original text intact. The current normalizer can be
called from such a node already, but no tested LangGraph adapter/example is
included in this change.

## What existing projects establish

| Source | What is already available | Consequence for this package |
|---|---|---|
| [NeMo Text Processing](https://github.com/NVIDIA/NeMo-text-processing) and its [language matrix](https://docs.nvidia.com/nemo-framework/user-guide/25.04/nemotoolkit/nlp/text_normalization/wfst/wfst_text_normalization.html#language-support-matrix) | WFST text normalization and inverse normalization, including Arabic. Apache-2.0. Upstream documents Pynini/OpenFst installation constraints. | Arabic normalization is an existing category. Compare against it before claiming novelty or better coverage; a bounded integration path without native WFST dependencies can still be useful. |
| [CAMeL Tools](https://github.com/CAMeL-Lab/camel_tools) | Arabic preprocessing, morphological modeling/disambiguation, dialect identification and other NLP components. MIT library; individual data packages have their own entries/licenses. | Do not market four word substitutions as dialect understanding or rebuild a general Arabic NLP toolkit. Assess a backend only when a concrete need justifies model/data installation. |
| [Mishkal](https://github.com/linuxscout/mishkal) | Rule-based Arabic vocalization, library/CLI interfaces and linguistic analysis. Upstream states GPL licensing. | Diacritization and a CLI alone are not new. Do not copy its code into this MIT project without checking the licensing implications. |
| [LiveKit audio customization](https://docs.livekit.io/agents/multimodality/audio/customization/) | Streaming pronunciation substitutions and TTS customization hooks. | A plain replacement dictionary is weak differentiation. Arabic structured readings, preservation policies and a compatible streaming contract are the more relevant gaps to investigate. |
| [LangGraph Graph API](https://docs.langchain.com/oss/python/langgraph/graph-api) | Nodes are ordinary sync/async functions operating on state. | Start with an example function, not a new graph framework or a mandatory graph dependency. |

The package's current number conversion already depends on `num2words`; it
should not take credit for implementing that library's cardinal engine. The new
normalizer adds policy and structured-span handling. It still installs the
package's existing LangChain/Pydantic dependencies and is not a dependency-free
replacement for the above tools.

## Research that changes the design decision

[More Data, Fewer Diacritics: Scaling Arabic TTS (2026)](https://arxiv.org/abs/2603.01622)
compares Arabic TTS training with and without diacritics. Its abstract reports
that diacritized models generally fare better, while more training data reduces
the disadvantage of unvocalized input. This does not evaluate ElevenLabs or this
package. It supports testing each provider and keeping automatic diacritization
optional, rather than promising that adding marks always improves modern TTS.

[Shallow Fusion of WFST and Language Model for Text Normalization (2022)](https://arxiv.org/abs/2203.15917)
describes the tradeoff between contextual ambiguity and unrecoverable neural
normalization errors. Its approach constrains candidates with rules and ranks
them with a model. This is relevant precedent for explicit ambiguity policies;
it does not establish Arabic performance for our implementation. A future model
should select or request a known interpretation rather than freely rewrite
critical identifiers and amounts.

Our own [targeted ElevenLabs check](speech-audio-check.md) is consistent with a
narrower pitch: the new path avoids observed legacy conversion regressions, but
raw speech already worked on those examples. No overall speech improvement or
market differentiation has been established yet.

## Proposed next slice: JSON command plus a shared skill

The [OpenAI skills documentation](https://learn.chatgpt.com/docs/build-skills)
and [Claude Code skills documentation](https://code.claude.com/docs/en/skills)
both describe `SKILL.md` workflows using the open Agent Skills format. Use one
provider-neutral source skill, with documented installation into Codex's
`.agents/skills/<name>/` and Claude Code's `.claude/skills/<name>/` for local
repository use. Support varies by host/session; test actual activation rather
than treating the shared file format as universal installation compatibility.

Proposed command (not implemented):

```text
python -m langchain_arabic normalize --json
```

Read UTF-8 JSON from stdin, return JSON to stdout, diagnostics to stderr, and
nonzero status for invalid configuration/input. This avoids shell quoting of
Arabic text and gives both coding assistants and other languages a deterministic
interface. Return `display_text`, `speech_text`, the existing source-offset
decisions and explicit warnings for retained ambiguous/invalid structures.
No API keys, network calls or automatic dependency installation in the command.

The skill should do one job: “Add Arabic speech preparation to this application's
TTS path.” It should inspect the current provider path, choose the installed API
version, disable overlapping legacy number conversion, preserve display text,
add one regression using the application's own synthetic example, and run it.
Do not advertise installation of the new API from PyPI until a release includes it.

Acceptance criteria:

1. JSON round-trip tests cover Arabic/English mixing, IDs with leading zeros,
   precision, invalid dates and configuration errors; stdout remains machine-readable.
2. A fresh Codex session and a fresh Claude Code session each use the skill to
   integrate one small key-free fixture app and pass its regression. Record exact
   host versions, prompts, installed package version and failures.
3. Compare the same tasks with ordinary docs alone. Record completion, incorrect
   API calls and required corrections; do not infer popularity or download growth.

A skill is a distribution aid, not the product's core correctness argument.
Start with a repo-distributed source; consider marketplace packaging only after
the workflow has been exercised. Do not add a public remote service yet.

## Proposed LangGraph example

Use a key-free booking/support scenario with a real graph and a fake model:

```text
booking tool facts → assistant display text → Arabic speech preparation → TTS
                              │                         │
                         retained original          decision records
```

Use separate state fields for source facts, display text, speech text and
normalization decisions. Call `normalizer.explain(display_text)` in an ordinary
node; serialize its frozen records with `dataclasses.asdict` for graph state.
Route retained ambiguous dates to clarification when the application requires an
exact booking date. Keep the normalized speech out of the conversation/business
record so later agent turns do not treat spoken expansions as source data.

Known booking fields can provide explicit date/currency/ID interpretation instead
of asking a regex or LLM to infer it. Verifying those known fields is a bounded
check; it is not verification of the entire reply's meaning. Never send the TTS
request in a graph step that might be replayed without accounting for duplicate
external calls.

Acceptance: execute the graph without credentials, verify that source/display
state is unchanged, test supported and ambiguous input paths, and test replay
behavior before adding a real TTS side effect. Keep LangGraph optional and
document the example's supported Python/dependency versions.

## Later, only when justified

- **Sentence-aware streaming:** measure time to first audio with the actual
  provider. Test chunk boundaries inside IDs, dates, decimals and protected code.
  The current normalizer buffers the whole reply; a streaming voice workflow
  must account for this added delay.
- **Provider adapters:** LiveKit or another voice framework after a real example
  demonstrates that setup code is repeated. Retain the provider's native owner
  for TTS and interruptions; avoid a second agent lifecycle.
- **MCP:** useful when a real non-Python or remote client needs callable tools.
  A Python function/CLI is a smaller starting point for a local text operation.
- **Dialect rewriting:** a distinct capability requiring evidence and a concrete
  dialect, not a follow-up justified merely by calling the package an agent tool.

No downloads, adopters, superiority scores, or latency claims are inferred from
this research. The next defensible promise is easier integration with explicit
preservation rules; real-world value and growth still need to be measured.
