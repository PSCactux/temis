# Aroa Assistant

A law firm's assistant that runs entirely on the firm's own machine. It is built as an explicit
**LangGraph** state machine with **RAG memory** walled by matter, **pseudonymisation** of personal
data before any model call, a **policy gate**, **claims verified in code**, and **observability**
built in (OpenTelemetry traces, per-turn metrics, audit log, a plain-language path for every reply,
evals). Local models through Ollama or any OpenAI-compatible gateway; Anthropic if you want it.

```text
$ uv run aroa ask -v --matter 2026-014 "El director financiero del cliente es Pedro Sanz, DNI 44556677K, \
  y prefiere un acuerdo antes que ir a juicio. ¿Qué dice la hoja de encargo si hay acuerdo?"
· Recall: Personal data in the message was replaced by placeholders. Nothing relevant in memory.
· Route: documents. Query: «honorarios en caso de acuerdo, hoja de encargo expediente 2026/014»
· Search documents (0.4s): 6 fragments from 3 documents (hybrid search, 2 queries).
· Keep what helps (2.4s): 3 of 6 fragments kept.
· Answer from documents (7.2s): Coverage full, 1 citation.
· Check citations: Every quote appears verbatim in its fragment.
· Memorize (5.4s): 2 new facts stored, 0 already known.
    - stored · [PERSON_2] es el director financiero de [ORG_8].
    - stored · [ORG_8] prefiere resolver el asunto mediante acuerdo antes que ir a juicio.

Si el asunto se resuelve mediante acuerdo, los honorarios se reducen a 1.800 euros más IVA [1].

**Sources**
- [1] Hoja de encargo profesional — Expediente 2026/014 — Cláusula 2. Honorarios (`hoja_de_encargo_honorarios.md`)
  > «Si el asunto se resuelve mediante acuerdo antes de la vista, los honorarios se reducirán a 1.800 euros más IVA.»
```

The model saw `[PERSON_2]` and `[ORG_8]`, never "Pedro Sanz" or the DNI; the lawyer sees the real
names because the interface restores them. The facts stored in memory carry the matter `2026-014`
and are never recalled while working on another matter.

## What it does

| | |
|---|---|
| **Converses and remembers** | Facts are extracted from every turn, deduplicated and stored with the matter they were learned in. Recall is hybrid (dense + BM25) over facts, past episodes and documents. |
| **Answers from the firm's documents** | Corrective RAG with `[n]` citations and verbatim quotes, both verified in code; abstains when the documents do not answer. Prompt injection inside a document cannot become a cited fact. |
| **Untrusted content is quarantined** | Documents, tool results, pages, attachments and memory reach the model inside tags that name their origin, never in a system prompt; facts are extracted from the user's words only when the reply carries external content; `run_command` runs one allowlisted command without a shell (ADR 6). |
| **Keeps personal data out of models** | Identifiers by patterns, names by a model proposal filtered in code; stable placeholders; revealed only at the interface. `redact_file` produces black-box copies for sharing. |
| **Applies the firm's policy** | Fabricating or altering evidence, deceiving, crossing matters, leaking data: blocked before any work, with the reason. |
| **Sets deadline reminders** | Alarms in SQLite; a scheduler wakes the conversation when one is due. |
| **Reads the web** | Search (SearXNG or DuckDuckGo lite) and read pages or PDFs as text; private addresses refused. |
| **Works in matter folders** | Read, grep, edit, write, run commands and redact files, confined to the matter's folder; its `AGENTS.md` is in the prompt. |
| **Verifies its own claims** | After any tool route, what the reply says it did is checked against the evidence (tool calls, files, alarm rows). Unproven claims send the turn back; a second failure is delivered marked as unverified. |
| **Asks when it must** | `ask_user` pauses the thread with its state saved; the next message resumes it. |

## How it works

```mermaid
graph TD
    START([message]) --> recall --> route
    route -- blocked by policy --> END1([refusal])
    route -- chat --> chat --> memorize
    route -- documents --> retrieve --> grade --> answer_documents --> check_citations
    check_citations -- bad citations, retry left --> answer_documents
    check_citations --> finalize_documents --> memorize
    route -- alarms / web / project --> act
    act -- tool calls --> tools --> act
    act -- reply --> verify
    verify -- unproven claims, retry left --> act
    verify --> memorize
    memorize --> END([reply])
```

`recall` pseudonymises the message and searches memory (no model). `route` is one structured call
that returns the policy verdict, the route and a standalone query. Each route binds only its own
tools. `verify` extracts the reply's claims with the fast model and checks them in code. `memorize`
runs after the reply is streamed. Details and the reasons behind them: [docs/SPEC.md](docs/SPEC.md)
and [docs/adr/](docs/adr/).

## Supervision

| | Where |
|---|---|
| Path of every turn (what was recalled, routed, called, verified, stored) | web UI "path" button, `aroa ask -v`, `GET /api/turns/{id}` |
| Per-turn metrics: route, latency, per-node seconds, model calls, tokens, tool calls, claims, facts | SQLite `turns`, `GET /metrics` (Prometheus), `aroa report` |
| Audit: every tool call (arguments, outcome, result hash) and memory write | SQLite `tool_calls`, `memory_writes`, in the turn's path |
| Traces of model calls, tools and nodes | OpenTelemetry → Phoenix, Langfuse, Tempo... (`OTLP_ENDPOINT`) |
| Feedback per reply | 👍 👎 in the UI, `POST /api/turns/{id}/feedback` |
| Evals by kind, with the real models | `aroa eval`, results in `evals/results/` |

Nothing in the telemetry calls a model. Traces hide contents by default and contain only
pseudonymised text when they do not.

## Evaluation

`evals/dataset.jsonl` holds whole-turn cases by kind: identity, memory across threads, the wall
between matters, privacy (identifiers never reach the model, the stores or the checkpoint), policy,
documents (citations, abstention, prompt injection inside a document), alarms, web and matter-folder
work. `aroa eval` runs them against the configured models in a separate data folder and stores every
reply and check in `evals/results/`.

Model `ollama:qwen3:8b` (fast: `ollama:qwen3:8b`), privacy mode `full`, 20261107.

| Kind | Passed | Median seconds |
|---|---|---|
| alarms | 2/2 | 15.4 |
| chat | 2/2 | 12.8 |
| documents | 15/18 | 19.6 |
| identity | 1/1 | 3.8 |
| memory | 1/3 | 26.2 |
| policy | 3/3 | 5.2 |
| privacy | 2/2 | 17.8 |
| project | 2/2 | 15.9 |
| wall | 1/1 | 9.8 |
| web | 0/1 | 35.9 |
| **total** | **29/35** (83%) | |

All model calls run at temperature 0, yet an 8B model still varies between runs: full runs on the
same code have scored between 74 % and 86 %. The failures move around the same places: the router
choosing `chat` or `documents` when memory holds the answer, retrieval picking a neighbouring section,
a quote that is not verbatim, and the web case, which passes or fails on how the model phrases what it
read. A larger model improves the `documents`, `memory` and `web` rows; the privacy, policy, wall and
verification rows depend on code and hold with any model.

## Run it

```bash
uv sync --extra tracing
cp .env.example .env            # defaults to Ollama; `ollama pull qwen3:8b`
uv run aroa ingest              # indexes data/docs (a fictitious Spanish corpus is copied in on first run)
uv run aroa chat --matter 2026-014
uv run aroa serve               # http://127.0.0.1:8000
```

Your documents go in `data/docs/` (or drop `.md`, `.txt` and `.pdf` files on the web UI, which saves
them there and indexes them pseudonymised), matter folders in `data/projects/<matter>/`, the persona in
`data/persona/*.md`. None of those are tracked by git except the samples. A file attached to a message is
read for that turn only unless you tick "keep in documents"; clicking a citation opens the source with the
quote highlighted. `ASSISTANT_LANGUAGE=es` puts the texts the code writes (sources, blocked replies, the
path of a turn) in Spanish; the UI has its own ES/EN switch.

Tracing: run a collector (for example `uv run phoenix serve` from the `arize-phoenix` package) and set
`OTLP_ENDPOINT=http://localhost:6006/v1/traces`. LangGraph Studio: `uv run langgraph dev`.

## Development

```bash
uv run pytest -q                       # graph logic with a scripted model, no network
AROA_INTEGRATION=1 uv run pytest -q    # plus the real vector store
uv run aroa eval --only documents      # one kind of eval
```

How the repo is built with agents, and the rules they follow: [AGENTS.md](AGENTS.md),
[CLAUDE.md](CLAUDE.md) and the skills in [.claude/skills/](.claude/skills/).

## License

MIT.
