# Working on Temis with an agent

This file is read by coding agents (Claude Code, Codex, Cursor...) before they touch the repo.
Humans: it is also the shortest accurate description of how the project is built.

## What this is

A law firm's assistant that runs on the firm's own machine: a LangGraph state machine with RAG
memory, pseudonymisation of personal data before any model call, a policy gate, verified claims and
built-in observability. `docs/SPEC.md` is the contract; `docs/PLAN.md` the phases; `docs/adr/` the
reasons behind the non-obvious choices. Read the spec before changing behaviour.

## Commands

```bash
uv sync --extra tracing        # install (Python 3.12, uv)
uv run ruff format . && uv run ruff check .
uv run pytest -q               # graph logic with a scripted model, no network, < 2 s
AROA_INTEGRATION=1 uv run pytest -q   # also the real vector store (needs the embedding models)
uv run aroa ingest             # index data/docs (copies data/samples on first run)
uv run aroa ask -v "..."       # one turn, printing the path
uv run aroa eval               # whole-turn evals with the configured models, writes evals/results/
uv run aroa serve              # web app on http://127.0.0.1:8000
uv run langgraph dev           # LangGraph Studio
```

## Map

| Where | What |
|---|---|
| `src/aroa/graph.py` | the state machine: nodes, edges, limits. Control flow lives here and nowhere else |
| `src/aroa/prompts.py` | every prompt. Do not write prompt text inside nodes |
| `src/aroa/schemas.py` | structured outputs. Every model decision is a Pydantic model |
| `src/aroa/privacy.py` | pseudonymisation; the model proposes names, the code decides |
| `src/aroa/verify.py`, `citations.py` | what counts as proven: claims against evidence, quotes against fragments |
| `src/aroa/memory/` | Qdrant store (facts, episodes, documents), chunking, loaders |
| `src/aroa/tools/` | one module per route; `registry.py` says which route gets which tools |
| `src/aroa/telemetry/` | SQLite schema, metrics, audit, OpenTelemetry |
| `src/aroa/app.py` | wiring and the turn runner (streaming, metrics row, reveal) |
| `src/aroa/server.py`, `web/` | API, SSE, alarm scheduler, UI |
| `src/aroa/evaluation.py`, `evals/dataset.jsonl` | whole-turn evals by kind |
| `tests/` | `FakeChatModel` + `FakeMemory`; the graph is tested by scripting the model's answers |
| `data/persona/` | who the assistant is; `data/samples/` fictitious corpus; `data/docs`, `data/projects` are yours (ignored) |

## Rules

1. **The model proposes, the code decides.** Retries, limits, what is a loop, what is proven, what is a
   name: in code, tested. Never move one of these decisions into a prompt.
2. **Every model call returns a schema** (`schemas.py`) except the final reply. New decision → new schema.
3. **Nothing personal reaches a model, a store or a trace.** New input paths (a tool, a loader) must pass
   through the anonymizer like the existing ones. Add a test that proves it with `contains_personal_data`.
4. **Each route gets only its tools** (`tools/registry.py`). Adding a tool means adding it to one route and
   teaching `verify.py` how to check the claim it enables.
5. **Every node adds a trace entry** (`trace.py`) in plain language. The UI, the CLI and the evals read it.
6. **Tests script the model, they do not mock the graph.** A new branch in the graph gets a test in
   `tests/test_graph.py` with the exact sequence of model answers that reaches it.
7. **Evals are the acceptance test.** Behaviour changes need a case in `evals/dataset.jsonl` and a run
   whose numbers go into the README table.
8. No machine-specific values in the repo: addresses, hostnames, keys and paths live in `.env`.
9. Commits: atomic, English, imperative, describing the technical change only.

## When something looks wrong
- Content the firm did not write reaches a prompt as plain text: wrap it with `prompts.data_block`
  and keep it out of system prompts (ADR 6). `tests/test_untrusted.py` shows what is expected.

- A small model returns empty structured output: check `LLM_REASONING` (thinking eats the budget).
- Names not pseudonymised: `plausible_name` rejected them (role words, public bodies) or the extractor
  did not propose them. Print `Anonymizer.mapping()`.
- Memory recalled across matters: the filter must be on each Qdrant prefetch (see ADR 3).
- A turn "lies": look at the `verify` step in the path; if the claim kind is not covered, add it.
