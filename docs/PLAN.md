# Plan

Each phase ends with a verifiable "done" and a commit.

## Phase 1 — core
Graph with `recall`, `route`, `chat`, documents route, `memorize`; memory store with
three collections; telemetry (tracing, metrics, audit, path); CLI (`chat`, `ask`,
`ingest`, `eval`, `report`); evals for identity, memory and citations.
**Done when:** `pytest` passes with the fake model, `aroa eval` passes identity,
memory and citation cases with a local Ollama model, and `/metrics` lists the turns.

## Phase 2 — tools
Alarms with scheduler, web search and reading, project tools confined to a folder,
`ask_user` interrupts, `verify` node, loop guards. FastAPI server with SSE streaming,
web UI with the path view and feedback.
**Done when:** evals for alarms, web and project pass, a turn that claims an unwritten
file is sent back and corrected, and an interrupted thread resumes after the answer.

## Phase 3 — deployment
Run against an OpenAI-compatible local gateway on the target machine, traces to its
collector, comparison table against the previous assistant on the same cases.
**Done when:** the table is in the README with real numbers.

## Phase 4 — polish
README with diagram and real transcript, GitHub Actions (ruff + pytest with the fake
model), skills for adding a tool and running evals, ADRs.
