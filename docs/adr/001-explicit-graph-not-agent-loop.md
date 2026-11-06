# ADR 001 — An explicit graph instead of a hand-written agent loop

**Status:** accepted · 2026-11

## Context

The previous assistant was a ReAct loop: one prompt, every tool, the model deciding at each step.
Over months it accumulated guards (identical-call counters, read-only streak limits, output caps)
as patches on the loop. Small local models got lost choosing among twenty tools, replies claimed
actions that never happened, and a restart lost whatever the loop was doing.

## Decision

The assistant is a LangGraph `StateGraph` with typed state and named nodes: `recall`, `route`,
`chat`, the documents branch, `act`/`tools`, `verify`, `memorize`. Each route binds only its tools.
Retries, limits and loop detection are edges and counters in `graph.py`, tested with a scripted model.
A SQLite checkpointer makes every thread resumable; questions to the user are `interrupt()`s.

## Consequences

- A chat turn costs one main-model call plus one fast routing call. Tool turns cost the loop plus
  one verification call. More calls than the raw loop in the worst case, fewer tools per call always.
- Control flow is readable in one file and visible in LangGraph Studio.
- Adding behaviour means adding a node or an edge, with a scripted test, not a longer prompt.
