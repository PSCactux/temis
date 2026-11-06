# ADR 005 — Observability with open standards, computed in code

**Status:** accepted · 2026-11

## Context

Nobody can supervise an assistant they cannot see. The firm needs to know what it does, how long it
takes, what it stores, when it fails, and to be able to prove it afterwards. The data must not
leave the firm and must not depend on one vendor's dashboard.

## Decision

- Traces: OpenInference instrumentation of LangChain/LangGraph, exported over OTLP/HTTP to any
  collector (Phoenix, Langfuse, Tempo). Contents hidden by default; pseudonymised when shown.
- Metrics: one row per turn in SQLite (route, latency, per-node seconds, model calls, tokens, tool
  calls and errors, retrieved/kept fragments, claims checked/failed, facts written, interruption,
  error) and a Prometheus text endpoint.
- Audit: every tool call and memory write, with a hash of the result rather than the result.
- Path: a plain-language trace of every turn, stored with it and shown in the UI and the CLI.
- Feedback: thumbs per reply, tied to the turn id.
- Report: daily aggregates computed in code, never by the model.
- Evals: a dataset of whole turns by kind, run against the real models, numbers in the README.

## Consequences

- The telemetry costs no model calls and no network beyond the optional collector.
- The same `turn_id` ties the UI, the SQLite rows and the spans together.
