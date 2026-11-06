# ADR 002 — Pseudonymise before the model, reveal only at the interface

**Status:** accepted · 2026-11

## Context

A law firm's assistant handles names, identity numbers, bank accounts and phone numbers of clients,
witnesses and counterparties. Even with local models, that data would otherwise end up in prompts,
vector stores, checkpoints, logs and traces, which are kept longer and read by more people than a
conversation.

## Decision

Everything that enters the graph is pseudonymised first: the user's message in `recall`, tool outputs
in `tools`, document chunks at ingest. Identifiers are matched by patterns in code. Names are proposed
by a short structured call to the fast model and accepted only if `plausible_name` says they look like
a person or a private organisation (role words such as "the client" and public bodies are never
replaced). Each original maps to one stable placeholder (`[PERSON_3]`) in a local SQLite table, shared
across conversations and documents so the model sees a consistent world. Only `Assistant.reveal`
restores originals, and only for the person at the interface. `redact` makes the mapping irreversible
for documents that leave the firm.

## Consequences

- Models, stores, checkpoints and spans never contain personal data; this is tested.
- A revealed short form shows the full name (one placeholder per person).
- Quality of name detection depends on the fast model; the code filters its mistakes but cannot add
  names it never proposed. `PRIVACY_MODE=identifiers` keeps only the pattern-based part.
