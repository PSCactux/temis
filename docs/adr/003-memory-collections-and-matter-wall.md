# ADR 003 — Three memory collections and a wall between matters

**Status:** accepted · 2026-11

## Context

The previous assistant kept memory in one Markdown file searched by keyword, injected the last weeks
on every wake-up, and relied on the model remembering to call a "remember" tool. It also had no notion
of which matter a fact belonged to, which in a law firm is a confidentiality problem, not a nuisance.

## Decision

Qdrant with hybrid vectors (dense + BM25, fused by RRF) and three collections: `facts` (one sentence
each, extracted from every turn by the fast model, deduplicated by cosine similarity before writing),
`episodes` (one per turn, re-ranked by recency with a half-life), `documents` (chunked files, indexed
incrementally by file hash). Every fact and episode carries the matter it was learned in. Recall
filters to the current matter plus general memory, and the filter is applied on every prefetch of the
hybrid query, not only on the fused result.

## Consequences

- Facts are captured whether or not the model thinks to save them; duplicates are skipped and logged.
- What was said while working on matter A is never recalled in matter B (integration test).
- The persona is never retrieved: it is always in the prompt, which is what keeps identity stable.
- Recall has a floor (2026-11): fusion ranks whatever the collection holds, so without one the same
  top facts came back for every message. An item is recalled when it is close in meaning to the message,
  or fairly close and sharing words with it; the floors are measured for the embedding model in use
  (`fact_floors`, `episode_floors`). An empty recall is the usual answer to small talk.
