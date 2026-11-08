# ADR 009 — Long documents are analysed whole, reduced in code, cited or dropped

**Status:** accepted · 2026-11

## Context

Lawyers need the whole contract read, not the three fragments retrieval liked. Context windows are a
budget; a 200-page document is not a prompt. Summaries written in one pass repeat themselves across
sections, drift from the text, and cannot be checked.

## Decision

- Map: every stored fragment of the document is read in order by the fast model with the goal, and
  returns findings of a fixed set of kinds, each with a verbatim quote. The fragment travels tagged as
  data (ADR 006).
- Reduce, in code: a finding whose quote is not verbatim in its fragment, or whose quote is a planted
  instruction, is dropped and counted. Findings that say the same thing (same quote, or statements that
  share most content words) are merged into one that keeps every location. Findings are numbered.
- Overview: the large model may write a few sentences from the numbered findings only; it is kept only
  if every citation exists and every sentence carries one; otherwise it is cut or dropped and the
  report says so.
- Delivery: a job with progress (fragments done / total) that the UI polls; the report cites `[n]` and
  each citation opens the fragment at the quote. Each model call takes the turn lock, so a long analysis
  never blocks a conversation for more than one call.

## Consequences

- The budget is time: minutes for a long contract on a local 8B model, with the Mac free in between.
- Repetition across sections is handled where it is cheap and checkable, in the merge step, not by
  asking the model not to repeat itself.
- What the reader sees is exactly what the document says, with the location; what the model added
  (statements, overview) is always one click from the proof. A finding without proof is not shown.
