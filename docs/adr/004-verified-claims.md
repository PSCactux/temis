# ADR 004 — Claims about actions are verified in code

**Status:** accepted · 2026-11

## Context

Small models say "done" when nothing was done: the alarm was never created, the file never written,
the page never opened. Asking the model to be honest does not work reliably; checking does.

## Decision

After any tool route, if no tool was called at all the reply is rejected in code: nothing can have
been done. Otherwise a fast structured call lists the actions the reply states as done (kind +
target). Code checks each one against the turn's evidence: successful tool calls with matching
arguments, and the artefact where there is one (the file exists, the alarm row exists). Unproven
claims remove the reply and send the turn back to `act` once with the list of failures; if the retry
still fails, the reply is delivered with an explicit "unverified" note.

The same principle applies to document answers: citations must point to provided fragments and quotes
must appear verbatim in them (`citations.py`), with one retry.

## Consequences

- One extra fast call per tool turn; none for chat.
- A claim kind the verifier does not know is checked loosely ("some tool succeeded"); new tools must
  add their kind.
- The model's extraction is itself a model call; it can miss a claim. Evals of kind `alarms` and
  `project` measure the end result, not the extraction.
