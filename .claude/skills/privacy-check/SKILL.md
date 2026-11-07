---
name: privacy-check
description: Verify that personal data cannot reach a model, a store or a trace through a given path (a new tool, loader, prompt or endpoint). Use before merging anything that adds an input or an output.
---

# Privacy check

- Inputs: user messages are pseudonymised in `recall`; tool outputs in `tools()`; document chunks at
  ingest (`Assistant.ingest`). A new input must join one of these, or get its own call to
  `anonymizer.anonymize`.
- Outputs: only `Assistant.reveal` restores originals, and only for the person using the interface.
  Nothing written to SQLite (turns, audit, memory_writes, alarms), Qdrant or spans may be revealed.
- Prove it: a test in the style of `test_personal_data_never_reaches_the_model` that feeds identifiers
  through the new path and asserts `contains_personal_data(...) == []` on everything the model saw,
  plus an eval of kind `privacy` if the path is user-facing.
- Names come from the fast model and are filtered by `plausible_name`; check `Anonymizer.mapping()`
  after an ingest for role words or public bodies that slipped through, and extend the lists.
- Traces: `TRACE_CONTENT=false` hides inputs and outputs entirely; with it on, spans contain the
  pseudonymised text only. Never set it on against a collector outside the firm.
- Instructions hidden in data: any new text the model reads that the firm did not write (a file, a
  page, an attachment, a memory) must be wrapped with `prompts.data_block` and must not be placed in
  a system prompt; a fact may only be extracted from the user's words when the reply carries external
  content (see ADR 6 and `tests/test_untrusted.py`).
