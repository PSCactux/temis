# ADR 006 — Untrusted content is tagged, kept out of the system prompt and never memorised on its own

**Status:** accepted · 2026-11

## Context

Most of what the model reads was not written by the firm: documents a client sent, web pages, files
in a matter folder, attachments, and facts the assistant itself extracted in earlier turns. Any of
them can contain text that looks like an instruction. A small local model does not reliably tell
where an instruction ends and a quotation begins, and once such a sentence is stored as a "fact" it
is injected into the system prompt of every later conversation.

Before this decision, document fragments were pasted into the router's system prompt (the same call
that applies the firm's policy), tool results went to the model as plain text, memory was a plain
list, and the fact extractor read the whole exchange, including replies built from documents.
`run_command` ran through a shell.

## Decision

- **One wrapper for everything untrusted.** `prompts.data_block(tag, text, **attrs)` wraps content in
  a tag that names its origin (`<fragment n= source=>`, `<tool_result tool= path|url=>`,
  `<attachment name=>`, `<memory>`, `<document_hints>`, `<message>`). Closing tags inside the content
  are defused, so a planted `</tool_result>` cannot end the block. The persona tells the model that
  those tags hold data, which it reports and never obeys.
- **Nothing retrieved goes into a system prompt.** The router receives the user's message, the
  document hints and the remembered facts as a tagged human turn; its system prompt holds only the
  firm's instructions and policy. The document writer receives `<fragments>`.
- **Memory is written from the user's words on routes that read external content.** On `documents`,
  `web` and `project`, the reply carries text nobody at the firm wrote, so the fact extractor sees the
  user's message only. On `chat` and `alarms` it sees the exchange. Every fact stores an `origin`
  (`documents:user`, `chat:exchange`) so a poisoned source can be traced and purged.
- **No shell.** `run_command` accepts one command from an allowlist (`ALLOWED_COMMANDS`, read-only by
  default), parsed with `shlex` and executed without a shell; pipes, redirections, substitutions,
  chaining, absolute or climbing paths, `find -exec/-delete` and `sed -i` are refused in code.

## Consequences

- An instruction planted in a document, page or file can still be *read*; it cannot change the
  route or the policy decision, cannot end its own block, cannot become a durable fact and cannot
  run a command the allowlist does not name. The existing checks (citations verified against the
  fragment, claims verified against evidence, placeholders instead of personal data) keep standing
  behind it.
- Models trained on XML-delimited prompts follow the tags well; the 8B test model lost nothing on
  the document evals after the change.
- `run_command` can no longer do work that needs a pipeline; the model is told to use the file
  tools instead. A firm that needs more adds commands to `ALLOWED_COMMANDS` knowingly.
- Not covered: a document that persuades the lawyer, not the model. That is a human problem; the
  path of each turn shows what was read.
