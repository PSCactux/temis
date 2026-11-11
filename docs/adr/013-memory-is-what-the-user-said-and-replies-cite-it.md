# ADR 013 — Memory holds what the user said; a reply from memory cites it; an attachment is a fragment

**Status:** accepted · 2026-11

## Context

Three things showed up in use. A reply from memory ("yes, he worked there") gave no way to see what it
rested on. A document answer, repeated in a chat follow-up, had become a memory *fact* and from then on
the assistant cited memory instead of the document. And a question about an attached file was answered
from the file but cited a fragment number that did not exist, because attachments travelled as message
text, not as fragments.

## Decision

- Memory is fed only with the user's own words, on every route. What Temis replied is either document
  content (cited there) or the model's text (not a fact). The extractor is told what a fact is (durable,
  stated as information) and what is not (requests, questions, document content, replies). A fact can be
  forgotten from the UI; the memory list shows text, date, matter and origin.
- Recalled memory enters the prompt as numbered blocks (`<memory id="m1" kind="fact" when="…">`). A
  conversational reply cites what it rests on as `[m1]`; markers that name no block are removed in code,
  and the cited items are listed under the reply like the sources of a document answer. The UI opens the
  remembered text on click.
- Files attached to a message are chunked for that turn and placed first among the fragments of the
  documents route, so they are graded, cited `[n]`, verified and shown like any fragment; they are never
  indexed. A cited attachment fragment travels with the answer (the UI shows it with the quote marked).

## Consequences

- Fewer facts, better ones: a summary request leaves no trace in memory. Recall is bounded by the
  configured top-k, so memory size never grows the prompt; what grows is noise, and this cuts it.
- A fact that is also in a document is still a fact if the user stated it; the router sees the
  document titles and prefers documents for document questions.
