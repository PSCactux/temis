# ADR 012 — A procedure is a document with four sections, and the code follows it

**Status:** accepted · 2026-11

## Context

Firms work by kinds of matter: a "reclamación patrimonial" is always written from the same template,
needs the same data, follows the same house rules and is checked against the same list before it goes
out. The ask was to let a user define that flow without a workflow engine.

## Decision

- A procedure is one document in `data/procedures/` (Markdown or HTML written in the app's editor) with
  a title, a description and four sections found by heading name: **Plantilla** (a library document or
  an inline template), **Datos** (a list; each item is a form field, with a hint after the colon),
  **Redacción** (the firm's instructions for writing) and **Revisión** (a checklist).
- Drafting is a background job. `{{Campo}}` placeholders in the template are filled in code from the
  form. Only passages marked `[[instruction]]` are written by the model, one call each, with the firm's
  instructions, the data and the surrounding text as data blocks; a reply that is empty or still holds
  placeholders is rejected and the instruction stays visible rather than invented. The data the user
  typed is checked back against the draft; what appears nowhere is reported. The result opens in the
  editor, unsaved, named after the procedure and the matter.
- The review pass takes the procedure's checklist: the model answers each item over the draft in
  windows, and an item counts as met only with a verbatim quote of the draft (verified in code);
  otherwise it is an issue of kind `lista_revision`, a warning with the model's note.
- Procedures are never indexed as evidence; the template, when it is a library document, is.
- Pseudonymisation as everywhere: the form data and the template are pseudonymised before the model
  sees them and the draft is revealed for the user.

## Consequences

- Predictable and cheap: a ten-field template with three written blocks costs three model calls; the
  rest is substitution. The firm changes the flow by editing a document, not a diagram.
- The sample `reclamacion_patrimonial` ships with its template in the sample documents.
