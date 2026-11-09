# ADR 015 — A procedure is a checked record, edited through a form

**Status:** accepted · 2026-11 · supersedes the storage and editing part of ADR 012

## Context

ADR 012 kept each procedure as a free document with four sections found by heading name, edited in the
same rich-text editor as any document. That made it easy to break a procedure without noticing: a
renamed heading silently dropped a section, a list item typed as prose lost a field, and a data field
whose name no longer matched the template's `{{gap}}` only showed up as a missing value when drafting.
A procedure is a working tool the firm relies on, so it should not be possible to save one that cannot
be followed.

## Decision

- A procedure is a record saved as `data/procedures/<slug>.json`: name, description, template (a
  document of the library, by file name), data fields (name, hint, whether it takes several lines),
  writing instructions and review checklist.
- It is edited through a form with one control per part: a picker of library documents for the
  template, rows for the fields and the checklist (add, reorder, remove), text boxes for the rest.
- `procedures.check` runs on every change (`POST /api/procedures/check`) and again on save. Errors stop
  the save (HTTP 422): no name, a name already taken, no template, a template not in the library, a
  field without a name, with braces or repeated, and above all a `{{gap}}` of the template with no
  field of that name. Warnings are shown but do not stop it: a field the template never uses, a template
  with nothing to fill, passages for the model without writing instructions, no checklist.
- The template stays an ordinary library document, edited in the document editor. Because it can change
  after the procedure was saved, the procedure is checked again each time it is opened and before
  drafting; a procedure whose template lost its match is shown with the reasons and is not followed
  (HTTP 409) until it is fixed.
- Issues travel as codes with the name they are about, so the page words them in the user's language.
- Markdown procedures with the four headings are still read, and become JSON the first time they are
  saved through the form. Drafting, the review checklist and pseudonymisation are unchanged from ADR 012.

## Consequences

- A saved procedure always matches its template at the time of saving, and a later mismatch is caught
  before any model call.
- The flow is still changed by the firm without a diagram, but through fields rather than free text, so
  the form can say exactly what is wrong and offer the fix (add the fields the template asks for).
- Inline templates of old Markdown procedures are not offered by the form; such a procedure has to point
  at a library document when it is saved.
