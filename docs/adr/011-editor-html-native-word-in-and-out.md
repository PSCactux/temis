# ADR 011 — The editor works on HTML; Word goes in and out through a structural subset

**Status:** accepted · 2026-11

## Context

Lawyers write in Word. A reader is not enough: the firm's documents must be edited where they are
answered from, reviewed and cited, and the result must leave as `.docx`. A full Word clone is out of
reach; a lossy round trip is acceptable if it is honest about what it keeps.

## Decision

- The editor is TinyMCE 6 (MIT), vendored under `web/vendor/` and served by the app, so nothing is
  loaded from the internet. Toolbar without menus: blocks, emphasis, alignment, lists, tables, links,
  find and replace, page breaks, word count.
- The firm's own documents are `.html` files in the documents folder, the editor's native format. They
  are indexed like any other document through `html_to_text`, which produces the Markdown-like text
  the chunker already understands (headings become sections, lists and tables keep their lines), so
  citations keep opening at the right passage.
- A `.docx` opens in the editor through `mammoth` and is indexed through the same text; saving it
  creates an editable `.html` copy and offers to drop the original from the index so the content is not
  duplicated. Any document exports to `.docx` through `python-docx` with the same subset: headings,
  paragraphs, bold/italic/underline/strike, lists, quotes, tables, page breaks, alignment.
- Every save over an existing document keeps the previous content in `.versions/<name>/`; versions can
  be viewed and restored, and restoring keeps the replaced text as a version too. Hidden folders are
  never indexed.
- Fragment ids carry the file extension, so `contrato.md` and `contrato.html` are two documents in the
  index (before, same-stem files overwrote each other).

## Consequences

- What Word layout does not survive: page setup, headers and footers, footnotes, tracked changes,
  images. The export is a clean document in Times New Roman 12 that Word opens and the firm formats.
- A chat answer and the corrected draft of a review open in the editor with one click, which is how
  Temis's text becomes the firm's document.
