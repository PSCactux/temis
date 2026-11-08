# ADR 010 — A draft is reviewed in a separate pass, and only the code says "wrong"

**Status:** accepted · 2026-11

## Context

Writing and checking are different jobs. A model that reviews its own draft in the same turn tends
to confirm it; a model asked "is this citation right?" answers with the confidence of the question.
The things that most often sink a brief are mechanical: a quote that is not verbatim, an article
number that is off by three, a paragraph pasted twice. Those can be decided without a model.

## Decision

- The review is a job of its own (`review.py`, kind `review` in the `jobs` table), run over the text
  of a draft the user pastes, attaches or picks from the documents. It never rewrites the draft; it
  returns issues anchored to verbatim passages and, where the code knows the right text, a proposed
  change the user accepts or rejects one by one.
- In code: paragraphs that repeat an earlier one (normalised equality or ≥ 0.85 Jaccard of content
  words) → delete the later one. Quoted passages («…», “…”) not verbatim in the indexed documents →
  the closest original sentence (SequenceMatcher ≥ 0.7) aligned to the quoted span is the fix; with no
  close original the quote is "not found". Legal citations go through the checker of ADR 008; a quote
  in the same paragraph as an article citation is taken as that article's text, and when it lives in
  another article of the same norm the fix is the right number.
- The model reads the draft in windows (≤ 5000 characters of whole paragraphs) and points at legal
  issues; a pointer survives only if it quotes the draft verbatim and is not a planted instruction. Its
  findings are always warnings. Only code-decided findings are errors.
- Privacy as everywhere (ADR 002): the draft is pseudonymised before review so quotes match the
  pseudonymised index; every string of the result is revealed at the edge.

## Consequences

- A draft with a wrong article number, a non-verbatim quote and a repeated paragraph yields exactly
  those findings (eval `draft_review`); the model's part is measured separately as noise on a sound
  draft.
- Corrections are only offered where they can be derived: a wrong number with a quote nearby, a quote
  with a close original, a repeat. An inexistent article without a quote is flagged, not guessed.
- Case law stays unverified (the public database forbids bulk reuse); it is reported as such.
