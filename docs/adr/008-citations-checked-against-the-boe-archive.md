# ADR 008 — Legal citations are checked in code against the firm's archive of the BOE

**Status:** accepted · 2026-11

## Context

Spanish courts already sanction briefs with invented citations; the failure they punish is not using an
assistant but not checking. Checking that a cited article exists, and reading what it currently says,
is mechanical work that a model does badly and code does perfectly, provided the firm has the law at
hand. The firm's archive of the BOE (a sibling project with full provenance) holds the consolidated
legislation with every version of every article and the date each text was fetched.

## Decision

- `aroa.lawcheck` extracts citations with patterns (article numbers with apartados and lists, norm
  references by number or by the abbreviations lawyers use, judgments and ECLI) and verifies each one
  against a `LawSource`: the norm is resolved by alias (`data/laws.json`) or by title in the archive,
  the article is read from the norm's current version, and the result is one of found, article not
  found, norm not found, norm not yet downloaded, or not automatically verifiable.
- The report puts the verdict first, quotes the current text, links the official page and states when
  the archive fetched it. Judgments are never guessed: the public case-law database forbids bulk
  reuse, so they are flagged for manual checking until a lawful source exists.
- The model gets one tool, `check_citations`, on a `review` route (and on the project route for files
  in a matter folder), and is told to relay the verdicts as they are. `aroa citas FILE` does the same
  without any model.

## Consequences

- A wrong article number is caught in seconds with any model, and the lawyer sees the real text next to
  the verdict. What stays with the lawyer: whether the article, existing, supports the point.
- Norms the archive has not downloaded are reported as such with the command to fetch them; the checker
  never calls the network itself.
- Pattern extraction misses unusual phrasings; every miss is a citation the report does not list, never
  a wrong verdict. Extending the patterns is a test away.
