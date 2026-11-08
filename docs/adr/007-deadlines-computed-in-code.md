# ADR 007 — Procedural deadlines are computed in code

**Status:** accepted · 2026-11

## Context

"When does the term end?" is the most frequent question in a firm and the most expensive one to get
wrong: a missed deadline is malpractice. Language models count days badly, forget that August does not
count in civil proceedings, that a month runs date to date, that a term ending on a Saturday moves, or
that a filing may still be made until 15:00 of the next working day. Spanish courts already sanction
lawyers who trusted an assistant without checking.

## Decision

- `aroa.deadlines` computes the due date deterministically: days (working or natural), months and
  years, per jurisdiction (civil, contencioso, social, penal, administrativo), with the rules of the
  LEC, LOPJ, LJCA, LRJS and Ley 39/2015 encoded as data, each with its article.
- Holidays are data, never code: `data/calendar/holidays.json` holds the national calendar by year and
  regional examples; the firm sets `COURT_REGION` and `LOCAL_HOLIDAYS`. A missing year or region is
  said out loud in the result, never silently ignored.
- The model gets one tool, `compute_deadline`, on the alarms route, and is told never to count days
  itself. The tool's output lists the due date, the grace limit, every excluded day with its reason and
  the rules applied, so the lawyer can audit the arithmetic in seconds.
- Evals of kind `deadlines` assert that a tool call happened and that the right date is in the reply.

## Consequences

- The date is right whenever the inputs and the calendar are right, with any model. What remains the
  lawyer's judgement is which rule governs (urgent proceeding, natural days, the phase of a criminal
  case), and the result says so where it matters.
- The calendar must be kept current each year; the result flags missing years. Local holidays are the
  firm's responsibility and the result reminds them until they are configured.
- Not covered: terms that run from events other than a notification (publication in a gazette, the
  end of another term), and notifications through LexNET whose effective date depends on access time.
