@AGENTS.md

## Claude Code specifics

- Skills in `.claude/skills/`: `add-tool`, `run-evals`, `review-turn`, `privacy-check`. Use them instead
  of improvising the procedure.
- Before saying a change works, run `uv run pytest -q`, and `uv run aroa eval --only <kind>` for the
  behaviour touched. Paste the real numbers.
- The eval run uses the local model configured in `.env`; it is slow (minutes). Run it in the background
  and keep working.
