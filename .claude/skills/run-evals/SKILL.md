---
name: run-evals
description: Run the whole-turn evaluation dataset against the configured models, read the results and update the README table. Use after any change to prompts, graph, memory, privacy or tools.
---

# Run the evals

- Full run: `uv run aroa eval` (background; minutes with a local model). One kind:
  `uv run aroa eval --only documents`. One case: `uv run aroa eval --only docs-lease`.
- The run uses its own data folder (`.data/eval`), re-ingests `data/samples`, and writes
  `evals/results/<timestamp>.json` with every reply and every check.
- Read failures from the JSON, not from the console: `checks` says which assertion failed,
  `reply` shows what the model said. Decide whether the case, the prompt or the code is wrong,
  in that order of suspicion.
- Never "fix" a failing case by loosening its checks unless the check was wrong.
- When numbers change, regenerate the table with `uv run python scripts/eval_table.py` and paste it into
  `README.md` → Evaluation.
