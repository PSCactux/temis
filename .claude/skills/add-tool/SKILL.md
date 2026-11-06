---
name: add-tool
description: Add a tool to one of the assistant's routes (alarms, web, project), with its claim check, trace, tests and eval case. Use when asked to give the assistant a new capability.
---

# Add a tool

1. **Pick the route.** A tool belongs to exactly one of `alarms`, `web`, `project` (`src/aroa/tools/`).
   A new route is a bigger change: it needs a `Routing` value, a prompt line in `prompts.ROUTE`, an entry
   in `TOOL_ROUTES` and a branch in `tools_for`.
2. **Write the tool** in that module's `make_tools(ctx)` with the `@tool` decorator. Raise `ToolError`
   for anything that did not happen; never return a success message for a failure. Keep the docstring
   short: the model reads it.
3. **Pseudonymise what it brings in.** If the tool returns text from outside the conversation (a file, a
   page, a command), add its name to the list in `graph.py` → `tools()` that passes outputs through the
   anonymizer.
4. **Teach the verifier.** If the assistant can now claim a new kind of action, add the kind to
   `schemas.ClaimKind`, a line to `prompts.CLAIMS`, and a branch in `verify.check_claim` that passes
   only with matching successful evidence (and the artefact, when there is one).
5. **Test.** `tests/test_graph.py`: a scripted turn that calls the tool and is verified; a scripted turn
   where the model claims the action without calling it and gets sent back. `tests/test_verify.py` for
   the claim check.
6. **Eval.** One case in `evals/dataset.jsonl` of the right kind. Run `uv run aroa eval --only <kind>`.
7. Mention the tool in `docs/SPEC.md` → Tools, and in the README table if it is user-facing.
