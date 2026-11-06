---
name: review-turn
description: Diagnose one turn of the assistant from its path, metrics and audit (why it routed there, what it recalled, what it claimed, what failed). Use when a reply was wrong, slow or unverified.
---

# Review a turn

1. Get the turn id (printed by the CLI after each reply; shown by the web UI's "path" button).
2. Read it: `curl -s localhost:8000/api/turns/<id> | jq` or, without the server,
   `sqlite3 .data/aroa.sqlite "select * from turns where turn_id='<id>'"` and the `trace` column.
3. Walk the path in order: Recall (what memory came in) → Route (which route and query, or blocked) →
   the route's steps → Verify (claims and failures) → Memorize (what was stored).
4. `audit.tool_calls` has every tool with arguments and outcome; `audit.memory_writes` every fact.
5. With tracing on, the same turn id is in the span metadata of the collector (Phoenix, Langfuse...).
6. Decide where the fix belongs: routing → `prompts.ROUTE`/`schemas.Routing`; recall → `memory/store.py`;
   a lie → `verify.py` (a missing claim kind) or the tool; latency → the per-node seconds in `nodes`.
