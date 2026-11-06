"""A plain-language account of each step of a turn, stored with the turn and shown in the UI and the CLI."""

TITLES = {
    "recall": "Recall",
    "route": "Route",
    "chat": "Reply",
    "retrieve": "Search documents",
    "grade": "Keep what helps",
    "answer_documents": "Answer from documents",
    "check_citations": "Check citations",
    "act": "Think",
    "tools": "Act",
    "verify": "Verify claims",
    "memorize": "Memorize",
}

MAX_ITEMS = 10


def entry(step: str, summary: str, items: list[str] | None = None) -> dict:
    return {"step": step, "title": TITLES[step], "summary": summary, "items": (items or [])[:MAX_ITEMS]}


def recalled(memories: dict, *, anonymized: bool = False) -> dict:
    facts, episodes = memories.get("facts", []), memories.get("episodes", [])
    privacy = "Personal data in the message was replaced by placeholders. " if anonymized else ""
    if not facts and not episodes:
        return entry("recall", privacy + "Nothing relevant in memory.")
    items = [f"fact · {f['text']}" for f in facts] + [f"episode {e['ts'][:10]} · {e['user'][:80]}" for e in episodes]
    return entry("recall", privacy + f"{len(facts)} facts and {len(episodes)} past episodes retrieved.", items)


def blocked(reason: str) -> dict:
    return entry("route", f"Blocked by the firm's policy: {reason}")


def routed(route: str, query: str, project: str) -> dict:
    where = f" in project {project!r}" if project else ""
    return entry("route", f"Route: {route}{where}. Query: «{query}»")


def retrieved(chunks, queries: list[str]) -> dict:
    titles = {c.title for c in chunks}
    return entry(
        "retrieve",
        f"{len(chunks)} fragments from {len(titles)} documents (hybrid search, {len(set(queries))} queries).",
        [c.label() for c in chunks],
    )


def graded(kept, total: int) -> dict:
    if not kept:
        return entry("grade", f"None of the {total} fragments helps.")
    return entry("grade", f"{len(kept)} of {total} fragments kept.", [c.label() for c in kept])


def answered(coverage: str, cited: int, attempt: int) -> dict:
    retry = "Second attempt after citation errors. " if attempt > 1 else ""
    return entry("answer_documents", f"{retry}Coverage {coverage}, {cited} citations.")


def citations_checked(errors: list[str]) -> dict:
    if not errors:
        return entry("check_citations", "Every quote appears verbatim in its fragment.")
    return entry("check_citations", "Citation problems:", errors)


def thought(tool_calls: list[dict], concluded: bool) -> dict:
    if concluded:
        return entry("act", "Replied without more tool calls.")
    return entry("act", f"Decided to call {len(tool_calls)} tool(s).", [c["name"] for c in tool_calls])


def nudged() -> dict:
    return entry("act", "Announced the work without using a tool; asked once to act.")


def acted(results: list[dict]) -> dict:
    items = [f"{'ok' if r['ok'] else 'FAILED'} · {r['tool']} {r['summary']}" for r in results]
    failed = sum(1 for r in results if not r["ok"])
    return entry("tools", f"{len(results)} tool call(s), {failed} failed.", items)


def verified(checked: int, failures: list[str], retrying: bool) -> dict:
    if checked == 0:
        return entry("verify", "The reply claims no actions.")
    if not failures:
        return entry("verify", f"All {checked} claims are backed by this turn's evidence.")
    what = "sending the turn back to fix them" if retrying else "the reply was corrected to say so"
    return entry("verify", f"{len(failures)} of {checked} claims unproven, {what}:", failures)


def memorized(stored: list[str], skipped: list[str]) -> dict:
    if not stored and not skipped:
        return entry("memorize", "Nothing durable to remember.")
    items = [f"stored · {s}" for s in stored] + [f"duplicate · {s}" for s in skipped]
    return entry("memorize", f"{len(stored)} new facts stored, {len(skipped)} already known.", items)


def chatted() -> dict:
    return entry("chat", "Replied from persona, memory and history.")
