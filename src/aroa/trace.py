"""A plain-language account of each step of a turn, stored with the turn and shown in the UI and the CLI."""

from .i18n import t

MAX_ITEMS = 10


class Steps:
    """Builds the entries of a turn's path in the configured language."""

    def __init__(self, lang: str = "en") -> None:
        self.lang = lang

    def _(self, key: str, **values) -> str:
        return t(self.lang, key, **values)

    def entry(self, step: str, summary: str, items: list[str] | None = None) -> dict:
        return {"step": step, "title": self._(step), "summary": summary, "items": (items or [])[:MAX_ITEMS]}

    def recalled(self, memories: dict, *, anonymized: bool = False) -> dict:
        facts, episodes = memories.get("facts", []), memories.get("episodes", [])
        privacy = self._("recall_privacy") if anonymized else ""
        if not facts and not episodes:
            return self.entry("recall", privacy + self._("recall_none"))
        items = [f"{self._('fact')} · {f['text']}" for f in facts] + [
            f"{self._('episode')} {e['ts'][:10]} · {e['user'][:80]}" for e in episodes
        ]
        return self.entry("recall", privacy + self._("recall_some", facts=len(facts), episodes=len(episodes)), items)

    def blocked(self, reason: str) -> dict:
        return self.entry("route", self._("blocked", reason=reason))

    def routed(self, route: str, query: str, project: str) -> dict:
        where = self._("in_project", project=project) if project else ""
        return self.entry("route", self._("routed", route=route, where=where, query=query))

    def retrieved(self, chunks, queries: list[str]) -> dict:
        titles = {c.title for c in chunks}
        summary = self._("retrieved", chunks=len(chunks), titles=len(titles), queries=len(set(queries)))
        return self.entry("retrieve", summary, [c.label() for c in chunks])

    def graded(self, kept, total: int) -> dict:
        if not kept:
            return self.entry("grade", self._("graded_none", total=total))
        return self.entry("grade", self._("graded", kept=len(kept), total=total), [c.label() for c in kept])

    def answered(self, coverage: str, cited: int, attempt: int) -> dict:
        retry = self._("answered_retry") if attempt > 1 else ""
        return self.entry("answer_documents", self._("answered", retry=retry, coverage=coverage, cited=cited))

    def citations_checked(self, errors: list[str]) -> dict:
        if not errors:
            return self.entry("check_citations", self._("citations_ok"))
        return self.entry("check_citations", self._("citations_bad"), errors)

    def thought(self, tool_calls: list[dict], concluded: bool) -> dict:
        if concluded:
            return self.entry("act", self._("concluded"))
        return self.entry("act", self._("thought", n=len(tool_calls)), [c["name"] for c in tool_calls])

    def acted(self, results: list[dict]) -> dict:
        items = [f"{self._('ok') if r['ok'] else self._('failed')} · {r['tool']} {r['summary']}" for r in results]
        failed = sum(1 for r in results if not r["ok"])
        return self.entry("tools", self._("acted", n=len(results), failed=failed), items)

    def verified(self, checked: int, failures: list[str], retrying: bool) -> dict:
        if checked == 0:
            return self.entry("verify", self._("verify_none"))
        if not failures:
            return self.entry("verify", self._("verify_ok", checked=checked))
        what = self._("verify_retry") if retrying else self._("verify_marked")
        return self.entry("verify", self._("verify_bad", failed=len(failures), checked=checked, what=what), failures)

    def memorized(self, stored: list[str], skipped: list[str]) -> dict:
        if not stored and not skipped:
            return self.entry("memorize", self._("memorized_none"))
        items = [f"{self._('stored')} · {s}" for s in stored] + [f"{self._('duplicate')} · {s}" for s in skipped]
        return self.entry("memorize", self._("memorized", stored=len(stored), skipped=len(skipped)), items)

    def chatted(self, cited: int = 0) -> dict:
        if cited:
            return self.entry("chat", self._("chatted_cited", n=cited))
        return self.entry("chat", self._("chatted"))
