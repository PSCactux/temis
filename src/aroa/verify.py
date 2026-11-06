"""Claims the reply makes about actions, checked against the turn's evidence in code.

The model extracts the claims (a short structured call); nothing here trusts the model's
word for whether the action happened. A claim passes only if a successful tool call of the
right kind with a matching target is in the evidence, and, for files and alarms, if the
artefact exists.
"""

import re
from pathlib import Path
from urllib.parse import urlsplit

from .schemas import Claim
from .tools import alarms

WRITE_TOOLS = {"write_file", "edit_file"}
READ_TOOLS = {"read_file", "grep", "list_files"}


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower().strip("`'\"«»"))


def _path_matches(target: str, path: str) -> bool:
    target, path = _norm(target).lstrip("./"), _norm(path).lstrip("./")
    return (
        target == path
        or target.endswith("/" + path)
        or path.endswith("/" + target)
        or Path(target).name == Path(path).name
    )


def _host(url: str) -> str:
    return (urlsplit(url).hostname or "").lower().removeprefix("www.")


def _words(text: str) -> set[str]:
    return {w for w in re.findall(r"\w+", _norm(text)) if len(w) > 3}


def check_claim(claim: Claim, evidence: list[dict], *, project_root: Path | None, conn, thread_id: str) -> str | None:
    """None when the claim is backed by evidence, else a sentence saying what is missing."""
    ok_calls = [e for e in evidence if e["ok"]]
    kind, target = claim.kind, claim.target

    if kind == "file_written":
        hits = [e for e in ok_calls if e["tool"] in WRITE_TOOLS and _path_matches(target, e["args"].get("path", ""))]
        if not hits:
            return f"says it wrote {target!r}, but no successful write_file/edit_file call touched that file"
        if project_root and not any((project_root / e["args"]["path"]).is_file() for e in hits):
            return f"says it wrote {target!r}, but the file does not exist"
        return None
    if kind == "file_read":
        if any(e["tool"] in READ_TOOLS and _path_matches(target, e["args"].get("path", target)) for e in ok_calls):
            return None
        return f"says it read {target!r}, but no successful read of that file happened"
    if kind == "command_run":
        for e in ok_calls:
            if e["tool"] != "run_command":
                continue
            command = _norm(e["args"].get("command", ""))
            if _norm(target) in command or command in _norm(target) or _words(target) & _words(command):
                return None
        return f"says it ran {target!r}, but no such command succeeded"
    if kind == "url_read":
        pages = [e for e in ok_calls if e["tool"] == "read_page"]
        if not _host(target) and pages:
            return None  # the claim names a title, not a URL: some page was read
        if any(_host(e["args"].get("url", "")) == _host(target) for e in pages):
            return None
        if _host(target) and any(e["tool"] == "search_web" for e in ok_calls):
            return f"says it read {target!r}, but it only searched; the page was never opened"
        return f"says it read {target!r}, but no page from that site was fetched"
    if kind == "search_done":
        return (
            None if any(e["tool"] == "search_web" for e in ok_calls) else "says it searched the web, but no search ran"
        )
    if kind == "alarm_created":
        created = [e for e in ok_calls if e["tool"] == "create_alarm"]
        if not created:
            return f"says it set an alarm ({target!r}), but create_alarm never succeeded"
        pending = alarms.pending(conn, thread_id)
        if not pending:
            return f"says it set an alarm ({target!r}), but there is no pending alarm in the database"
        return None
    if kind == "alarm_deleted":
        return None if any(e["tool"] == "delete_alarm" for e in ok_calls) else "says it deleted an alarm, but none was"
    # other: an action we cannot check specifically; at least some tool must have succeeded this turn
    return None if ok_calls else f"says it did {target!r}, but no tool ran successfully this turn"


def check_claims(claims: list[Claim], evidence: list[dict], **context) -> tuple[int, list[str]]:
    failures = [f for c in claims if (f := check_claim(c, evidence, **context))]
    return len(claims), failures
