"""Offline evaluation of whole turns against evals/dataset.jsonl, in a separate data folder.

Case kinds and what passes:
- identity   the reply contains the assistant's name
- memory     facts from `setup` turns are recalled in a new thread of the same matter
- wall       facts from another matter are NOT recalled
- privacy    identifiers in the message never reach the model, the stores or the checkpoint; the reply is revealed
- policy     the request is blocked
- documents  cites the expected section, includes the key facts, abstains when it should (corrective RAG)
- web        at least one tool was used and no claim failed (needs internet)
- alarms     a pending alarm exists after the turn and the claim is verified
- review     the verdict on each citation comes from the checker (a tool call happened) and is the right one
- analysis   a long document analysed whole: planted facts found, repeats merged, every finding quoted
- deadlines  the due date in the reply comes from the deadline tool (a tool call happened) and is the right one
- project    the file the assistant says it created exists and no claim failed
"""

import json
import shutil
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from .config import Settings
from .memory.chunking import Chunk
from .privacy import contains_personal_data
from .tools import alarms


@dataclass
class Case:
    id: str
    kind: str
    question: str
    matter: str = ""
    setup: list[str] = field(default_factory=list)  # earlier turns, in their own thread of the same matter
    answerable: bool = True
    expected: list[str] = field(default_factory=list)
    must_include: list[str] = field(default_factory=list)
    must_not_include: list[str] = field(default_factory=list)
    files: dict[str, str] = field(default_factory=dict)  # project files to create before the turn
    expect_file: str = ""
    documents: list[str] = field(default_factory=list)  # sources the turn is confined to
    ingest: str = ""  # analysis cases: a file (relative to the repo) indexed for this case only
    goal: str = ""  # analysis cases: what the analysis looks for


def load_cases(path: Path) -> list[Case]:
    return [Case(**json.loads(line)) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def matches(chunk: dict, expected: str) -> bool:
    doc_id, _, section = expected.partition(":")
    chunk = Chunk(**chunk)
    return chunk.doc_id == doc_id and chunk.section.startswith(section)


def _fresh(settings: Settings) -> None:
    for path in (settings.data_dir, settings.docs_dir):
        shutil.rmtree(path, ignore_errors=True)


def run_evals(settings: Settings, dataset: Path, *, only: str | None = None, on_case: Callable | None = None) -> dict:
    from .app import Assistant

    _fresh(settings)
    cases = [c for c in load_cases(dataset) if not only or c.kind == only or c.id.startswith(only)]
    assistant = Assistant(settings)
    assistant.ingest()
    rows = []
    for case in cases:
        started = time.perf_counter()
        try:
            checks = evaluate_case(assistant, case)
            reply = checks.pop("_reply")
        except Exception as exc:  # one broken case must not end the run
            checks, reply = {"error": False}, f"{type(exc).__name__}: {exc}"
        row = {
            "id": case.id,
            "kind": case.kind,
            "passed": all(checks.values()),
            "checks": checks,
            "reply": reply,
            "seconds": round(time.perf_counter() - started, 1),
        }
        rows.append(row)
        if on_case:
            on_case(row)
    by_kind: dict[str, dict] = {}
    for kind in sorted({r["kind"] for r in rows}):
        group = [r for r in rows if r["kind"] == kind]
        seconds = sorted(r["seconds"] for r in group)
        by_kind[kind] = {
            "cases": len(group),
            "passed": sum(r["passed"] for r in group),
            "median_seconds": seconds[len(seconds) // 2],
        }
    return {
        "model": settings.llm_model,
        "fast_model": settings.llm_fast_model,
        "privacy_mode": settings.privacy_mode,
        "cases": len(rows),
        "passed": sum(r["passed"] for r in rows),
        "pass_rate": round(sum(r["passed"] for r in rows) / max(1, len(rows)), 3),
        "by_kind": by_kind,
        "rows": rows,
    }


def evaluate_case(assistant, case: Case) -> dict:
    matter = case.matter
    if case.setup:
        setup_thread = f"eval-setup-{case.id}-{uuid.uuid4().hex[:6]}"
        for text in case.setup:
            assistant.ask(setup_thread, text, matter=matter, source="eval")
    if case.files:
        root = assistant.settings.projects_dir / matter
        root.mkdir(parents=True, exist_ok=True)
        for name, content in case.files.items():
            (root / name).write_text(content, encoding="utf-8")
    thread = f"eval-{case.id}-{uuid.uuid4().hex[:6]}"
    if case.kind == "analysis":
        return evaluate_analysis(assistant, case)
    result = assistant.ask(thread, case.question, matter=matter, source="eval", documents=case.documents)
    reply, lowered = result.reply, result.reply.lower()
    checks: dict = {"_reply": reply, "no_error": result.kind != "error"}
    if case.must_include:
        checks["includes"] = all(any(t.lower() in lowered for t in term.split("|")) for term in case.must_include)
    if case.must_not_include:
        checks["excludes"] = not any(term.lower() in lowered for term in case.must_not_include)

    if case.kind == "identity":
        checks["name"] = assistant.settings.assistant_name.lower() in lowered
    elif case.kind == "policy":
        checks["blocked"] = result.kind == "blocked"
    elif case.kind == "privacy":
        state = assistant.graph.get_state({"configurable": {"thread_id": thread}}).values
        stored = " ".join(getattr(m, "content", "") for m in state.get("messages", []) if not isinstance(m, str))
        checks["checkpoint_clean"] = contains_personal_data(stored) == []
        checks["memory_clean"] = not any(
            contains_personal_data(f["text"]) for f in assistant.memory.list_facts(limit=10000)
        )
        checks["revealed"] = all(any(t in reply for t in term.split("|")) for term in case.must_include)
    elif case.kind == "documents":
        answered = result.kind == "documents"
        checks["behaviour"] = answered == case.answerable
        if case.answerable and answered:
            sources = assistant.graph.get_state({"configurable": {"thread_id": thread}}).values.get("sources", [])
            checks["cites_expected"] = any(matches(c, e) for c in sources for e in case.expected)
            checks["grounded"] = result.extras.get("verified", False)
    elif case.kind == "web":
        checks["used_tools"] = result.metrics["tool_calls"] >= 1
        checks["claims_verified"] = result.metrics["claims_failed"] == 0
    elif case.kind == "review":
        checks["used_tools"] = result.metrics["tool_calls"] >= 1  # verdicts must come from the checker
        checks["claims_verified"] = result.metrics["claims_failed"] == 0
    elif case.kind == "deadlines":
        checks["used_tools"] = result.metrics["tool_calls"] >= 1  # the date must come from compute_deadline
        checks["claims_verified"] = result.metrics["claims_failed"] == 0
    elif case.kind == "alarms":
        checks["alarm_exists"] = bool(alarms.pending(assistant.conn, thread))
        checks["claims_verified"] = result.metrics["claims_failed"] == 0
    elif case.kind == "project":
        checks["file_exists"] = (assistant.settings.projects_dir / matter / case.expect_file).is_file()
        checks["claims_verified"] = result.metrics["claims_failed"] == 0
    return checks


def evaluate_analysis(assistant, case: Case) -> dict:
    """Index the case's long document, analyse it whole, check the planted facts, then drop it again."""
    import shutil

    from .analysis import DEFAULT_GOAL, analyze

    src = Path(case.ingest)
    target = assistant.settings.docs_dir / src.name
    shutil.copy(src, target)
    assistant.ingest()
    checks: dict = {}
    try:
        result = analyze(
            memory=assistant.memory,
            fast_llm=assistant.fast_llm,
            llm=None,
            source=src.name,
            goal=case.goal or DEFAULT_GOAL,
        )
        report = assistant.reveal(result.as_dict()["report"])
        lowered = report.lower()
        checks["no_error"] = True
        checks["includes"] = all(any(t.lower() in lowered for t in term.split("|")) for term in case.must_include)
        checks["excludes"] = not any(term.lower() in lowered for term in case.must_not_include)
        checks["all_quoted"] = all(f.quote and f.locations for f in result.findings)
        checks["repeats_merged"] = result.merged >= 1  # the sample repeats two facts on purpose
        checks["_reply"] = (
            f"{len(result.findings)} findings, {result.merged} merged, {result.dropped} dropped, "
            f"{result.chunks} fragments, {result.seconds}s"
        )
    except Exception as exc:  # noqa: BLE001
        checks = {"no_error": False, "_reply": f"{type(exc).__name__}: {exc}"}
    finally:
        target.unlink(missing_ok=True)
        assistant.ingest()
    return checks
