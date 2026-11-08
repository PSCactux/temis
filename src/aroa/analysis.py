"""Long documents analysed whole: every stored fragment is read, findings are merged in code, and nothing
is reported without a verbatim quote that points to where it came from.

Map: the fast model reads one fragment at a time with the goal and returns findings, each with a quote.
Reduce (code): quotes that are not verbatim in their fragment are dropped, findings that say the same
thing from different places are merged and keep every location, and the result is numbered. An optional
overview is written by the large model from the numbered findings and is kept only if every citation
in it resolves. The budget is time, not context: a document of any length is a longer loop."""

from __future__ import annotations

import json
import re
import sqlite3
import time
import uuid
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field

from langchain_core.messages import HumanMessage, SystemMessage

from . import prompts
from .citations import cited_numbers, from_injection, normalize, quote_in, tidy_markers
from .memory.chunking import Chunk
from .schemas import Findings
from .telemetry.metrics import now_iso

KINDS = ("obligación", "derecho", "plazo", "importe", "penalización", "riesgo", "dato", "otro")
DEFAULT_GOAL = "Obligaciones, derechos, plazos, importes, penalizaciones y riesgos relevantes para el cliente."
SIMILAR_WORDS = 0.6  # Jaccard of content words above which two statements are the same finding


@dataclass
class Location:
    chunk_id: str
    label: str
    page: int | None = None


@dataclass
class Finding:
    n: int
    kind: str
    statement: str
    quote: str
    locations: list[Location] = field(default_factory=list)


@dataclass
class Analysis:
    source: str
    title: str
    goal: str
    chunks: int
    findings: list[Finding]
    dropped: int  # findings whose quote was not verbatim (or came from a planted instruction)
    merged: int  # findings folded into another because they said the same thing
    overview: str | None
    overview_note: str | None
    seconds: float

    def sources(self) -> list[dict]:
        """The same shape the documents route uses, so a [n] in the UI opens the fragment."""
        return [{"n": f.n, "label": f.locations[0].label, "file": self.source, "quote": f.quote} for f in self.findings]

    def as_dict(self) -> dict:
        return {**asdict(self), "sources": self.sources(), "report": format_report(self)}


def _words(text: str) -> set[str]:
    return {w for w in re.findall(r"\w+", normalize(text)) if len(w) > 3}


def same_finding(a: Finding, b: Finding) -> bool:
    if normalize(a.quote) == normalize(b.quote):
        return True
    wa, wb = _words(a.statement), _words(b.statement)
    return bool(wa and wb) and len(wa & wb) / len(wa | wb) >= SIMILAR_WORDS


def merge(found: list[tuple[Finding, Chunk]]) -> tuple[list[Finding], int]:
    """Fold findings that say the same thing into one, keeping every location. Returns (findings, merged)."""
    kept: list[Finding] = []
    merged = 0
    for finding, chunk in found:
        location = Location(chunk.id, chunk.label(), chunk.page)
        twin = next((k for k in kept if same_finding(k, finding)), None)
        if twin is None:
            finding.locations = [location]
            kept.append(finding)
        else:
            merged += 1
            if all(loc.chunk_id != location.chunk_id for loc in twin.locations):
                twin.locations.append(location)
    for i, f in enumerate(kept, start=1):
        f.n = i
    return kept, merged


def analyze(
    *,
    memory,
    fast_llm,
    llm=None,
    source: str,
    goal: str = DEFAULT_GOAL,
    progress=None,
    lock=None,
) -> Analysis:
    """Read every fragment of `source`, extract findings for `goal`, verify and merge them in code."""
    started = time.perf_counter()
    chunks: list[Chunk] = memory.document_chunks(source)
    if not chunks:
        raise ValueError(f"no indexed document named {source!r}")
    mapper = fast_llm.with_structured_output(Findings, method="json_schema")
    guard = lock if lock is not None else nullcontext()
    found: list[tuple[Finding, Chunk]] = []
    dropped = 0
    for i, chunk in enumerate(chunks, start=1):
        with guard:
            result: Findings = mapper.invoke(
                [
                    SystemMessage(prompts.ANALYZE_MAP.format(goal=goal, kinds=", ".join(KINDS))),
                    HumanMessage(prompts.data_block("fragment", chunk.text, id=chunk.id, source=chunk.label())),
                ]
            )
        for item in result.findings:
            quote = item.quote.strip()
            if not quote or not quote_in(quote, chunk.text) or from_injection(quote, chunk.text):
                dropped += 1  # no verbatim quote, no finding
                continue
            kind = item.kind if item.kind in KINDS else "otro"
            found.append((Finding(0, kind, item.statement.strip(), quote), chunk))
        if progress:
            progress(i, len(chunks))
    findings, merged = merge(found)
    overview, note = None, None
    if findings and llm is not None:
        overview, note = write_overview(llm, goal, findings, guard)
    title = chunks[0].title
    return Analysis(
        source,
        title,
        goal,
        len(chunks),
        findings,
        dropped,
        merged,
        overview,
        note,
        round(time.perf_counter() - started, 1),
    )


def write_overview(llm, goal: str, findings: list[Finding], guard) -> tuple[str | None, str | None]:
    """A short overview that cites findings as [n]; kept only if every citation exists and every sentence has one."""
    listed = "\n".join(f"[{f.n}] ({f.kind}) {f.statement}" for f in findings)
    with guard:
        reply = llm.invoke(
            [
                SystemMessage(prompts.ANALYZE_OVERVIEW.format(goal=goal)),
                HumanMessage(prompts.data_block("findings", listed)),
            ]
        )
    text = tidy_markers(str(reply.content).strip())
    valid = {f.n for f in findings}
    cited = cited_numbers(text)
    if not cited or not cited <= valid:
        return (
            None,
            "la visión general citaba hallazgos inexistentes y se ha descartado"
            if cited
            else "la visión general no citaba hallazgos y se ha descartado",
        )
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    uncited = [s for s in sentences if not cited_numbers(s)]
    if uncited:
        text = " ".join(s for s in sentences if cited_numbers(s))
        return text or None, f"{len(uncited)} frase(s) sin cita eliminadas de la visión general"
    return text, None


def format_report(a: Analysis) -> str:
    lines = [
        f"# Análisis de «{a.title}»",
        "",
        f"Objetivo: {a.goal}",
        "",
        f"Fragmentos leídos: {a.chunks} · hallazgos: {len(a.findings)} · repetidos fundidos: {a.merged} · sin cita verificable (descartados): {a.dropped}",
        "",
    ]
    if a.overview:
        lines += ["## Visión general", "", a.overview, ""]
    if a.overview_note:
        lines += [f"_{a.overview_note}_", ""]
    for kind in KINDS:
        group = [f for f in a.findings if f.kind == kind]
        if not group:
            continue
        lines += [f"## {kind.capitalize()}", ""]
        for f in group:
            where = "; ".join(loc.label + (f", p. {loc.page}" if loc.page else "") for loc in f.locations)
            lines.append(f"- [{f.n}] {f.statement}")
            lines.append(f"  > «{f.quote}» — {where}")
        lines.append("")
    if not a.findings:
        lines.append("Ningún hallazgo con cita verificable para este objetivo.")
    return "\n".join(lines).rstrip() + "\n"


# ── jobs: one row per analysis, progress visible while it runs ─────────────
def create_job(conn: sqlite3.Connection, kind: str, source: str, goal: str) -> str:
    job_id = uuid.uuid4().hex[:12]
    conn.execute(
        "INSERT INTO jobs (job_id, kind, source, goal, status, done, total, created_ts, updated_ts) VALUES (?,?,?,?,?,?,?,?,?)",
        (job_id, kind, source, goal, "queued", 0, 0, now_iso(), now_iso()),
    )
    conn.commit()
    return job_id


def update_job(conn: sqlite3.Connection, job_id: str, **fields) -> None:
    if "result" in fields and not isinstance(fields["result"], str):
        fields["result"] = json.dumps(fields["result"], ensure_ascii=False)
    fields["updated_ts"] = now_iso()
    sets = ", ".join(f"{k} = ?" for k in fields)
    conn.execute(f"UPDATE jobs SET {sets} WHERE job_id = ?", (*fields.values(), job_id))
    conn.commit()


def get_job(conn: sqlite3.Connection, job_id: str) -> dict | None:
    row = conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
    if row is None:
        return None
    data = dict(row)
    if data.get("result"):
        data["result"] = json.loads(data["result"])
    return data


def list_jobs(conn: sqlite3.Connection, source: str | None = None, limit: int = 20) -> list[dict]:
    query, params = (
        "SELECT job_id, kind, source, goal, status, done, total, created_ts, updated_ts, error FROM jobs",
        (),
    )
    if source:
        query, params = query + " WHERE source = ?", (source,)
    rows = conn.execute(query + " ORDER BY created_ts DESC LIMIT ?", (*params, limit)).fetchall()
    return [dict(r) for r in rows]


def run_job(conn: sqlite3.Connection, job_id: str, *, memory, fast_llm, llm, lock=None, reveal=None) -> None:
    """Execute a queued analysis job, recording progress and the result (revealed for the user) or the error."""
    job = get_job(conn, job_id)
    if job is None:
        return
    update_job(conn, job_id, status="running")
    try:
        result = analyze(
            memory=memory,
            fast_llm=fast_llm,
            llm=llm,
            source=job["source"],
            goal=job["goal"] or DEFAULT_GOAL,
            lock=lock,
            progress=lambda done, total: update_job(conn, job_id, done=done, total=total),
        )
        data = result.as_dict()
        if reveal:
            data["report"] = reveal(data["report"])
            for f in data["findings"]:
                f["statement"], f["quote"] = reveal(f["statement"]), reveal(f["quote"])
            for s in data["sources"]:
                s["quote"] = reveal(s["quote"])
            if data.get("overview"):
                data["overview"] = reveal(data["overview"])
        update_job(conn, job_id, status="done", result=data, done=result.chunks, total=result.chunks)
    except Exception as exc:  # noqa: BLE001 — the job records its failure; the server keeps serving
        update_job(conn, job_id, status="failed", error=f"{type(exc).__name__}: {exc}")
