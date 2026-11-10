"""HTTP API and web app: streaming turns over SSE, thread history, the path and audit of every turn,
feedback, Prometheus metrics, the daily report, and the alarm scheduler."""

import asyncio
import json
import queue
import tempfile
import threading
from contextlib import asynccontextmanager
from functools import partial
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, PlainTextResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from . import analysis, clippings, documents, lawcheck, procedures, prompts, review
from .app import Assistant
from .config import get_settings
from .i18n import t as say
from .memory.loaders import EXTENSIONS, read_text
from .models import make_chat_model
from .telemetry.audit import turn_audit
from .telemetry.metrics import dashboard, now_iso, prometheus_text, report
from .tools import alarms

WEB = Path(__file__).parent / "web"
ALARM_TICK_SECONDS = 20
MAX_UPLOAD_BYTES = 25 * 1024 * 1024
MAX_ATTACHMENT_CHARS = 12000


class Attachment(BaseModel):
    name: str
    text: str


class Message(BaseModel):
    text: str
    matter: str = ""
    attachments: list[Attachment] = []  # read for this turn only; indexing is a separate, explicit upload
    documents: list[str] = []  # sources this turn is confined to; empty means every indexed document


class Feedback(BaseModel):
    score: int  # 1 or -1
    comment: str = ""


class AnalysisRequest(BaseModel):
    goal: str = ""


class Clipping(BaseModel):
    source: str
    text: str
    note: str = ""
    matter: str = ""


class DocumentSave(BaseModel):
    html: str
    name: str = ""  # for a new document; ignored when saving over an existing .html


class ExportRequest(BaseModel):
    html: str
    name: str = "documento"


class ProcedureDraft(BaseModel):
    data: dict[str, str] = {}
    matter: str = ""


class ProcedureField(BaseModel):
    name: str = ""
    hint: str = ""
    long: bool = False


class ProcedureSave(BaseModel):
    """A procedure as the form sends it; the server checks it before anything is written."""

    name: str = ""
    description: str = ""
    template: str = ""  # a document of the library
    fields: list[ProcedureField] = []
    instructions: str = ""
    checklist: list[str] = []


class ReviewRequest(BaseModel):
    procedure: str = ""  # slug of a procedure whose checklist the review also checks
    name: str = "borrador"
    text: str = ""  # the draft; or
    source: str = ""  # an indexed document reviewed as a draft (excluded from the documents it is checked against)
    documents: list[str] = []  # documents the quotes are checked against; empty means every indexed document
    legal: bool = True  # also let the model look for legal issues (slower)


class Runtime:
    """One assistant, one lock: a local model answers one turn at a time, alarms included."""

    def __init__(self) -> None:
        self.assistant = Assistant(get_settings())
        self.lock = threading.Lock()
        self.jobs: queue.Queue[str] = queue.Queue()
        threading.Thread(target=self._work, daemon=True, name="aroa-jobs").start()

    def enqueue_analysis(self, source: str, goal: str) -> str:
        job_id = analysis.create_job(self.assistant.conn, "analysis", source, goal or analysis.DEFAULT_GOAL)
        self.jobs.put(job_id)
        return job_id

    def enqueue_review(self, body: ReviewRequest, text: str) -> str:
        spec = {"text": text, "documents": body.documents, "exclude": body.source or None, "legal": body.legal}
        if body.procedure:
            proc = procedures.find(
                self.assistant.settings.procedures_dir, body.procedure, self.assistant.settings.procedures_samples_dir
            )
            if proc is None:
                raise HTTPException(404, "unknown procedure")
            spec["checklist"] = proc.checklist
            spec["procedure"] = proc.slug
        name = body.name or body.source or "borrador"
        job_id = analysis.create_job(self.assistant.conn, "review", name, json.dumps(spec))
        self.jobs.put(job_id)
        return job_id

    def enqueue_draft(self, proc: procedures.Procedure, body: ProcedureDraft) -> str:
        spec = {"slug": proc.slug, "data": body.data, "matter": body.matter}
        job_id = analysis.create_job(self.assistant.conn, "draft", proc.name, json.dumps(spec, ensure_ascii=False))
        self.jobs.put(job_id)
        return job_id

    def _run_draft(self, job_id: str) -> None:
        """A procedure followed: data filled in code, marked blocks written by the model, data checked back."""
        a, s = self.assistant, self.assistant.settings
        job = analysis.get_job(a.conn, job_id)
        analysis.update_job(a.conn, job_id, status="running")
        log = analysis.JobLog(a.conn, job_id)
        try:
            spec = json.loads(job["goal"] or "{}")
            proc = procedures.find(s.procedures_dir, spec.get("slug", ""), s.procedures_samples_dir)
            if proc is None:
                raise ValueError("unknown procedure")
            template = procedure_template(proc)
            if template is None:
                raise ValueError(f"template {proc.template_ref} is not in the documents folder")
            if proc.template_ref:
                log(f"Plantilla «{proc.template_ref}» leída de la biblioteca")
            data = {k: v for k, v in (spec.get("data") or {}).items()}
            if a.anonymizer:  # the model sees placeholders, the firm sees names: same rule as everywhere
                with self.lock:
                    data = {k: a.anonymizer.anonymize(v) for k, v in data.items()}
                template = a.anonymizer.anonymize(template, names=False)
            result = procedures.draft(proc, data, llm=a.llm, template_text=template, guard=self.lock, log=log)
            result["text"] = a.reveal(result["text"])
            result["html"] = documents.text_to_html(result["text"])
            result["name"] = f"{proc.slug}-{spec.get('matter') or 'borrador'}"
            result["procedure"] = proc.slug
            analysis.update_job(a.conn, job_id, status="done", result=result, done=1, total=1)
        except Exception as exc:  # noqa: BLE001 — the job records its failure; the server keeps serving
            analysis.update_job(a.conn, job_id, status="failed", error=f"{type(exc).__name__}: {exc}")

    def _work(self) -> None:
        """One job at a time; each model call takes the turn lock, so conversations keep flowing between them."""
        a = self.assistant
        while True:
            job_id = self.jobs.get()
            job = analysis.get_job(a.conn, job_id)
            if job and job["kind"] == "draft":
                self._run_draft(job_id)
                continue
            if job and job["kind"] == "review":
                archive = a.settings.boe_archive_dir
                review.run_review_job(
                    a.conn,
                    job_id,
                    memory=a.memory,
                    law_source_factory=partial(lawcheck.BoeArchive, archive) if archive else None,
                    aliases=lawcheck.Aliases.load(a.settings.laws_path),
                    llm=a.llm,
                    lock=self.lock,
                    anonymize=a.anonymizer.anonymize if a.anonymizer else None,
                    reveal=a.reveal,
                )
                continue
            analysis.run_job(
                a.conn, job_id, memory=a.memory, fast_llm=a.fast_llm, llm=a.llm, lock=self.lock, reveal=a.reveal
            )

    def touch_thread(self, thread_id: str, matter: str, title: str) -> None:
        conn = self.assistant.conn
        row = conn.execute("SELECT matter FROM threads WHERE thread_id = ?", (thread_id,)).fetchone()
        if row is None:
            conn.execute(
                "INSERT INTO threads (thread_id, matter, title, created_ts, updated_ts) VALUES (?,?,?,?,?)",
                (thread_id, matter, title[:80], now_iso(), now_iso()),
            )
        else:
            conn.execute("UPDATE threads SET updated_ts = ? WHERE thread_id = ?", (now_iso(), thread_id))
        conn.commit()

    def matter_of(self, thread_id: str) -> str:
        row = self.assistant.conn.execute("SELECT matter FROM threads WHERE thread_id = ?", (thread_id,)).fetchone()
        return row["matter"] if row else ""

    def ingest(self) -> dict:
        with self.lock:
            return self.assistant.ingest()

    def fire_due_alarms(self) -> int:
        fired = 0
        for alarm in alarms.due_now(self.assistant.conn):
            alarms.mark_fired(self.assistant.conn, alarm["id"])
            text = f"[Alarm due now] {alarm['text']}"
            with self.lock:
                self.assistant.ask(alarm["thread_id"], text, source="alarm", matter=self.matter_of(alarm["thread_id"]))
            self.touch_thread(alarm["thread_id"], self.matter_of(alarm["thread_id"]), alarm["text"])
            fired += 1
        return fired


runtime: Runtime | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global runtime
    runtime = Runtime()
    task = asyncio.create_task(_alarm_loop())
    yield
    task.cancel()


async def _alarm_loop() -> None:
    while True:
        await asyncio.sleep(ALARM_TICK_SECONDS)
        try:
            await run_in_threadpool(runtime.fire_due_alarms)
        except Exception as exc:  # noqa: BLE001 — a failed alarm turn is logged as a turn error, the loop goes on
            print(f"alarm loop: {type(exc).__name__}: {exc}")


app = FastAPI(title="Aroa Assistant", lifespan=lifespan)
app.mount("/vendor", StaticFiles(directory=WEB / "vendor"), name="vendor")


@app.get("/")
def index():
    return FileResponse(WEB / "index.html")


@app.get("/api/health")
def health():
    a = runtime.assistant
    return {
        "ok": True,
        "name": a.settings.assistant_name,
        "model": a.settings.llm_model,
        "privacy_mode": a.settings.privacy_mode,
        "traced": a.traced,
        "language": a.settings.assistant_language,
        "memory": a.memory.stats(),
    }


@app.get("/api/threads")
def threads():
    rows = runtime.assistant.conn.execute("SELECT * FROM threads ORDER BY updated_ts DESC LIMIT 100").fetchall()
    return [dict(r) for r in rows]


@app.get("/api/threads/{thread_id}")
def thread(thread_id: str):
    a = runtime.assistant
    return {
        "thread_id": thread_id,
        "matter": runtime.matter_of(thread_id),
        "messages": a.history(thread_id),
        "waiting": a.waiting_for(thread_id),
    }


@app.post("/api/threads/{thread_id}/messages")
async def send(thread_id: str, message: Message):
    """Server-sent events: token, node, done. The thread resumes if it was waiting for an answer."""
    runtime.touch_thread(thread_id, message.matter, message.text)
    matter = message.matter or runtime.matter_of(thread_id)
    attached = "".join(
        "\n\n" + prompts.data_block("attachment", a.text[:MAX_ATTACHMENT_CHARS], name=a.name)
        for a in message.attachments
    )
    text = message.text + attached
    queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def produce() -> None:
        with runtime.lock:
            a = runtime.assistant
            for kind, item in a.stream(thread_id, text, matter=matter, documents=message.documents):
                payload = item.__dict__ if kind == "done" else item
                if kind == "node":
                    payload = {
                        **item,
                        "summary": a.reveal(item.get("summary", "")),
                        "items": [a.reveal(i) for i in item.get("items", [])],
                    }
                loop.call_soon_threadsafe(queue.put_nowait, (kind, payload))
        loop.call_soon_threadsafe(queue.put_nowait, None)

    threading.Thread(target=produce, daemon=True).start()

    async def events():
        while True:
            event = await queue.get()
            if event is None:
                break
            kind, payload = event
            yield f"event: {kind}\ndata: {json.dumps(payload, ensure_ascii=False, default=str)}\n\n"

    return StreamingResponse(events(), media_type="text/event-stream")


@app.get("/api/turns/{turn_id}")
def turn(turn_id: str):
    conn = runtime.assistant.conn
    row = conn.execute("SELECT * FROM turns WHERE turn_id = ?", (turn_id,)).fetchone()
    if row is None:
        raise HTTPException(404, "unknown turn")
    data = dict(row)
    data["nodes"] = json.loads(data["nodes"] or "{}")
    data["calls"] = json.loads(data["calls"] or "[]")
    reveal = runtime.assistant.reveal
    data["trace"] = [
        {**e, "summary": reveal(e.get("summary", "")), "items": [reveal(i) for i in e.get("items", [])]}
        for e in json.loads(data["trace"] or "[]")
    ]
    data["audit"] = runtime.assistant.reveal_all(turn_audit(conn, turn_id))
    data["error"] = reveal(data["error"]) if data["error"] else data["error"]
    feedback = conn.execute("SELECT score, comment FROM feedback WHERE turn_id = ?", (turn_id,)).fetchone()
    data["feedback"] = dict(feedback) if feedback else None
    return data


@app.get("/api/turns/{turn_id}/calls/{n}")
def model_call(turn_id: str, n: int):
    """What one model call of the turn received and returned, exactly as the model saw it."""
    row = runtime.assistant.conn.execute(
        "SELECT prompt, params, response FROM model_calls WHERE turn_id = ? AND n = ?", (turn_id, n)
    ).fetchone()
    if row is None:
        raise HTTPException(404, "call not recorded")
    return {k: json.loads(row[k]) if row[k] else None for k in ("prompt", "params", "response")}


@app.post("/api/turns/{turn_id}/feedback")
def feedback(turn_id: str, body: Feedback):
    if body.score not in (1, -1):
        raise HTTPException(400, "score must be 1 or -1")
    conn = runtime.assistant.conn
    conn.execute(
        "INSERT OR REPLACE INTO feedback (turn_id, ts, score, comment) VALUES (?,?,?,?)",
        (turn_id, now_iso(), body.score, body.comment[:1000]),
    )
    conn.commit()
    return {"ok": True}


class Forget(BaseModel):
    text: str


MODEL_SETTINGS = (
    "llm_model",
    "llm_fast_model",
    "llm_base_url",
    "llm_structured",
    "llm_reasoning",
    "privacy_mode",
    "assistant_language",
)
SECRET_SETTINGS = ("anthropic_api_key", "openai_api_key", "openai_base_url")


class SettingsUpdate(BaseModel):
    llm_model: str | None = None
    llm_fast_model: str | None = None
    llm_base_url: str | None = None
    llm_structured: str | None = None
    llm_reasoning: bool | None = None
    privacy_mode: str | None = None
    assistant_language: str | None = None
    anthropic_api_key: str | None = None
    openai_api_key: str | None = None
    openai_base_url: str | None = None


class ModelProbe(BaseModel):
    llm_model: str
    llm_base_url: str = ""
    llm_structured: str = "json_schema"


def write_env(values: dict[str, str], path: Path = Path(".env")) -> None:
    """Set keys in the .env file, keeping everything else (comments, other keys) as it is."""
    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    done = set()
    out = []
    for line in lines:
        key = line.split("=", 1)[0].strip().lstrip("#").strip() if "=" in line else ""
        if key in values and not line.lstrip().startswith("#"):
            out.append(f"{key}={values[key]}")
            done.add(key)
        elif key in values and line.lstrip().startswith("#") and key not in done:
            out.append(f"{key}={values[key]}")  # a commented example becomes the live value
            done.add(key)
        else:
            out.append(line)
    for key, value in values.items():
        if key not in done:
            out.append(f"{key}={value}")
    path.write_text("\n".join(out).rstrip("\n") + "\n", encoding="utf-8")


@app.get("/api/settings")
def read_settings():
    """Model and privacy settings as they run now; secrets only as present/absent."""
    s = runtime.assistant.settings
    import os

    return {
        **{k: getattr(s, k) for k in MODEL_SETTINGS},
        "secrets": {k: bool(os.environ.get(k.upper())) for k in SECRET_SETTINGS},
        "openai_base_url": os.environ.get("OPENAI_BASE_URL", ""),
    }


@app.put("/api/settings")
async def update_settings(body: SettingsUpdate):
    """Write the changes to .env, apply them live (models, name detector and graph are rebuilt under the lock)."""
    import os

    changes = {k: v for k, v in body.model_dump().items() if v is not None}
    if not changes:
        raise HTTPException(400, "nothing to change")
    env = {k.upper(): (str(v).lower() if isinstance(v, bool) else str(v)) for k, v in changes.items()}
    for k in SECRET_SETTINGS:
        if k in changes and not changes[k]:
            env.pop(k.upper(), None)
    write_env(env)
    for k, v in env.items():
        os.environ[k] = v
    for k in SECRET_SETTINGS:
        if k in changes and not changes[k]:
            os.environ.pop(k.upper(), None)
    new = get_settings()
    try:
        with runtime.lock:
            await run_in_threadpool(runtime.assistant.reconfigure, new)
    except Exception as exc:  # noqa: BLE001 — a bad model spec must not take the server down
        raise HTTPException(400, f"{type(exc).__name__}: {exc}") from exc
    return read_settings()


@app.post("/api/settings/test")
async def test_model(body: ModelProbe):
    """One short call to a model spec: does it answer, how fast, and does structured output work."""
    import time

    from langchain_core.messages import HumanMessage, SystemMessage

    from .schemas import Routing

    s = runtime.assistant.settings
    try:
        model = make_chat_model(
            body.llm_model, 256, base_url=body.llm_base_url or s.llm_base_url, structured=body.llm_structured
        )
        t0 = time.perf_counter()
        reply = await run_in_threadpool(model.invoke, [HumanMessage("Responde solo con la palabra: listo")])
        plain = time.perf_counter() - t0
        tokens = (getattr(reply, "usage_metadata", None) or {}).get("output_tokens")
        t1 = time.perf_counter()
        routed = await run_in_threadpool(
            model.with_structured_output(Routing, method="json_schema").invoke,
            [SystemMessage("Classify the message."), HumanMessage("Recuérdame mañana llamar al juzgado")],
        )
        structured = time.perf_counter() - t1
        return {
            "ok": True,
            "reply": str(reply.content)[:120],
            "seconds": round(plain, 2),
            "output_tokens": tokens,
            "structured_ok": isinstance(routed, Routing),
            "structured_seconds": round(structured, 2),
            "route": getattr(routed, "route", None),
        }
    except Exception as exc:  # noqa: BLE001 — the probe reports, it does not raise
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"[:400]}


@app.get("/api/memory/facts")
def memory_facts(limit: int = 500):
    """What Aroa remembers, revealed: text, when, matter and where it came from."""
    a = runtime.assistant
    rows = a.memory.list_facts(limit=limit)
    return [{**r, "text": a.reveal(r.get("text", ""))} for r in rows]


@app.post("/api/memory/forget")
def forget_fact(body: Forget):
    """Remove a remembered fact by its exact text (the user decides what Aroa keeps)."""
    a = runtime.assistant
    removed = a.memory.forget(body.text) or (
        a.memory.forget(a.anonymizer.anonymize(body.text, names=False)) if a.anonymizer else 0
    )
    if not removed and a.anonymizer:  # the user sees names; the store holds placeholders
        for row in a.memory.list_facts(limit=10000):
            if a.reveal(row["text"]) == body.text:
                removed += a.memory.forget(row["text"])
    if not removed:
        raise HTTPException(404, "no such fact")
    return {"removed": removed}


@app.get("/api/alarms")
def list_alarms():
    a = runtime.assistant
    return [{**row, "text": a.reveal(row["text"])} for row in alarms.pending(a.conn)]


@app.get("/api/report")
def daily_report(days: int = 1):
    return report(runtime.assistant.conn, days=days)


@app.get("/api/observability")
def observability(days: int = 7):
    """The metrics page: the dashboard numbers plus what is running now and where its traces go."""
    a = runtime.assistant
    endpoint = a.settings.otlp_endpoint or ""
    data = dashboard(a.conn, days=max(1, min(days, 90)))
    data["errors"] = [{**e, "kind": a.reveal(e["kind"])} for e in data["errors"]]
    data["steps"] = [{**st, "title": say(a.settings.assistant_language, st["step"])} for st in data["steps"]]
    data["turns"] = [{**r, "title": a.reveal(r["title"]), "error": a.reveal(r["error"])} for r in data["turns"]]
    return {
        **data,
        "status": {
            "model": a.settings.llm_model,
            "fast_model": a.settings.llm_fast_model,
            "privacy_mode": a.settings.privacy_mode,
            "memory": a.memory.stats(),
            "traced": a.traced,
            # Phoenix serves its UI where it receives traces: http://host:6006/v1/traces → http://host:6006
            "traces_ui": endpoint.removesuffix("/v1/traces") if a.traced and endpoint.endswith("/v1/traces") else "",
        },
    }


@app.get("/metrics", response_class=PlainTextResponse)
def metrics():
    return prometheus_text(runtime.assistant.conn)


@app.get("/api/documents")
def list_documents():
    sources = runtime.assistant.memory.indexed_sources()
    return sorted(({"source": name, **info} for name, info in sources.items()), key=lambda d: d["title"].lower())


@app.post("/api/documents")
async def upload_document(name: str, request: Request):
    """Raw body upload: the file is saved in docs_dir, pseudonymised and indexed before the call returns."""
    safe = Path(name).name.lstrip(".")
    if not safe or Path(safe).suffix.lower() not in EXTENSIONS:
        raise HTTPException(400, f"only {', '.join(sorted(EXTENSIONS))} files")
    body = await request.body()
    if not body or len(body) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "empty file or larger than 25 MB")
    docs = runtime.assistant.settings.docs_dir
    docs.mkdir(parents=True, exist_ok=True)
    (docs / safe).write_bytes(body)
    result = await run_in_threadpool(runtime.ingest)
    return {"source": safe, **result}


@app.delete("/api/documents/{source}")
async def delete_document(source: str):
    path = runtime.assistant.settings.docs_dir / Path(source).name
    if not path.exists():
        raise HTTPException(404, "unknown document")
    path.unlink()
    return await run_in_threadpool(runtime.ingest)


@app.get("/api/documents/{source}/text")
def document_text(source: str):
    """The original file, for the reader that opens when a citation is clicked."""
    path = runtime.assistant.settings.docs_dir / Path(source).name
    if not path.exists():
        raise HTTPException(404, "unknown document")
    return {"source": path.name, "text": read_text(path)}


@app.get("/api/documents/{source}/html")
def document_html(source: str):
    """The document as the editor needs it: HTML for .html and .docx, raw text otherwise (the client renders it)."""
    path = runtime.assistant.settings.docs_dir / Path(source).name
    if not path.exists():
        raise HTTPException(404, "unknown document")
    suffix = path.suffix.lower()
    if suffix in (".html", ".htm"):
        return {"source": path.name, "format": "html", "content": path.read_text(encoding="utf-8", errors="replace")}
    if suffix == ".docx":
        return {"source": path.name, "format": "html", "content": documents.docx_to_html(path)}
    return {"source": path.name, "format": "markdown" if suffix == ".md" else "text", "content": read_text(path)}


@app.post("/api/documents/html")
async def create_html_document(body: DocumentSave, overwrite: bool = False):
    """A new editable document from the editor's HTML; refuses to replace an existing one unless asked."""
    docs = runtime.assistant.settings.docs_dir
    try:
        name = documents.safe_name(body.name)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    if (docs / name).exists() and not overwrite:
        raise HTTPException(409, f"{name} already exists")
    documents.save_html(docs, name, body.html)
    result = await run_in_threadpool(runtime.ingest)
    return {"source": name, **result}


@app.put("/api/documents/{source}")
async def save_html_document(source: str, body: DocumentSave):
    """Save over an existing editable (.html) document; the previous content becomes a version."""
    path = runtime.assistant.settings.docs_dir / Path(source).name
    if not path.exists():
        raise HTTPException(404, "unknown document")
    if path.suffix.lower() not in (".html", ".htm"):
        raise HTTPException(400, "only .html documents are saved in place; save a copy instead")
    documents.save_html(runtime.assistant.settings.docs_dir, path.name, body.html)
    result = await run_in_threadpool(runtime.ingest)
    return {"source": path.name, **result}


@app.get("/api/documents/{source}/versions")
def document_versions(source: str):
    return documents.versions(runtime.assistant.settings.docs_dir, source)


@app.get("/api/documents/{source}/versions/{version}")
def document_version(source: str, version: str):
    content = documents.version_html(runtime.assistant.settings.docs_dir, source, version)
    if content is None:
        raise HTTPException(404, "unknown version")
    return {"source": Path(source).name, "version": version, "content": content}


@app.post("/api/documents/{source}/versions/{version}/restore")
async def restore_version(source: str, version: str):
    if documents.restore(runtime.assistant.settings.docs_dir, source, version) is None:
        raise HTTPException(404, "unknown version")
    return await run_in_threadpool(runtime.ingest)


@app.post("/api/export/docx")
def export_docx(body: ExportRequest):
    """The editor's HTML as a Word file."""
    name = documents.safe_name(body.name or "documento", ".docx")
    data = documents.html_to_docx(body.html, title=Path(name).stem)
    return Response(
        data,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        headers={"Content-Disposition": f'attachment; filename="{name}"'},
    )


@app.post("/api/attachments")
async def read_attachment(name: str, request: Request, full: bool = False):
    """Extract the text of a file for one turn, without saving or indexing it; `full` skips the turn-sized cut
    (the review pass needs the whole draft)."""
    safe = Path(name).name
    if Path(safe).suffix.lower() not in EXTENSIONS:
        raise HTTPException(400, f"only {', '.join(sorted(EXTENSIONS))} files")
    body = await request.body()
    if not body or len(body) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "empty file or larger than 25 MB")
    with tempfile.NamedTemporaryFile(suffix=Path(safe).suffix, delete=True) as tmp:
        tmp.write(body)
        tmp.flush()
        text = await run_in_threadpool(read_text, Path(tmp.name))
    limit = len(text) if full else MAX_ATTACHMENT_CHARS
    return {"name": safe, "text": text[:limit], "chars": len(text), "truncated": len(text) > limit}


@app.post("/api/documents/{source}/analyze")
def analyze_document(source: str, body: AnalysisRequest):
    """Queue a whole-document analysis; poll /api/jobs/{id} for progress and the result."""
    if source not in runtime.assistant.memory.indexed_sources():
        raise HTTPException(404, "unknown document")
    return {"job_id": runtime.enqueue_analysis(source, body.goal.strip())}


def procedure_template(proc: procedures.Procedure) -> str | None:
    """The template's text: the library document it names (None when it is not there), or the inline one."""
    if not proc.template_ref:
        return proc.template_text
    path = runtime.assistant.settings.docs_dir / Path(proc.template_ref).name
    return read_text(path) if path.is_file() else None


def _procedures() -> list[procedures.Procedure]:
    s = runtime.assistant.settings
    return procedures.load_all(s.procedures_dir, s.procedures_samples_dir)


def _procedure(slug: str) -> procedures.Procedure:
    proc = next((p for p in _procedures() if p.slug == slug), None)
    if proc is None:
        raise HTTPException(404, "unknown procedure")
    return proc


def _checked(body: ProcedureSave, slug: str | None) -> tuple[procedures.Procedure, dict]:
    """The form's record as a procedure, and its check; a new one takes its slug from the name."""
    record = body.model_dump()
    current = _procedure(slug) if slug else None
    proc = procedures.from_dict(record, slug or procedures.slugify(body.name), current.source if current else "")
    taken = {p.slug for p in _procedures()} - ({slug} if slug else set())
    return proc, procedures.check(proc, procedure_template(proc), taken)


@app.get("/api/procedures")
def list_procedures():
    return [p.as_dict() for p in _procedures()]


@app.get("/api/procedures/{slug}")
def get_procedure(slug: str):
    proc = _procedure(slug)
    return {**proc.as_dict(), "check": procedures.check(proc, procedure_template(proc))}


@app.post("/api/procedures/check")
def check_procedure(body: ProcedureSave, slug: str = ""):
    """What the form would save, checked: errors stop the save, warnings are shown next to the fields."""
    return _checked(body, slug or None)[1]


def _save_procedure(body: ProcedureSave, slug: str | None) -> dict:
    proc, result = _checked(body, slug)
    if result["errors"]:
        raise HTTPException(422, {"check": result})
    procedures.save(runtime.assistant.settings.procedures_dir, proc)
    return {**proc.as_dict(), "check": result}


@app.post("/api/procedures")
def create_procedure(body: ProcedureSave):
    return _save_procedure(body, None)


@app.put("/api/procedures/{slug}")
def update_procedure(slug: str, body: ProcedureSave):
    return _save_procedure(body, slug)


@app.delete("/api/procedures/{slug}")
def delete_procedure(slug: str):
    proc = _procedure(slug)
    (runtime.assistant.settings.procedures_dir / proc.source).unlink()
    return {"removed": proc.slug}


@app.post("/api/procedures/{slug}/draft")
def draft_from_procedure(slug: str, body: ProcedureDraft):
    """Queue the drafting: the result (a document for the editor) comes through /api/jobs/{id}.
    A procedure whose check fails (its template lost a field, or is gone) is not followed."""
    proc = _procedure(slug)
    result = procedures.check(proc, procedure_template(proc))
    if result["errors"]:
        raise HTTPException(409, {"check": result})
    return {"job_id": runtime.enqueue_draft(proc, body)}


@app.post("/api/reviews")
def review_draft(body: ReviewRequest):
    """Queue the review of a draft (pasted text or an indexed document); poll /api/jobs/{id} for the result."""
    text = body.text
    if body.source:
        path = runtime.assistant.settings.docs_dir / Path(body.source).name
        if body.source not in runtime.assistant.memory.indexed_sources() or not path.is_file():
            raise HTTPException(404, "unknown document")
        text = read_text(path)
    if not text.strip():
        raise HTTPException(400, "the draft is empty")
    return {"job_id": runtime.enqueue_review(body, text)}


@app.get("/api/jobs")
def jobs(source: str | None = None):
    return analysis.list_jobs(runtime.assistant.conn, source)


@app.get("/api/jobs/{job_id}")
def job(job_id: str):
    data = analysis.get_job(runtime.assistant.conn, job_id)
    if data is None:
        raise HTTPException(404, "unknown job")
    return data


@app.post("/api/clippings")
async def create_clipping(body: Clipping):
    """Cut a passage out of a document into a clipping: a small document, indexed like any other."""
    sources = runtime.assistant.memory.indexed_sources()
    if body.source not in sources:
        raise HTTPException(404, "unknown document")
    try:
        name = clippings.write(
            runtime.assistant.settings.docs_dir,
            source=body.source,
            source_title=sources[body.source]["title"],
            text=body.text,
            note=body.note,
            matter=body.matter,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    result = await run_in_threadpool(runtime.ingest)
    return {"source": name, **result}
