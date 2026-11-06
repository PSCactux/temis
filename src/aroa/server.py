"""HTTP API and web app: streaming turns over SSE, thread history, the path and audit of every turn,
feedback, Prometheus metrics, the daily report, and the alarm scheduler."""

import asyncio
import json
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, PlainTextResponse, StreamingResponse
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from .app import Assistant
from .config import get_settings
from .memory.loaders import EXTENSIONS
from .telemetry.audit import turn_audit
from .telemetry.metrics import now_iso, prometheus_text, report
from .tools import alarms

WEB = Path(__file__).parent / "web"
ALARM_TICK_SECONDS = 20
MAX_UPLOAD_BYTES = 25 * 1024 * 1024


class Message(BaseModel):
    text: str
    matter: str = ""


class Feedback(BaseModel):
    score: int  # 1 or -1
    comment: str = ""


class Runtime:
    """One assistant, one lock: a local model answers one turn at a time, alarms included."""

    def __init__(self) -> None:
        self.assistant = Assistant(get_settings())
        self.lock = threading.Lock()

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


@app.get("/")
def index():
    return FileResponse(WEB / "index.html")


@app.get("/api/health")
def health():
    a = runtime.assistant
    return {
        "ok": True,
        "model": a.settings.llm_model,
        "privacy_mode": a.settings.privacy_mode,
        "traced": a.traced,
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
    queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def produce() -> None:
        with runtime.lock:
            for kind, item in runtime.assistant.stream(thread_id, message.text, matter=matter):
                payload = item.__dict__ if kind == "done" else item
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
    data["trace"] = json.loads(data["trace"] or "[]")
    data["audit"] = turn_audit(conn, turn_id)
    feedback = conn.execute("SELECT score, comment FROM feedback WHERE turn_id = ?", (turn_id,)).fetchone()
    data["feedback"] = dict(feedback) if feedback else None
    return data


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


@app.get("/api/alarms")
def list_alarms():
    a = runtime.assistant
    return [{**row, "text": a.reveal(row["text"])} for row in alarms.pending(a.conn)]


@app.get("/api/report")
def daily_report(days: int = 1):
    return report(runtime.assistant.conn, days=days)


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
