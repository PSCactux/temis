"""Wire settings, models, memory, privacy, telemetry and the graph into one object the CLI and server share."""

import sqlite3
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from .config import Settings
from .graph import build_graph, load_persona
from .i18n import TEXTS
from .memory.store import MemoryStore
from .models import make_chat_model
from .privacy import NAMES_PROMPT, Anonymizer, Names
from .telemetry.db import connect
from .telemetry.metrics import TurnCollector, record_calls, record_turn, turn_metrics
from .telemetry.tracing import setup_tracing


@dataclass
class TurnResult:
    turn_id: str
    thread_id: str
    reply: str  # revealed: placeholders replaced by the real values for the user
    kind: str
    trace: list[dict]
    metrics: dict
    interrupted: str | None = None  # the question the assistant is waiting for
    extras: dict = field(default_factory=dict)


class Assistant:
    def __init__(self, settings: Settings, *, llm=None, fast_llm=None, memory: MemoryStore | None = None):
        load_dotenv()
        self.settings = settings
        self.traced = setup_tracing(settings)
        self.conn = connect(settings.db_path)
        self.llm = llm or self.model_from(settings, settings.llm_model, settings.llm_max_tokens, settings.llm_reasoning)
        self.fast_llm = fast_llm or self.model_from(settings, settings.llm_fast_model, settings.llm_fast_max_tokens)
        self.memory = memory or MemoryStore(settings)
        self.anonymizer = self._make_anonymizer()
        settings.data_dir.mkdir(parents=True, exist_ok=True)
        self.checkpointer = SqliteSaver(sqlite3.connect(str(settings.checkpoint_db), check_same_thread=False))
        self.graph = self._build_graph()

    @staticmethod
    def model_from(settings: Settings, spec: str, max_tokens: int, reasoning: bool = False):
        return make_chat_model(
            spec, max_tokens, reasoning=reasoning, base_url=settings.llm_base_url, structured=settings.llm_structured
        )

    def _build_graph(self):
        s = self.settings
        return build_graph(
            self.llm,
            self.fast_llm,
            self.memory,
            s,
            self.conn,
            persona=load_persona(s.persona_dir, s.assistant_name),
            anonymizer=self.anonymizer,
            checkpointer=self.checkpointer,
        )

    def reconfigure(self, settings: Settings) -> None:
        """New model settings without restarting: models, name detector and graph are rebuilt; stores stay open."""
        self.settings = settings
        self.llm = self.model_from(settings, settings.llm_model, settings.llm_max_tokens, settings.llm_reasoning)
        self.fast_llm = self.model_from(settings, settings.llm_fast_model, settings.llm_fast_max_tokens)
        self.anonymizer = self._make_anonymizer()
        self.graph = self._build_graph()

    def _make_anonymizer(self) -> Anonymizer | None:
        mode = self.settings.privacy_mode
        if mode == "off":
            return None
        extractor = None
        if mode == "full":
            names_model = self.fast_llm.with_structured_output(Names, method="json_schema")

            def extractor(text: str) -> Names:
                return names_model.invoke([SystemMessage(NAMES_PROMPT), HumanMessage(text[:8000])])

        return Anonymizer(self.conn, extractor)

    # ── documents ──────────────────────────────────────────────────────
    def ingest(self) -> dict:
        anonymize = self.anonymizer.anonymize if self.anonymizer else None
        return self.memory.sync_documents(anonymize)

    # ── turns ──────────────────────────────────────────────────────────
    def config(self, thread_id: str, turn_id: str, source: str, matter: str, collector, documents=()) -> dict:
        return {
            "configurable": {
                "thread_id": thread_id,
                "turn_id": turn_id,
                "source": source,
                "matter": matter,
                "documents": list(documents),  # sources the turn is confined to; empty means all
            },
            "callbacks": [collector],
            "metadata": {"turn_id": turn_id, "thread_id": thread_id, "matter": matter, "source": source},
            "recursion_limit": 60,
        }

    def waiting_for(self, thread_id: str) -> str | None:
        """The question the thread is paused on, if any."""
        state = self.graph.get_state({"configurable": {"thread_id": thread_id}})
        for task in state.tasks:
            for item in task.interrupts:
                return (item.value or {}).get("question", "")
        return None

    def stream(self, thread_id: str, text: str, *, source: str = "user", matter: str = "", documents=()):
        """Yield ("token", str) | ("node", dict) | ("done", TurnResult). Resumes the thread if it was waiting."""
        turn_id = uuid.uuid4().hex[:12]
        collector = TurnCollector()
        config = self.config(thread_id, turn_id, source, matter, collector, documents)
        payload: Any = Command(resume=text) if self.waiting_for(thread_id) else {"messages": [HumanMessage(text)]}
        started = time.perf_counter()
        state: dict = {}
        error: str | None = None
        try:
            for mode, chunk in self.graph.stream(payload, config, stream_mode=["messages", "updates", "debug"]):
                if mode == "debug":
                    if isinstance(chunk, dict) and chunk.get("type") == "task":
                        name = (chunk.get("payload") or {}).get("name", "")
                        if name and not name.startswith("__"):
                            yield "start", {"step": name, "title": step_title(self.settings.assistant_language, name)}
                    continue
                if mode == "messages":
                    message, meta = chunk
                    if (
                        isinstance(message, AIMessage)
                        and message.content
                        and meta.get("langgraph_node") in ("chat", "act")
                    ):
                        yield "token", self.reveal(message.content)
                elif mode == "updates":
                    for node, update in chunk.items():
                        if node == "__interrupt__":
                            continue
                        if update and update.get("trace"):
                            yield "node", update["trace"][-1]
        except Exception as exc:  # noqa: BLE001 — the turn is recorded as failed and the error shown
            error = f"{type(exc).__name__}: {exc}"
        snapshot = self.graph.get_state({"configurable": {"thread_id": thread_id}})
        state = snapshot.values
        waiting = self.waiting_for(thread_id)
        seconds = time.perf_counter() - started
        metrics = turn_metrics(state, collector, seconds=seconds)
        metrics["interrupted"] = int(bool(waiting))
        metrics["error"] = error
        metrics["trace"] = __import__("json").dumps(state.get("trace") or [])
        record_turn(self.conn, turn_id, thread_id, source, metrics)
        record_calls(self.conn, turn_id, collector.bodies)
        reply = waiting or error or self._last_reply(state)
        last = state["messages"][-1] if state.get("messages") else None
        kind = (
            "error"
            if error
            else "question"
            if waiting
            else (getattr(last, "additional_kwargs", {}) or {}).get("kind", "")
        )
        yield (
            "done",
            TurnResult(
                turn_id=turn_id,
                thread_id=thread_id,
                reply=self.reveal(reply),
                kind=kind,
                trace=state.get("trace") or [],
                metrics=metrics,
                interrupted=self.reveal(waiting) if waiting else None,
                extras=self.reveal_all(
                    {k: v for k, v in (getattr(last, "additional_kwargs", {}) or {}).items() if k != "turn_id"}
                ),
            ),
        )

    def ask(self, thread_id: str, text: str, **kwargs) -> TurnResult:
        for kind, item in self.stream(thread_id, text, **kwargs):
            if kind == "done":
                return item
        raise RuntimeError("stream ended without a result")

    def _last_reply(self, state: dict) -> str:
        for message in reversed(state.get("messages") or []):
            if isinstance(message, AIMessage) and not message.tool_calls:
                return message.content
        return ""

    def reveal(self, text: str) -> str:
        return self.anonymizer.reveal(text) if self.anonymizer and text else text

    def reveal_all(self, value):
        """`reveal` on every string inside a structure (sources, quotes, labels), so nothing shown is a placeholder."""
        if isinstance(value, str):
            return self.reveal(value)
        if isinstance(value, list):
            return [self.reveal_all(v) for v in value]
        if isinstance(value, dict):
            return {k: self.reveal_all(v) for k, v in value.items()}
        return value

    def history(self, thread_id: str) -> list[dict]:
        state = self.graph.get_state({"configurable": {"thread_id": thread_id}}).values
        rows = []
        for message in state.get("messages") or []:
            if isinstance(message, HumanMessage):
                rows.append({"role": "user", "text": self.reveal(message.content)})
            elif isinstance(message, AIMessage) and not message.tool_calls:
                rows.append(
                    {
                        "role": "assistant",
                        "text": self.reveal(message.content),
                        **self.reveal_all(message.additional_kwargs),
                    }
                )
        return rows

    def now(self) -> str:
        return datetime.now().astimezone().isoformat(timespec="minutes")


def step_title(lang: str, node: str) -> str:
    """The plain-language title of a graph node, or its name when none is defined."""
    return (TEXTS.get(lang) or {}).get(node) or TEXTS["en"].get(node) or node.replace("_", " ")
