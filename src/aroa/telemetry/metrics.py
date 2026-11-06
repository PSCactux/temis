"""Per-turn metrics: a LangChain callback counts model calls and tokens, the final state supplies the rest,
and one row per turn goes to SQLite. Prometheus text and the daily report are computed from those rows."""

import json
import sqlite3
import time
from datetime import UTC, datetime, timedelta
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler


def now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class TurnCollector(BaseCallbackHandler):
    """Counts model calls, tokens and model latency for one turn. Attach via config["callbacks"]."""

    def __init__(self) -> None:
        self.llm_calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.model_seconds = 0.0
        self._started: dict[Any, float] = {}

    def on_chat_model_start(self, serialized, messages, *, run_id, **kwargs) -> None:
        self.llm_calls += 1
        self._started[run_id] = time.perf_counter()

    def on_llm_end(self, response, *, run_id, **kwargs) -> None:
        started = self._started.pop(run_id, None)
        if started is not None:
            self.model_seconds += time.perf_counter() - started
        for generation in response.generations:
            for g in generation:
                usage = getattr(g.message, "usage_metadata", None) if hasattr(g, "message") else None
                if usage:
                    self.input_tokens += int(usage.get("input_tokens") or 0)
                    self.output_tokens += int(usage.get("output_tokens") or 0)


def turn_metrics(state: dict, collector: TurnCollector, *, seconds: float) -> dict:
    """Numbers for the turn row, read from the final graph state and the collector."""
    trace = state.get("trace") or []
    evidence = state.get("evidence") or []
    verification = state.get("verification") or {}
    return {
        "route": state.get("route"),
        "seconds": round(seconds, 3),
        "llm_calls": collector.llm_calls,
        "input_tokens": collector.input_tokens,
        "output_tokens": collector.output_tokens,
        "tool_calls": len(evidence),
        "tool_errors": sum(1 for e in evidence if not e.get("ok")),
        "retrieved": len(state.get("retrieved") or []),
        "kept": len(state.get("sources") or []),
        "claims": verification.get("checked", 0),
        "claims_failed": verification.get("failed", 0),
        "facts_written": len(state.get("facts_written") or []),
        "nodes": json.dumps({t["step"]: t.get("seconds", 0) for t in trace}),
    }


def record_turn(conn: sqlite3.Connection, turn_id: str, thread_id: str, source: str, metrics: dict) -> None:
    columns = ["turn_id", "thread_id", "ts", "source", *metrics]
    values = [turn_id, thread_id, now_iso(), source, *metrics.values()]
    conn.execute(f"INSERT OR REPLACE INTO turns ({', '.join(columns)}) VALUES ({', '.join('?' * len(values))})", values)
    conn.commit()


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    index = min(len(values) - 1, max(0, round(p * (len(values) - 1))))
    return values[index]


def prometheus_text(conn: sqlite3.Connection) -> str:
    """Counters over all recorded turns plus latency quantiles over the last 24 hours."""
    rows = conn.execute("SELECT * FROM turns").fetchall()
    since = (datetime.now(UTC) - timedelta(days=1)).isoformat(timespec="seconds")
    recent = [r for r in rows if r["ts"] >= since]
    lines = []

    def gauge(name: str, value, help_text: str, labels: str = "") -> None:
        lines.append(f"# HELP {name} {help_text}")
        lines.append(f"# TYPE {name} gauge")
        lines.append(f"{name}{labels} {value}")

    by_route: dict[str, int] = {}
    for r in rows:
        by_route[r["route"] or "none"] = by_route.get(r["route"] or "none", 0) + 1
    lines += ["# HELP aroa_turns_total Turns processed, by route", "# TYPE aroa_turns_total counter"]
    lines += [f'aroa_turns_total{{route="{route}"}} {n}' for route, n in sorted(by_route.items())]
    for column, help_text in [
        ("llm_calls", "Model calls"),
        ("input_tokens", "Prompt tokens"),
        ("output_tokens", "Completion tokens"),
        ("tool_calls", "Tool calls"),
        ("tool_errors", "Tool calls that failed"),
        ("claims", "Claims checked by the verifier"),
        ("claims_failed", "Claims the verifier rejected"),
        ("facts_written", "Facts written to memory"),
    ]:
        lines += [f"# HELP aroa_{column}_total {help_text}", f"# TYPE aroa_{column}_total counter"]
        lines.append(f"aroa_{column}_total {sum(r[column] or 0 for r in rows)}")
    errors = sum(1 for r in rows if r["error"])
    lines += ["# HELP aroa_turn_errors_total Turns that ended in an error", "# TYPE aroa_turn_errors_total counter"]
    lines.append(f"aroa_turn_errors_total {errors}")
    seconds = [r["seconds"] for r in recent if r["seconds"] is not None]
    gauge("aroa_turn_seconds_p50", round(_percentile(seconds, 0.5), 3), "Median turn latency, last 24h")
    gauge("aroa_turn_seconds_p95", round(_percentile(seconds, 0.95), 3), "p95 turn latency, last 24h")
    gauge("aroa_turns_24h", len(recent), "Turns in the last 24h")
    feedback = conn.execute("SELECT score, COUNT(*) AS n FROM feedback GROUP BY score").fetchall()
    lines += ["# HELP aroa_feedback_total Feedback received, by score", "# TYPE aroa_feedback_total counter"]
    lines += [f'aroa_feedback_total{{score="{r["score"]}"}} {r["n"]}' for r in feedback]
    pending = conn.execute("SELECT COUNT(*) AS n FROM alarms WHERE fired_ts IS NULL").fetchone()["n"]
    gauge("aroa_alarms_pending", pending, "Alarms not yet fired")
    return "\n".join(lines) + "\n"


def report(conn: sqlite3.Connection, days: int = 1) -> dict:
    """Aggregates for the last N days, computed in code."""
    since = (datetime.now(UTC) - timedelta(days=days)).isoformat(timespec="seconds")
    rows = conn.execute("SELECT * FROM turns WHERE ts >= ? ORDER BY ts", (since,)).fetchall()
    seconds = [r["seconds"] for r in rows if r["seconds"] is not None]
    routes: dict[str, int] = {}
    for r in rows:
        routes[r["route"] or "none"] = routes.get(r["route"] or "none", 0) + 1
    feedback = conn.execute("SELECT score, COUNT(*) AS n FROM feedback WHERE ts >= ? GROUP BY score", (since,))
    scores = {str(r["score"]): r["n"] for r in feedback}
    return {
        "since": since,
        "turns": len(rows),
        "by_route": routes,
        "by_source": _count(rows, "source"),
        "latency_p50": round(_percentile(seconds, 0.5), 2),
        "latency_p95": round(_percentile(seconds, 0.95), 2),
        "llm_calls": sum(r["llm_calls"] or 0 for r in rows),
        "input_tokens": sum(r["input_tokens"] or 0 for r in rows),
        "output_tokens": sum(r["output_tokens"] or 0 for r in rows),
        "tool_calls": sum(r["tool_calls"] or 0 for r in rows),
        "tool_errors": sum(r["tool_errors"] or 0 for r in rows),
        "claims_checked": sum(r["claims"] or 0 for r in rows),
        "claims_failed": sum(r["claims_failed"] or 0 for r in rows),
        "facts_written": sum(r["facts_written"] or 0 for r in rows),
        "errors": [r["error"] for r in rows if r["error"]],
        "feedback": {"up": scores.get("1", 0), "down": scores.get("-1", 0)},
        "memory_facts": conn.execute(
            "SELECT COUNT(*) AS n FROM memory_writes WHERE kind='fact' AND stored=1"
        ).fetchone()["n"],
    }


def _count(rows, column: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for r in rows:
        out[r[column] or "none"] = out.get(r[column] or "none", 0) + 1
    return out
