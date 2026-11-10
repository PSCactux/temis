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
    """Counts model calls, tokens and model latency for one turn, and keeps one line per call (graph step,
    provider, model, seconds, tokens). Attach via config["callbacks"]."""

    def __init__(self) -> None:
        self.llm_calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.model_seconds = 0.0
        self.calls: list[dict] = []
        self._started: dict[Any, tuple[float, dict]] = {}

    def on_chat_model_start(self, serialized, messages, *, run_id, metadata=None, **kwargs) -> None:
        self.llm_calls += 1
        meta = metadata or {}
        model = meta.get("ls_model_name") or ((serialized or {}).get("kwargs") or {}).get("model", "")
        call = {"step": meta.get("langgraph_node", ""), "provider": meta.get("ls_provider", ""), "model": model}
        self.calls.append(call)
        self._started[run_id] = (time.perf_counter(), call)

    def on_llm_end(self, response, *, run_id, **kwargs) -> None:
        started, call = self._started.pop(run_id, (None, {}))
        if started is not None:
            seconds = time.perf_counter() - started
            self.model_seconds += seconds
            call["seconds"] = round(seconds, 3)
        for generation in response.generations:
            for g in generation:
                usage = getattr(g.message, "usage_metadata", None) if hasattr(g, "message") else None
                if usage:
                    self.input_tokens += int(usage.get("input_tokens") or 0)
                    self.output_tokens += int(usage.get("output_tokens") or 0)
                    call["input_tokens"] = call.get("input_tokens", 0) + int(usage.get("input_tokens") or 0)
                    call["output_tokens"] = call.get("output_tokens", 0) + int(usage.get("output_tokens") or 0)

    def on_llm_error(self, error, *, run_id, **kwargs) -> None:
        started, call = self._started.pop(run_id, (None, {}))
        if started is not None:
            call["seconds"] = round(time.perf_counter() - started, 3)
            call["error"] = type(error).__name__


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
        "calls": json.dumps(collector.calls),
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


def _error_kind(error: str) -> str:
    """`OutputParserException: Failed to parse…` → `OutputParserException`; anything else, its first words."""
    head = error.strip().split("\n", 1)[0]
    name = head.split(":", 1)[0].strip()
    return name if name and " " not in name else head[:60]


def dashboard(conn: sqlite3.Connection, days: int = 7, *, now: datetime | None = None) -> dict:
    """Everything the metrics page draws for the last N days, from the same SQLite rows as `report`.

    The activity series has one bucket per local hour for a single day and one per local day otherwise, with
    empty buckets filled in so the chart has no gaps. Turns are the most recent 300, errors cut to their first line."""
    now = now or datetime.now(UTC)
    start = now - timedelta(days=days)
    hourly = days <= 1
    rows = conn.execute("SELECT * FROM turns WHERE ts >= ? ORDER BY ts", (start.isoformat(timespec="seconds"),))
    rows = rows.fetchall()

    def bucket(ts: datetime) -> str:
        return ts.strftime("%Y-%m-%dT%H:00") if hourly else ts.strftime("%Y-%m-%d")

    step = timedelta(hours=1) if hourly else timedelta(days=1)
    cursor = start.astimezone().replace(minute=0, second=0, microsecond=0)  # buckets in the local time of the machine
    if not hourly:
        cursor = cursor.replace(hour=0)
    series: dict[str, dict] = {}
    while cursor <= now:
        series[bucket(cursor)] = {"t": bucket(cursor), "turns": 0, "errors": 0, "seconds": []}
        cursor += step
    for r in rows:
        b = series.get(bucket(datetime.fromisoformat(r["ts"]).astimezone()))
        if b is None:
            continue
        b["turns"] += 1
        b["errors"] += 1 if r["error"] else 0
        if r["seconds"] is not None:
            b["seconds"].append(r["seconds"])
    activity = [
        {"t": b["t"], "turns": b["turns"], "errors": b["errors"], "p50": round(_percentile(b["seconds"], 0.5), 2)}
        for b in series.values()
    ]

    nodes: dict[str, list[float]] = {}
    for r in rows:
        for node, seconds in json.loads(r["nodes"] or "{}").items():
            nodes.setdefault(node, []).append(seconds or 0.0)
    steps = sorted(
        (
            {"step": n, "avg": round(sum(v) / len(v), 2), "p95": round(_percentile(v, 0.95), 2), "n": len(v)}
            for n, v in nodes.items()
        ),
        key=lambda s: -s["avg"],
    )

    since = start.isoformat(timespec="seconds")
    tools = [
        dict(t)
        for t in conn.execute(
            "SELECT tool, COUNT(*) AS calls, SUM(1 - ok) AS errors, ROUND(AVG(seconds), 2) AS avg"
            " FROM tool_calls WHERE ts >= ? GROUP BY tool ORDER BY calls DESC",
            (since,),
        )
    ]
    writes = conn.execute(
        "SELECT kind, stored, COUNT(*) AS n FROM memory_writes WHERE ts >= ? GROUP BY kind, stored", (since,)
    ).fetchall()
    memory = {"facts": 0, "duplicates": 0, "episodes": 0, "forgotten": 0}
    for w in writes:
        if w["kind"] == "fact":
            memory["facts" if w["stored"] else "duplicates"] += w["n"]
        elif w["kind"] == "episode":
            memory["episodes"] += w["n"]
        elif w["kind"] == "forget":
            memory["forgotten"] += w["n"]

    errors: dict[str, dict] = {}
    for r in rows:
        if r["error"]:
            kind = _error_kind(r["error"])
            e = errors.setdefault(kind, {"kind": kind, "n": 0, "last": r["ts"], "turn_id": r["turn_id"]})
            e["n"] += 1
            e["last"], e["turn_id"] = r["ts"], r["turn_id"]

    titles = {t["thread_id"]: t["title"] for t in conn.execute("SELECT thread_id, title FROM threads")}
    scores = {f["turn_id"]: f["score"] for f in conn.execute("SELECT turn_id, score FROM feedback")}
    recent = []
    for r in reversed(rows[-300:]):
        recent.append(
            {
                "turn_id": r["turn_id"],
                "thread_id": r["thread_id"],
                "title": titles.get(r["thread_id"], ""),
                "ts": r["ts"],
                "source": r["source"],
                "route": r["route"] or "",
                "seconds": r["seconds"],
                "llm_calls": r["llm_calls"],
                "tokens": (r["input_tokens"] or 0) + (r["output_tokens"] or 0),
                "tool_calls": r["tool_calls"],
                "tool_errors": r["tool_errors"],
                "claims": r["claims"],
                "claims_failed": r["claims_failed"],
                "facts_written": r["facts_written"],
                "error": _error_kind(r["error"]) if r["error"] else "",
                "score": scores.get(r["turn_id"]),
            }
        )

    summary = report(conn, days=days)
    summary.pop("errors")
    return {
        "days": days,
        "bucket": "hour" if hourly else "day",
        "summary": {**summary, "error_turns": sum(1 for r in rows if r["error"])},
        "activity": activity,
        "steps": steps,
        "tools": tools,
        "memory": memory,
        "errors": sorted(errors.values(), key=lambda e: -e["n"]),
        "turns": recent,
    }
