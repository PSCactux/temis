"""Metrics rows, Prometheus text and the daily report are computed from SQLite in code."""

from aroa.telemetry.db import connect
from aroa.telemetry.metrics import prometheus_text, record_turn, report


def test_turns_feed_prometheus_and_report(tmp_path):
    conn = connect(tmp_path / "m.sqlite")
    record_turn(
        conn,
        "a",
        "t1",
        "user",
        {"route": "chat", "seconds": 1.0, "llm_calls": 2, "input_tokens": 100, "output_tokens": 20},
    )
    record_turn(
        conn,
        "b",
        "t1",
        "user",
        {"route": "alarms", "seconds": 3.0, "llm_calls": 4, "tool_calls": 2, "claims": 1, "claims_failed": 1},
    )
    record_turn(conn, "c", "t2", "alarm", {"route": "chat", "seconds": 2.0, "error": "boom"})
    text = prometheus_text(conn)
    assert 'aroa_turns_total{route="chat"} 2' in text and 'aroa_turns_total{route="alarms"} 1' in text
    assert (
        "aroa_llm_calls_total 6" in text and "aroa_claims_failed_total 1" in text and "aroa_turn_errors_total 1" in text
    )
    assert "aroa_turn_seconds_p50 2.0" in text
    day = report(conn)
    assert (
        day["turns"] == 3
        and day["by_route"] == {"chat": 2, "alarms": 1}
        and day["by_source"] == {"user": 2, "alarm": 1}
    )
    assert day["latency_p95"] == 3.0 and day["errors"] == ["boom"] and day["tool_calls"] == 2


def test_dashboard_fills_the_series_and_groups_errors(tmp_path):
    from datetime import UTC, datetime, timedelta

    from aroa.telemetry.audit import log_memory_write, log_tool_call
    from aroa.telemetry.metrics import dashboard

    conn = connect(tmp_path / "m.sqlite")
    nodes = '{"recall": 1.0, "route": 3.0}'
    record_turn(conn, "a", "t1", "user", {"route": "chat", "seconds": 4.0, "nodes": nodes})
    record_turn(conn, "b", "t1", "user", {"route": "documents", "seconds": 2.0, "nodes": '{"recall": 3.0}'})
    record_turn(conn, "c", "t1", "user", {"route": "web", "seconds": 1.0, "error": "OutputParserException: bad\nmore"})
    log_tool_call(conn, "b", "search_web", {}, ok=False, result="x", seconds=0.5)
    log_memory_write(conn, "a", "fact", "uno", stored=True)
    log_memory_write(conn, "b", "fact", "uno", stored=False)
    data = dashboard(conn, days=7, now=datetime.now(UTC) + timedelta(minutes=1))
    assert data["bucket"] == "day" and len(data["activity"]) in (7, 8)  # every day, empty ones included
    assert sum(b["turns"] for b in data["activity"]) == 3 and data["activity"][-1]["errors"] == 1
    assert data["steps"][0] == {"step": "route", "avg": 3.0, "p95": 3.0, "n": 1}
    assert data["steps"][1]["step"] == "recall" and data["steps"][1]["avg"] == 2.0
    assert data["errors"] == [{"kind": "OutputParserException", "n": 1, "last": data["turns"][0]["ts"], "turn_id": "c"}]
    assert data["turns"][0]["error"] == "OutputParserException" and data["summary"]["error_turns"] == 1
    assert data["tools"] == [{"tool": "search_web", "calls": 1, "errors": 1, "avg": 0.5}]
    assert data["memory"] == {"facts": 1, "duplicates": 1, "episodes": 0, "forgotten": 0}
    hours = dashboard(conn, days=1)
    assert hours["bucket"] == "hour" and len(hours["activity"]) in (24, 25)
