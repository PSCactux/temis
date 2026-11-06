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
