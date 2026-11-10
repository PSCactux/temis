"""Metrics rows, Prometheus text and the daily report are computed from SQLite in code."""

from aroa.telemetry.db import connect
from aroa.telemetry.metrics import TurnCollector, prometheus_text, record_calls, record_turn, report


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


def test_collector_keeps_one_line_per_model_call():
    from uuid import uuid4

    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
    from langchain_core.outputs import ChatGeneration, LLMResult

    c = TurnCollector()
    ok, failed = uuid4(), uuid4()
    meta = {"langgraph_node": "route", "ls_provider": "ollama", "ls_model_name": "qwen3:8b"}
    prompt = [[SystemMessage("route it"), HumanMessage("hola [PERSON_1]")]]
    c.on_chat_model_start({}, prompt, run_id=ok, metadata=meta, invocation_params={"temperature": 0})
    message = AIMessage("x", usage_metadata={"input_tokens": 120, "output_tokens": 8, "total_tokens": 128})
    c.on_llm_end(LLMResult(generations=[[ChatGeneration(message=message)]]), run_id=ok)
    c.on_chat_model_start({"kwargs": {"model": "qwen3-8b"}}, [], run_id=failed, metadata={"langgraph_node": "answer"})
    c.on_llm_error(ValueError("bad json"), run_id=failed)
    first, second = c.calls
    assert (first["step"], first["provider"], first["model"]) == ("route", "ollama", "qwen3:8b")
    assert (first["input_tokens"], first["output_tokens"]) == (120, 8) and first["seconds"] >= 0
    assert (second["step"], second["model"], second["error"]) == ("answer", "qwen3-8b", "ValueError")
    assert c.llm_calls == 2 and c.input_tokens == 120
    assert c.bodies[0]["prompt"] == [
        {"role": "system", "content": "route it"},
        {"role": "human", "content": "hola [PERSON_1]"},
    ]
    assert c.bodies[0]["params"] == {"temperature": 0} and c.bodies[0]["response"][0]["content"] == "x"
    assert c.bodies[1]["response"] == [{"role": "error", "content": "ValueError: bad json"}]


def test_model_calls_are_stored_by_turn_and_position(tmp_path):
    conn = connect(tmp_path / "m.sqlite")
    bodies = [
        {"prompt": [{"role": "human", "content": "a"}], "params": {}, "response": [{"role": "ai", "content": "b"}]}
    ]
    record_calls(conn, "t1", bodies)
    row = conn.execute("SELECT * FROM model_calls WHERE turn_id = 't1' AND n = 0").fetchone()
    assert '"b"' in row["response"] and '"a"' in row["prompt"]
