"""Claims against evidence: each kind passes only with a matching successful tool call."""

import sqlite3

from aroa.schemas import Claim
from aroa.telemetry.db import SCHEMA
from aroa.verify import check_claims


def conn():
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    c.executescript(SCHEMA)
    return c


def ev(tool, ok=True, **args):
    return {"tool": tool, "args": args, "ok": ok, "summary": "", "output": ""}


def test_written_file_needs_a_successful_write_and_the_file(tmp_path):
    (tmp_path / "notas.md").write_text("x")
    claim = Claim(kind="file_written", target="notas.md")
    assert check_claims(
        [claim], [ev("write_file", path="notas.md")], project_root=tmp_path, conn=conn(), thread_id="t"
    ) == (1, [])
    checked, failures = check_claims(
        [claim], [ev("write_file", ok=False, path="notas.md")], project_root=tmp_path, conn=conn(), thread_id="t"
    )
    assert checked == 1 and "no successful write_file" in failures[0]
    _, failures = check_claims(
        [Claim(kind="file_written", target="otro.md")],
        [ev("write_file", path="otro.md")],
        project_root=tmp_path,
        conn=conn(),
        thread_id="t",
    )
    assert "does not exist" in failures[0]


def test_url_read_distinguishes_search_from_open():
    claim = Claim(kind="url_read", target="https://www.boe.es/buscar/act.php?id=BOE-A-2000-323")
    assert (
        check_claims(
            [claim],
            [ev("read_page", url="https://boe.es/buscar/act.php")],
            project_root=None,
            conn=conn(),
            thread_id="t",
        )[1]
        == []
    )
    _, failures = check_claims([claim], [ev("search_web", query="LEC")], project_root=None, conn=conn(), thread_id="t")
    assert "only searched" in failures[0]


def test_alarm_claim_needs_a_row_in_the_database():
    c = conn()
    claim = Claim(kind="alarm_created", target="vista")
    _, failures = check_claims([claim], [ev("create_alarm", text="vista")], project_root=None, conn=c, thread_id="t")
    assert "no pending alarm" in failures[0]
    c.execute("INSERT INTO alarms (thread_id, created_ts, due_ts, text) VALUES ('t','now','2999-01-01T10:00','vista')")
    assert check_claims([claim], [ev("create_alarm", text="vista")], project_root=None, conn=c, thread_id="t")[1] == []


def test_other_claims_need_some_successful_tool():
    _, failures = check_claims(
        [Claim(kind="other", target="sent the email")], [], project_root=None, conn=conn(), thread_id="t"
    )
    assert "no tool ran" in failures[0]
    assert (
        check_claims(
            [Claim(kind="other", target="x")],
            [ev("run_command", command="ls")],
            project_root=None,
            conn=conn(),
            thread_id="t",
        )[1]
        == []
    )


def test_redacted_file_counts_as_written(tmp_path):
    (tmp_path / "declaracion.redacted.md").write_text("x")
    claim = Claim(kind="file_written", target="declaracion.redacted.md")
    evidence = [ev("redact_file", path="declaracion.md")]
    assert check_claims([claim], evidence, project_root=tmp_path, conn=conn(), thread_id="t") == (1, [])
