"""Audit log: what every tool was asked, whether it worked, and what memory was written."""

import hashlib
import json
import sqlite3

from .metrics import now_iso


def log_tool_call(
    conn: sqlite3.Connection, turn_id: str, tool: str, arguments: dict, ok: bool, result: str, seconds: float
):
    conn.execute(
        "INSERT INTO tool_calls (turn_id, ts, tool, arguments, ok, result_hash, seconds) VALUES (?,?,?,?,?,?,?)",
        (
            turn_id,
            now_iso(),
            tool,
            json.dumps(arguments, ensure_ascii=False)[:2000],
            int(ok),
            hashlib.sha256(result.encode("utf-8", "replace")).hexdigest()[:16],
            round(seconds, 3),
        ),
    )
    conn.commit()


def log_memory_write(conn: sqlite3.Connection, turn_id: str, kind: str, text: str, stored: bool) -> None:
    conn.execute(
        "INSERT INTO memory_writes (turn_id, ts, kind, text, stored) VALUES (?,?,?,?,?)",
        (turn_id, now_iso(), kind, text[:2000], int(stored)),
    )
    conn.commit()


def turn_audit(conn: sqlite3.Connection, turn_id: str) -> dict:
    tools = [dict(r) for r in conn.execute("SELECT * FROM tool_calls WHERE turn_id = ? ORDER BY id", (turn_id,))]
    memory = [dict(r) for r in conn.execute("SELECT * FROM memory_writes WHERE turn_id = ? ORDER BY id", (turn_id,))]
    return {"tool_calls": tools, "memory_writes": memory}
