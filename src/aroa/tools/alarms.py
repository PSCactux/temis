"""Alarms live in SQLite. A scheduler (see server.py) wakes the thread when one is due."""

import sqlite3
from datetime import datetime, timedelta

from langchain_core.tools import tool

from ..telemetry.metrics import now_iso
from .base import ToolContext, ToolError


def parse_due(at: str | None, minutes_from_now: int | None) -> datetime:
    if minutes_from_now is not None:
        if minutes_from_now <= 0:
            raise ToolError("minutes_from_now must be positive.")
        return (datetime.now().astimezone() + timedelta(minutes=minutes_from_now)).replace(second=0, microsecond=0)
    if not at:
        raise ToolError("Give either at (YYYY-MM-DD HH:MM) or minutes_from_now.")
    try:
        due = datetime.fromisoformat(at.strip().replace("T", " "))
    except ValueError as exc:
        raise ToolError("at must look like 2026-05-04 18:30") from exc
    if due.tzinfo is None:
        due = due.astimezone()
    if due <= datetime.now().astimezone():
        raise ToolError(f"{due.strftime('%Y-%m-%d %H:%M')} is in the past.")
    return due


def create(conn: sqlite3.Connection, thread_id: str, due: datetime, text: str) -> int:
    cursor = conn.execute(
        "INSERT INTO alarms (thread_id, created_ts, due_ts, text) VALUES (?,?,?,?)",
        (thread_id, now_iso(), due.isoformat(timespec="minutes"), text.strip()),
    )
    conn.commit()
    return int(cursor.lastrowid)


def pending(conn: sqlite3.Connection, thread_id: str | None = None) -> list[dict]:
    query = "SELECT * FROM alarms WHERE fired_ts IS NULL"
    params: tuple = ()
    if thread_id:
        query += " AND thread_id = ?"
        params = (thread_id,)
    return [dict(r) for r in conn.execute(query + " ORDER BY due_ts", params)]


def due_now(conn: sqlite3.Connection) -> list[dict]:
    now = datetime.now().astimezone().isoformat(timespec="minutes")
    return [a for a in pending(conn) if a["due_ts"] <= now]


def mark_fired(conn: sqlite3.Connection, alarm_id: int) -> None:
    conn.execute("UPDATE alarms SET fired_ts = ? WHERE id = ?", (now_iso(), alarm_id))
    conn.commit()


def delete(conn: sqlite3.Connection, alarm_id: int) -> bool:
    deleted = conn.execute("DELETE FROM alarms WHERE id = ? AND fired_ts IS NULL", (alarm_id,)).rowcount
    conn.commit()
    return deleted > 0


def _format(alarm: dict) -> str:
    return f"#{alarm['id']} · {alarm['due_ts'].replace('T', ' ')[:16]} · {alarm['text']}"


def make_tools(ctx: ToolContext) -> list:
    @tool
    def create_alarm(text: str, at: str | None = None, minutes_from_now: int | None = None) -> str:
        """Create a reminder. Give the time as at="YYYY-MM-DD HH:MM" (local time) or as minutes_from_now."""
        due = parse_due(at, minutes_from_now)
        alarm_id = create(ctx.conn, ctx.thread_id, due, text)
        return f"Alarm #{alarm_id} set for {due.strftime('%Y-%m-%d %H:%M')}: {text.strip()}"

    @tool
    def list_alarms() -> str:
        """List the pending alarms."""
        alarms = pending(ctx.conn, ctx.thread_id)
        return "\n".join(_format(a) for a in alarms) if alarms else "No pending alarms."

    @tool
    def delete_alarm(alarm_id: int) -> str:
        """Delete a pending alarm by its number."""
        if not delete(ctx.conn, alarm_id):
            raise ToolError(f"There is no pending alarm #{alarm_id}.")
        return f"Alarm #{alarm_id} deleted."

    return [create_alarm, list_alarms, delete_alarm]
