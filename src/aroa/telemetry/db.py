"""One SQLite file for everything the assistant records besides graph checkpoints and vectors:
turn metrics, every model call's prompt and response, the audit log of tool calls and memory writes,
feedback, and alarms."""

import sqlite3
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS turns (
    turn_id TEXT PRIMARY KEY,
    thread_id TEXT NOT NULL,
    ts TEXT NOT NULL,
    source TEXT NOT NULL,            -- user | alarm | eval
    route TEXT,
    seconds REAL,
    llm_calls INTEGER DEFAULT 0,
    input_tokens INTEGER DEFAULT 0,
    output_tokens INTEGER DEFAULT 0,
    tool_calls INTEGER DEFAULT 0,
    tool_errors INTEGER DEFAULT 0,
    retrieved INTEGER DEFAULT 0,
    kept INTEGER DEFAULT 0,
    claims INTEGER DEFAULT 0,
    claims_failed INTEGER DEFAULT 0,
    facts_written INTEGER DEFAULT 0,
    interrupted INTEGER DEFAULT 0,
    error TEXT,
    nodes TEXT,                       -- JSON {node: seconds}
    trace TEXT,                       -- JSON list: the plain-language path of the turn
    calls TEXT                        -- JSON list: one entry per model call (step, provider, model, seconds, tokens)
);
CREATE INDEX IF NOT EXISTS turns_ts ON turns(ts);
CREATE TABLE IF NOT EXISTS model_calls (
    turn_id TEXT NOT NULL,
    n INTEGER NOT NULL,              -- position in turns.calls
    prompt TEXT NOT NULL,            -- JSON list of messages, as the model received them
    params TEXT,                     -- JSON: invocation parameters (model, temperature, schema, tools)
    response TEXT,                   -- JSON list of messages the model returned, or the error
    PRIMARY KEY (turn_id, n)
);
CREATE TABLE IF NOT EXISTS tool_calls (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    turn_id TEXT NOT NULL,
    ts TEXT NOT NULL,
    tool TEXT NOT NULL,
    arguments TEXT NOT NULL,
    ok INTEGER NOT NULL,
    result_hash TEXT NOT NULL,
    seconds REAL
);
CREATE TABLE IF NOT EXISTS memory_writes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    turn_id TEXT NOT NULL,
    ts TEXT NOT NULL,
    kind TEXT NOT NULL,              -- fact | episode | forget
    text TEXT NOT NULL,
    stored INTEGER NOT NULL          -- 0 when skipped as a duplicate
);
CREATE TABLE IF NOT EXISTS feedback (
    turn_id TEXT PRIMARY KEY,
    ts TEXT NOT NULL,
    score INTEGER NOT NULL,          -- 1 or -1
    comment TEXT
);
CREATE TABLE IF NOT EXISTS threads (
    thread_id TEXT PRIMARY KEY,
    matter TEXT NOT NULL DEFAULT '',
    title TEXT NOT NULL DEFAULT '',
    created_ts TEXT NOT NULL,
    updated_ts TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS jobs (
    job_id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,               -- analysis
    source TEXT NOT NULL,             -- document file name
    goal TEXT,
    status TEXT NOT NULL,             -- queued | running | done | failed
    done INTEGER DEFAULT 0,
    total INTEGER DEFAULT 0,
    created_ts TEXT NOT NULL,
    updated_ts TEXT NOT NULL,
    result TEXT,                      -- JSON
    error TEXT,
    log TEXT                          -- JSON list of progress lines
);
CREATE TABLE IF NOT EXISTS alarms (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    thread_id TEXT NOT NULL,
    created_ts TEXT NOT NULL,
    due_ts TEXT NOT NULL,
    text TEXT NOT NULL,
    fired_ts TEXT
);
"""


def connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(SCHEMA)
    columns = {row["name"] for row in conn.execute("PRAGMA table_info(jobs)")}
    if "log" not in columns:  # databases created before jobs kept a readable log
        conn.execute("ALTER TABLE jobs ADD COLUMN log TEXT")
    columns = {row["name"] for row in conn.execute("PRAGMA table_info(turns)")}
    if "calls" not in columns:  # databases created before each model call was recorded
        conn.execute("ALTER TABLE turns ADD COLUMN calls TEXT")
    return conn
