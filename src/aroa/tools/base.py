"""What every tool shares: the turn's context and a failure type the executor turns into evidence."""

import sqlite3
from dataclasses import dataclass
from pathlib import Path

from ..config import Settings


class ToolError(Exception):
    """A tool could not do what was asked. The message goes back to the model; the call is logged as failed."""


@dataclass
class ToolContext:
    settings: Settings
    conn: sqlite3.Connection
    thread_id: str
    turn_id: str
    project: str = ""
    memory: object | None = None  # MemoryStore; typed loosely to keep tools importable without fastembed
    anonymizer: object | None = None  # privacy.Anonymizer
    matter: str = ""

    @property
    def project_root(self) -> Path:
        if not self.project:
            raise ToolError("No project selected for this turn.")
        root = (self.settings.projects_dir / self.project).resolve()
        if not root.is_dir():
            raise ToolError(f"Project {self.project!r} does not exist under {self.settings.projects_dir}.")
        return root
