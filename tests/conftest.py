"""Fixtures: an assistant wired to a scripted model and an in-memory "memory", in a temporary data folder."""

import os
from pathlib import Path

import pytest

from aroa.app import Assistant
from aroa.config import Settings
from aroa.memory.chunking import Chunk
from aroa.models import FakeChatModel


class FakeMemory:
    """Same interface as MemoryStore, no vectors: recall returns what the test put in."""

    def __init__(self):
        self.facts: list[dict] = []
        self.episodes: list[dict] = []
        self.chunks: list[Chunk] = []
        self.titles: list[str] = []

    def recall(self, query, *, exclude_thread=None, matter=""):
        facts = [f for f in self.facts if f.get("matter", "") in ("", matter)]
        return {"facts": facts, "episodes": []}

    def add_facts(self, facts, *, thread_id, turn_id, matter=""):
        stored, skipped = [], []
        for fact in facts:
            if any(f["text"] == fact for f in self.facts):
                skipped.append(fact)
            else:
                self.facts.append({"text": fact, "ts": "2026-01-01T00:00:00+00:00", "matter": matter})
                stored.append(fact)
        return stored, skipped

    def add_episode(self, user, assistant, *, thread_id, turn_id, matter=""):
        self.episodes.append({"user": user, "assistant": assistant, "matter": matter})

    def document_titles(self):
        return self.titles

    def search_documents(self, query, k=None):
        return list(self.chunks)

    def sync_documents(self, anonymize=None):
        return {"added": [], "removed": [], "total": 0}

    def stats(self):
        return {"facts": len(self.facts), "episodes": len(self.episodes), "documents": len(self.chunks)}


@pytest.fixture
def workdir(tmp_path, monkeypatch) -> Path:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data/projects/case-2026-014").mkdir(parents=True)
    (tmp_path / "data/persona").mkdir(parents=True)
    (tmp_path / "data/persona/10-who.md").write_text("# Who you are\n\nYou are Aroa, a law firm's assistant.")
    os.environ.pop("OTLP_ENDPOINT", None)
    return tmp_path


@pytest.fixture
def make_assistant(workdir):
    def factory(script: list, *, privacy_mode: str = "identifiers", memory: FakeMemory | None = None):
        settings = Settings(
            data_dir=workdir / ".data",
            projects_dir=workdir / "data/projects",
            persona_dir=workdir / "data/persona",
            docs_dir=workdir / "data/docs",
            samples_dir=workdir / "data/samples",
            privacy_mode=privacy_mode,
            _env_file=None,
        )
        model = FakeChatModel(script=script)
        assistant = Assistant(settings, llm=model, fast_llm=model, memory=memory or FakeMemory())
        assistant.model = model
        return assistant

    return factory
