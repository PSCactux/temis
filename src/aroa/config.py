"""Runtime settings, read from the environment and an optional `.env` file."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # "<provider>:<model>" with provider anthropic, openai (any OpenAI-compatible gateway) or ollama.
    llm_model: str = "ollama:qwen3:8b"
    # Routing, grading, claim extraction and fact extraction: short structured outputs.
    llm_fast_model: str = "ollama:qwen3:8b"
    llm_max_tokens: int = 4096
    llm_fast_max_tokens: int = 1024
    llm_reasoning: bool = False  # let thinking models think (slower; breaks structured output on small ones)

    assistant_name: str = "Aroa"
    assistant_language: Literal["en", "es"] = "en"  # the texts the code writes and the path of a turn
    persona_dir: Path = Path("data/persona")
    # off: nothing is pseudonymised; identifiers: patterns in code only; full: also names via the fast model
    privacy_mode: Literal["off", "identifiers", "full"] = "full"

    embedding_model: str = "intfloat/multilingual-e5-large"
    sparse_model: str = "Qdrant/bm25"
    sparse_language: str = "spanish"

    data_dir: Path = Path(".data")
    docs_dir: Path = Path("data/docs")
    samples_dir: Path = Path("data/samples")
    projects_dir: Path = Path("data/projects")
    qdrant_url: str | None = None  # a Qdrant server instead of the embedded store

    # Memory
    recall_facts: int = 6
    recall_episodes: int = 3
    episode_half_life_days: float = 14.0
    fact_duplicate_threshold: float = 0.9
    history_messages: int = 10

    # Documents
    chunk_max_chars: int = 1800
    top_k: int = 6
    prefetch_k: int = 20

    # Tools
    max_tool_turns: int = 12
    command_timeout_seconds: int = 60
    allowed_commands: str = "ls cat head tail wc grep find sort uniq diff cut tr stat file"  # run_command, no shell
    calendar_path: Path = Path("data/calendar/holidays.json")  # national and regional holidays, by year
    court_region: str = ""  # region code in the calendar (CT, MD...) for procedural deadlines
    local_holidays: str = ""  # comma-separated YYYY-MM-DD local holidays of the court
    boe_archive_dir: Path | None = None  # data folder of the BOE archive (sibling project) for citation checks
    laws_path: Path = Path("data/laws.json")  # abbreviations of the norms lawyers cite

    def local_holidays_list(self) -> list[str]:
        return [d.strip() for d in self.local_holidays.split(",") if d.strip()]

    search_url: str | None = None  # SearXNG-style endpoint returning JSON, e.g. http://localhost:8888
    web_max_chars: int = 12000

    # Observability
    otlp_endpoint: str | None = None  # e.g. http://localhost:6006/v1/traces (Phoenix) or any OTLP/HTTP collector
    trace_content: bool = False
    service_name: str = "aroa-assistant"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "aroa.sqlite"

    @property
    def checkpoint_db(self) -> Path:
        return self.data_dir / "checkpoints.sqlite"

    @property
    def qdrant_path(self) -> Path:
        return self.data_dir / "qdrant"


@lru_cache
def get_settings() -> Settings:
    return Settings()
