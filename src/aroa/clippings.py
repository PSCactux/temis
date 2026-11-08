"""Clippings: a passage cut out of a document and kept as a small document of its own.

A clipping is a Markdown file in the documents folder, so everything that works for documents works
for it: indexing, retrieval, citations that open it, deletion. Its header records where it came from,
and the quoted text is copied verbatim, so a citation to a clipping is also a citation to its source."""

from __future__ import annotations

import re
import uuid
from datetime import datetime
from pathlib import Path

PREFIX = "recorte-"


def is_clipping(source: str) -> bool:
    return source.startswith(PREFIX)


def write(docs_dir: Path, *, source: str, source_title: str, text: str, note: str = "", matter: str = "") -> str:
    """Save a clipping file and return its source name (the file name, as the index knows it)."""
    text = text.strip()
    if not text:
        raise ValueError("a clipping needs some text")
    stem = re.sub(r"[^a-z0-9]+", "-", source_title.lower()).strip("-")[:40] or "documento"
    name = f"{PREFIX}{stem}-{uuid.uuid4().hex[:6]}.md"
    first = re.sub(r"\s+", " ", text)[:70]
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    lines = [
        f"# Recorte · {source_title}: «{first}{'…' if len(text) > 70 else ''}»",
        "",
        f"> Recorte de «{source_title}» (`{source}`), guardado el {stamp}."
        + (f" Expediente {matter}." if matter else ""),
        "",
    ]
    if note:
        lines += [f"**Nota:** {note}", ""]
    lines += ["## Texto", "", text, ""]
    docs_dir.mkdir(parents=True, exist_ok=True)
    (docs_dir / name).write_text("\n".join(lines), encoding="utf-8")
    return name
