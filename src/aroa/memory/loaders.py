"""Read .md, .txt and .pdf files into chunks."""

import hashlib
import re
from pathlib import Path

from .chunking import Chunk, chunk_document

EXTENSIONS = {".md", ".txt", ".pdf"}


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def title_of(text: str, path: Path) -> str:
    match = re.search(r"^\s*#\s+(.+)$", text, re.MULTILINE)
    return match.group(1).strip() if match else path.stem.replace("_", " ").replace("-", " ").strip()


def load_file(path: Path, max_chars: int) -> list[Chunk]:
    chunks: list[Chunk] = []
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        pages = [(i + 1, page.extract_text() or "") for i, page in enumerate(reader.pages)]
        title = title_of(pages[0][1] if pages else "", path)
        for number, text in pages:
            chunks += chunk_document(
                text, title=title, source=path.name, max_chars=max_chars, page=number, start_index=len(chunks)
            )
    else:
        text = path.read_text(encoding="utf-8", errors="replace")
        chunks = chunk_document(text, title=title_of(text, path), source=path.name, max_chars=max_chars)
    digest = file_hash(path)
    for chunk in chunks:
        chunk.file_hash = digest
    return chunks


def document_files(folder: Path) -> list[Path]:
    return sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in EXTENSIONS)
