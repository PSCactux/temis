"""Read .md, .txt, .pdf, .docx and .html files into chunks."""

import hashlib
import re
from pathlib import Path

from .chunking import Chunk, chunk_document

EXTENSIONS = {".md", ".txt", ".pdf", ".docx", ".html", ".htm"}


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def title_of(text: str, path: Path) -> str:
    match = re.search(r"^\s*#\s+(.+)$", text, re.MULTILINE)
    return match.group(1).strip() if match else path.stem.replace("_", " ").replace("-", " ").strip()


def read_text(path: Path) -> str:
    """The plain text of a document; PDF pages are separated by a page marker line."""
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader

        pages = [page.extract_text() or "" for page in PdfReader(str(path)).pages]
        return "\n\n".join(f"[p. {i}]\n{text}" for i, text in enumerate(pages, start=1))
    if path.suffix.lower() == ".docx":
        from ..documents import docx_to_text

        return docx_to_text(path)
    if path.suffix.lower() in (".html", ".htm"):
        from ..documents import html_to_text

        return html_to_text(path.read_text(encoding="utf-8", errors="replace"))
    return path.read_text(encoding="utf-8", errors="replace")


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
        text = read_text(path)
        chunks = chunk_document(text, title=title_of(text, path), source=path.name, max_chars=max_chars)
    digest = file_hash(path)
    for chunk in chunks:
        chunk.file_hash = digest
    return chunks


def document_files(folder: Path) -> list[Path]:
    """Every document in the folder, skipping hidden folders such as the editor's `.versions`."""
    return sorted(
        p
        for p in folder.rglob("*")
        if p.is_file()
        and p.suffix.lower() in EXTENSIONS
        and not any(part.startswith(".") for part in p.relative_to(folder).parts)
    )
