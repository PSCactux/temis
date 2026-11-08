"""Structure-aware chunking for documents.

Chunks follow headings; a section is only split further when it is longer than
`max_chars`, and then by paragraphs (or lines, or sentences) with one unit of overlap.
Every chunk keeps the document title and section so it can be cited precisely and so
the embedded text carries its context.
"""

import re
from dataclasses import asdict, dataclass
from pathlib import Path

HEADING = re.compile(
    r"^\s*(?:#{1,6}\s+.+"
    r"|(?:art[íi]culo|cl[áa]usula|secci[óo]n|cap[íi]tulo|t[íi]tulo|anexo|article|section|chapter)\s+[\w.ºª]+.*)$",
    re.IGNORECASE,
)


@dataclass
class Chunk:
    id: str
    doc_id: str
    title: str
    section: str
    text: str
    source: str
    page: int | None = None
    file_hash: str = ""

    @property
    def context_text(self) -> str:
        return f"{self.title} — {self.section}\n\n{self.text}"

    def to_payload(self) -> dict:
        return asdict(self)

    def label(self) -> str:
        page = f", p. {self.page}" if self.page and not self.section.startswith("Page") else ""
        section = "" if self.section == self.title else f" — {self.section}"
        return f"{self.title}{section}{page}"


def slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def split_sections(text: str) -> list[tuple[str, str]]:
    """(heading, body) pairs; text before the first heading gets an empty heading."""
    sections: list[tuple[str, list[str]]] = [("", [])]
    for line in text.splitlines():
        if len(line) <= 120 and HEADING.match(line):
            sections.append((line.strip().lstrip("#").strip(), []))
        else:
            sections[-1][1].append(line)
    return [(h, "\n".join(body).strip()) for h, body in sections if h or "\n".join(body).strip()]


def _units(text: str, max_chars: int) -> list[tuple[str, str]]:
    units: list[tuple[str, str]] = []
    for paragraph in (p.strip() for p in re.split(r"\n\s*\n", text)):
        if not paragraph:
            continue
        if len(paragraph) <= max_chars:
            units.append(("\n\n", paragraph))
            continue
        lines = [line.strip() for line in paragraph.split("\n") if line.strip()]
        for i, line in enumerate(lines):
            first_sep = "\n\n" if i == 0 else "\n"
            if len(line) <= max_chars:
                units.append((first_sep, line))
            else:
                sentences = [x for x in re.split(r"(?<=[.;:!?])\s+", line) if x]
                units += [(first_sep if j == 0 else " ", x) for j, x in enumerate(sentences)]
    return units


def split_long(text: str, max_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    pieces: list[list[tuple[str, str]]] = [[]]
    size = 0
    for sep, unit in _units(text, max_chars):
        current = pieces[-1]
        if current and size + len(sep) + len(unit) > max_chars:
            overlap = [current[-1]] if len(current) > 1 else []
            pieces.append([*overlap, (sep, unit)])
            size = sum(len(u) + len(s) for s, u in pieces[-1])
        else:
            current.append((sep, unit))
            size += len(sep) + len(unit)
    return ["".join(u if i == 0 else s + u for i, (s, u) in enumerate(piece)) for piece in pieces]


def chunk_document(
    text: str, *, title: str, source: str, max_chars: int, page: int | None = None, start_index: int = 0
) -> list[Chunk]:
    doc_id = slugify(Path(source).stem)
    point = slugify(Path(source).name)  # ids carry the extension: contrato.md and contrato.html are two documents
    chunks: list[Chunk] = []
    for heading, body in split_sections(text):
        if not body:
            continue
        section = heading or (f"Page {page}" if page else "Start")
        for piece in split_long(body, max_chars):
            chunks.append(
                Chunk(
                    id=f"{point}:{start_index + len(chunks)}",
                    doc_id=doc_id,
                    title=title,
                    section=section,
                    text=piece,
                    source=source,
                    page=page,
                )
            )
    return chunks
