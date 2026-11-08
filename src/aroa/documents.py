"""Documents the firm edits: HTML in the editor, Word in and out, plain text for the index, a version on every save.

The editor works on HTML and the firm's own documents are stored as `.html` in the documents folder, next
to the `.md`, `.txt`, `.pdf` and `.docx` files that are only read. A `.docx` opens in the editor through
`mammoth` (headings, lists, tables and emphasis survive; page layout does not) and any document exports to
`.docx` through `python-docx` with the same structural subset. The index never sees HTML: `html_to_text`
turns it into the Markdown-like text the chunker already understands, so headings stay sections and citations
keep opening at the right place. Saving over an existing document first copies it to `.versions/<name>/`."""

from __future__ import annotations

import html
import re
import shutil
from datetime import UTC, datetime
from html.parser import HTMLParser
from io import BytesIO
from pathlib import Path

VERSIONS_DIR = ".versions"
HEADINGS = {"h1": 1, "h2": 2, "h3": 3, "h4": 3, "h5": 3, "h6": 3}
BLOCKS = {"p", "div", "li", "blockquote", "pre", "tr", "section", "article", "header", "footer"} | set(HEADINGS)
INLINE_BOLD = {"b", "strong"}
INLINE_ITALIC = {"i", "em"}
INLINE_UNDERLINE = {"u", "ins"}
INLINE_STRIKE = {"s", "del", "strike"}


def safe_name(name: str, suffix: str = ".html") -> str:
    """A file name that stays inside the documents folder and ends with `suffix`."""
    stem = Path(name).name.strip()
    if stem.lower().endswith(suffix):
        stem = stem[: -len(suffix)]
    stem = re.sub(r"[^\w.\- áéíóúÁÉÍÓÚñÑüÜçÇ]+", "", stem).strip(" .")
    stem = re.sub(r"\s+", "_", stem)
    if not stem:
        raise ValueError("empty document name")
    return stem + suffix


# ── HTML → text (what the index and the review see) ─────────────────────
class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.blocks: list[str] = []
        self.current: list[str] = []
        self.list_stack: list[tuple[str, int]] = []  # (kind, counter)
        self.row: list[str] | None = None
        self.cell: list[str] | None = None
        self.quote = 0
        self.skip = 0

    def _flush(self) -> None:
        text = re.sub(r"[ \t]+", " ", "".join(self.current)).strip()
        self.current = []
        if text:
            self.blocks.append(("> " if self.quote else "") + text)

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self.skip += 1
        elif tag in HEADINGS:
            self._flush()
            self.current.append("#" * HEADINGS[tag] + " ")
        elif tag in ("ul", "ol"):
            self._flush()
            self.list_stack.append((tag, 0))
        elif tag == "li":
            self._flush()
            if self.list_stack:
                kind, n = self.list_stack[-1]
                self.list_stack[-1] = (kind, n + 1)
                self.current.append("  " * (len(self.list_stack) - 1) + ("- " if kind == "ul" else f"{n + 1}. "))
        elif tag == "blockquote":
            self._flush()
            self.quote += 1
        elif tag == "tr":
            self.row = []
        elif tag in ("td", "th"):
            self.cell = []
        elif tag == "br":
            (self.cell if self.cell is not None else self.current).append("\n")
        elif tag in BLOCKS or tag == "table":
            self._flush()

    def handle_endtag(self, tag):
        if tag in ("script", "style"):
            self.skip = max(0, self.skip - 1)
        elif tag in ("td", "th") and self.cell is not None and self.row is not None:
            self.row.append(re.sub(r"\s+", " ", "".join(self.cell)).strip())
            self.cell = None
        elif tag == "tr" and self.row is not None:
            self.blocks.append(" | ".join(self.row))
            self.row = None
        elif tag in ("ul", "ol"):
            self._flush()
            if self.list_stack:
                self.list_stack.pop()
        elif tag == "blockquote":
            self._flush()
            self.quote = max(0, self.quote - 1)
        elif tag in BLOCKS or tag == "table":
            self._flush()

    def handle_data(self, data):
        if self.skip:
            return
        target = self.cell if self.cell is not None else self.current
        target.append(data)

    def text(self) -> str:
        self._flush()
        out: list[str] = []
        for block in self.blocks:
            out.append(block)
        text = "\n\n".join(out)
        return re.sub(r"\n{3,}", "\n\n", text).strip() + "\n"


def html_to_text(html_text: str) -> str:
    parser = _TextExtractor()
    parser.feed(html_text or "")
    return parser.text()


# ── Word in ──────────────────────────────────────────────────────────────
def docx_to_html(path: Path) -> str:
    import mammoth

    with open(path, "rb") as fh:
        return mammoth.convert_to_html(fh).value


def docx_to_text(path: Path) -> str:
    return html_to_text(docx_to_html(path))


# ── Word out ─────────────────────────────────────────────────────────────
class _DocxBuilder(HTMLParser):
    """Headings, paragraphs, emphasis, lists, quotes, tables and page breaks; the rest is text."""

    def __init__(self, document) -> None:
        super().__init__(convert_charrefs=True)
        from docx.enum.text import WD_ALIGN_PARAGRAPH

        self.align = {
            "center": WD_ALIGN_PARAGRAPH.CENTER,
            "right": WD_ALIGN_PARAGRAPH.RIGHT,
            "justify": WD_ALIGN_PARAGRAPH.JUSTIFY,
        }
        self.doc = document
        self.para = None
        self.fmt = {"bold": 0, "italic": 0, "underline": 0, "strike": 0}
        self.list_stack: list[str] = []
        self.pending_style: str | None = None
        self.pending_align = None
        self.table: list[list[list[tuple[str, dict]]]] | None = None
        self.skip = 0

    # paragraphs
    def _start_block(self, style: str | None = None, align=None) -> None:
        self.para = None
        self.pending_style, self.pending_align = style, align

    def _paragraph(self):
        if self.para is None:
            self.para = (
                self.doc.add_paragraph(style=self.pending_style) if self.pending_style else self.doc.add_paragraph()
            )
            if self.pending_align is not None:
                self.para.alignment = self.pending_align
            self.pending_style, self.pending_align = None, None
        return self.para

    def _run(self, text: str) -> None:
        if self.table is not None:
            if self.table and self.table[-1]:
                self.table[-1][-1].append((text, dict(self.fmt)))
            return
        if not text.strip() and self.para is None:
            return
        run = self._paragraph().add_run(text)
        run.bold = bool(self.fmt["bold"]) or None
        run.italic = bool(self.fmt["italic"]) or None
        run.underline = bool(self.fmt["underline"]) or None
        run.font.strike = bool(self.fmt["strike"]) or None

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag in ("script", "style"):
            self.skip += 1
        elif self.table is not None:
            if tag == "tr":
                self.table.append([])
            elif tag in ("td", "th"):
                self.table[-1].append([])
                if tag == "th":
                    self.fmt["bold"] += 1
            elif tag == "br":
                self._run("\n")
            elif tag in INLINE_BOLD:
                self.fmt["bold"] += 1
            elif tag in INLINE_ITALIC:
                self.fmt["italic"] += 1
            elif tag in INLINE_UNDERLINE:
                self.fmt["underline"] += 1
        elif tag == "table":
            self.table = []
        elif tag in HEADINGS:
            self._start_block(f"Heading {HEADINGS[tag]}")
        elif tag in ("ul", "ol"):
            self.list_stack.append(tag)
            self.para = None
        elif tag == "li":
            kind = self.list_stack[-1] if self.list_stack else "ul"
            self._start_block("List Bullet" if kind == "ul" else "List Number")
        elif tag == "blockquote":
            self._start_block("Quote")
        elif tag in ("p", "div", "section", "article"):
            if "mce-pagebreak" in a.get("class", "") or a.get("style", "").replace(" ", "").startswith("page-break"):
                self.doc.add_page_break()
                self.para = None
                return
            align = None
            m = re.search(r"text-align:\s*(\w+)", a.get("style", ""))
            if m:
                align = self.align.get(m.group(1))
            self._start_block(
                self.pending_style if self.para is None else None, align
            )  # a <p> inside a quote or item keeps its style
        elif tag == "br":
            if self.para is not None:
                self.para.add_run().add_break()
        elif tag == "hr":
            self.doc.add_page_break()
            self.para = None
        elif tag in INLINE_BOLD:
            self.fmt["bold"] += 1
        elif tag in INLINE_ITALIC:
            self.fmt["italic"] += 1
        elif tag in INLINE_UNDERLINE:
            self.fmt["underline"] += 1
        elif tag in INLINE_STRIKE:
            self.fmt["strike"] += 1

    def handle_endtag(self, tag):
        if tag in ("script", "style"):
            self.skip = max(0, self.skip - 1)
        elif tag == "table" and self.table is not None:
            rows = [r for r in self.table if r]
            self.table = None
            if rows:
                cols = max(len(r) for r in rows)
                table = self.doc.add_table(rows=len(rows), cols=cols)
                table.style = "Table Grid"
                for i, row in enumerate(rows):
                    for j, cell in enumerate(row):
                        paragraph = table.cell(i, j).paragraphs[0]
                        for text, fmt in cell:
                            if text == "\n":
                                paragraph.add_run().add_break()
                                continue
                            run = paragraph.add_run(text)
                            run.bold = bool(fmt["bold"]) or None
                            run.italic = bool(fmt["italic"]) or None
                            run.underline = bool(fmt["underline"]) or None
                self.para = None
        elif tag == "th":
            self.fmt["bold"] = max(0, self.fmt["bold"] - 1)
        elif tag in ("ul", "ol"):
            if self.list_stack:
                self.list_stack.pop()
            self.para = None
        elif tag in BLOCKS:
            self.para = None
        elif tag in INLINE_BOLD:
            self.fmt["bold"] = max(0, self.fmt["bold"] - 1)
        elif tag in INLINE_ITALIC:
            self.fmt["italic"] = max(0, self.fmt["italic"] - 1)
        elif tag in INLINE_UNDERLINE:
            self.fmt["underline"] = max(0, self.fmt["underline"] - 1)
        elif tag in INLINE_STRIKE:
            self.fmt["strike"] = max(0, self.fmt["strike"] - 1)

    def handle_data(self, data):
        if self.skip:
            return
        text = re.sub(r"\s+", " ", data)
        if text:
            self._run(text)


def html_to_docx(html_text: str, title: str | None = None) -> bytes:
    from docx import Document
    from docx.shared import Pt

    document = Document()
    style = document.styles["Normal"]
    style.font.name = "Times New Roman"
    style.font.size = Pt(12)
    if title:
        document.core_properties.title = title
    builder = _DocxBuilder(document)
    builder.feed(html_text or "")
    out = BytesIO()
    document.save(out)
    return out.getvalue()


# ── saving with versions ─────────────────────────────────────────────────
def _stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")


def save_html(docs_dir: Path, name: str, html_text: str) -> Path:
    """Write the document; the previous content, if any, is kept as a version."""
    target = docs_dir / safe_name(name)
    docs_dir.mkdir(parents=True, exist_ok=True)
    if target.exists():
        folder = docs_dir / VERSIONS_DIR / target.name
        folder.mkdir(parents=True, exist_ok=True)
        shutil.copy2(target, folder / f"{_stamp()}.html")
    target.write_text(html_text, encoding="utf-8")
    return target


def versions(docs_dir: Path, name: str) -> list[dict]:
    folder = docs_dir / VERSIONS_DIR / Path(name).name
    if not folder.is_dir():
        return []
    out = []
    for path in sorted(folder.glob("*.html"), reverse=True):
        stamp = path.stem
        try:
            when = datetime.strptime(stamp, "%Y%m%dT%H%M%S%fZ").replace(tzinfo=UTC).isoformat()
        except ValueError:
            when = stamp
        out.append({"version": stamp, "saved_at": when, "bytes": path.stat().st_size})
    return out


def version_html(docs_dir: Path, name: str, version: str) -> str | None:
    path = docs_dir / VERSIONS_DIR / Path(name).name / (re.sub(r"[^\w]", "", version) + ".html")
    return path.read_text(encoding="utf-8") if path.is_file() else None


def restore(docs_dir: Path, name: str, version: str) -> Path | None:
    """Make an earlier version the current document (the current one becomes a version first)."""
    content = version_html(docs_dir, name, version)
    if content is None:
        return None
    return save_html(docs_dir, name, content)


def text_to_html(text: str) -> str:
    """Plain or Markdown-ish text as simple HTML for the editor: headings, lists, quotes, paragraphs."""
    out: list[str] = []
    list_tag: str | None = None
    para: list[str] = []

    def flush() -> None:
        nonlocal list_tag
        if para:
            out.append("<p>" + "<br>".join(html.escape(p) for p in para) + "</p>")
            para.clear()
        if list_tag:
            out.append(f"</{list_tag}>")
            list_tag = None

    for raw in (text or "").replace("\r", "").split("\n"):
        line = raw.rstrip()
        if not line.strip():
            flush()
            continue
        m = re.match(r"^(#{1,3})\s+(.*)", line)
        if m:
            flush()
            out.append(f"<h{len(m.group(1))}>{html.escape(m.group(2))}</h{len(m.group(1))}>")
            continue
        m = re.match(r"^\s*[-*•]\s+(.*)", line)
        if m:
            if list_tag != "ul":
                flush()
                list_tag = "ul"
                out.append("<ul>")
            out.append(f"<li>{html.escape(m.group(1))}</li>")
            continue
        m = re.match(r"^\s*\d+[.)]\s+(.*)", line)
        if m:
            if list_tag != "ol":
                flush()
                list_tag = "ol"
                out.append("<ol>")
            out.append(f"<li>{html.escape(m.group(1))}</li>")
            continue
        m = re.match(r"^\s*>\s?(.*)", line)
        if m:
            flush()
            out.append(f"<blockquote><p>{html.escape(m.group(1))}</p></blockquote>")
            continue
        if list_tag:
            flush()
        para.append(line)
    flush()
    body = "\n".join(out)
    return re.sub(r"\*\*([^*\n]+?)\*\*", r"<strong>\1</strong>", body)
