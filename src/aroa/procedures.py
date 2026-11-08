"""Procedures: the firm's small flows, written as a document, followed by the code.

A procedure (for example "reclamación patrimonial") is one document with four sections: the template to
use, the data it needs, how to write it and how to review it. No workflow engine: the data is a form,
`{{Campo}}` placeholders in the template are filled in code, and only the passages marked
`[[instructions]]` are written by the model, one call each, with the data and the firm's instructions as
data blocks. The review pass then checks the procedure's checklist item by item, each verdict backed by
a verbatim quote or reported as not met. Procedures live in their own folder, are edited in the app and
are never indexed as evidence."""

from __future__ import annotations

import json
import re
import shutil
import unicodedata
from dataclasses import asdict, dataclass, field
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage

from . import prompts
from .documents import html_to_text, safe_name

PLACEHOLDER = re.compile(r"\{\{\s*([^{}]+?)\s*\}\}")
BLOCK = re.compile(r"\[\[\s*(.+?)\s*\]\]", re.S)
SECTIONS = {
    "plantilla": "template",
    "template": "template",
    "datos": "fields",
    "datos necesarios": "fields",
    "data": "fields",
    "redaccion": "instructions",
    "como rellenarla": "instructions",
    "writing": "instructions",
    "revision": "checklist",
    "lista de revision": "checklist",
    "review": "checklist",
}


def _fold(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    return "".join(c for c in text if not unicodedata.combining(c)).lower().strip(" .:")


def slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", _fold(name)).strip("-") or "procedimiento"


@dataclass
class Field:
    name: str
    hint: str = ""


@dataclass
class Procedure:
    slug: str
    name: str
    description: str = ""
    template_ref: str = ""  # a document of the library, when the template is not inline
    template_text: str = ""  # inline template (Markdown-like text)
    fields: list[Field] = field(default_factory=list)
    instructions: str = ""
    checklist: list[str] = field(default_factory=list)
    source: str = ""  # file name in the procedures folder

    def as_dict(self) -> dict:
        return asdict(self)


def parse(text: str, slug: str, source: str = "") -> Procedure:
    """Sections by heading name; lists become fields or checklist items; the rest is prose."""
    lines = text.replace("\r", "").split("\n")
    name, description, current = "", [], "description"
    buckets: dict[str, list[str]] = {"template": [], "fields": [], "instructions": [], "checklist": []}
    for line in lines:
        m = re.match(r"^(#{1,3})\s+(.*)$", line)
        if m:
            if m.group(1) == "#" and not name:
                name = m.group(2).strip()
                continue
            key = SECTIONS.get(_fold(m.group(2)))
            current = key or "other"
            continue
        if current == "description":
            description.append(line)
        elif current in buckets:
            buckets[current].append(line)
    proc = Procedure(slug, name or slug, "\n".join(description).strip(), source=source)
    template = "\n".join(buckets["template"]).strip()
    ref = re.match(r"^(?:documento|document)\s*:\s*(\S+)\s*$", template, re.I | re.M)
    if ref and len(template.splitlines()) <= 2:
        proc.template_ref = ref.group(1)
    else:
        proc.template_text = template
    for raw in buckets["fields"]:
        m = re.match(r"^\s*(?:[-*•]|\d+[.)])\s+(.*)$", raw)
        if m:
            label, _, hint = m.group(1).partition(":")
            proc.fields.append(Field(label.strip().strip("*"), hint.strip()))
    proc.instructions = "\n".join(buckets["instructions"]).strip()
    for raw in buckets["checklist"]:
        m = re.match(r"^\s*(?:[-*•]|\d+[.)]|\[[ x]\])\s+(.*)$", raw)
        if m and m.group(1).strip():
            proc.checklist.append(re.sub(r"^\[[ xX]\]\s*", "", m.group(1).strip()))
    return proc


def read_procedure(path: Path) -> Procedure:
    text = path.read_text(encoding="utf-8", errors="replace")
    if path.suffix.lower() in (".html", ".htm"):
        text = html_to_text(text)
    return parse(text, slugify(path.stem), path.name)


def load_all(folder: Path, samples: Path | None = None) -> list[Procedure]:
    """Every procedure in the folder (the samples are copied in on first run, like the documents)."""
    if not folder.exists() and samples is not None and samples.exists():
        shutil.copytree(samples, folder)
    folder.mkdir(parents=True, exist_ok=True)
    out = []
    for path in sorted(folder.iterdir()):
        if path.is_file() and path.suffix.lower() in (".md", ".txt", ".html", ".htm") and not path.name.startswith("."):
            out.append(read_procedure(path))
    return out


def find(folder: Path, slug: str, samples: Path | None = None) -> Procedure | None:
    return next((p for p in load_all(folder, samples) if p.slug == slug), None)


def save(folder: Path, name: str, html: str) -> Path:
    """A procedure written in the editor; saved as HTML next to the Markdown samples."""
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / safe_name(name)
    target.write_text(html, encoding="utf-8")
    return target


# ── filling ──────────────────────────────────────────────────────────────
def fill(template: str, data: dict[str, str]) -> tuple[str, list[str]]:
    """`{{Campo}}` placeholders replaced from the data (names matched without accents or case).
    Returns the text and the placeholders that had no value."""
    table = {_fold(k): str(v).strip() for k, v in data.items() if str(v).strip()}
    missing: list[str] = []

    def sub(m: re.Match) -> str:
        key = m.group(1).strip()
        value = table.get(_fold(key))
        if value is None:
            missing.append(key)
            return m.group(0)
        return value

    return PLACEHOLDER.sub(sub, template), sorted(set(missing))


def write_blocks(
    text: str, *, llm, procedure: Procedure, data: dict[str, str], guard=None, log=None
) -> tuple[str, int]:
    """Every `[[instruction]]` passage written by the model, one call each, with the data and the firm's
    instructions as data; a reply that is empty or still holds placeholders or brackets is rejected."""
    from contextlib import nullcontext

    say = log or (lambda _l: None)
    guard = guard if guard is not None else nullcontext()
    blocks = list(BLOCK.finditer(text))
    out, position, written = [], 0, 0
    facts = "\n".join(f"- {k}: {v}" for k, v in data.items() if str(v).strip())
    for i, m in enumerate(blocks, start=1):
        instruction = re.sub(r"\s+", " ", m.group(1)).strip()
        before = ("".join(out) + text[position : m.start()])[-600:]  # what is already written, blocks included
        after = text[m.end() : m.end() + 300]
        say(f"Redactando bloque {i}/{len(blocks)}: {instruction[:90]}")
        passage = ""
        for _attempt in range(2):
            with guard:
                reply = llm.invoke(
                    [
                        SystemMessage(prompts.PROCEDURE_BLOCK),
                        HumanMessage(
                            prompts.data_block(
                                "procedure",
                                procedure.instructions or "(sin instrucciones generales)",
                                name=procedure.name,
                            )
                            + "\n\n"
                            + prompts.data_block("facts", facts or "(sin datos)")
                            + "\n\n"
                            + prompts.data_block("context", f"{before}\n[[…]]\n{after}")
                            + f"\n\nWrite the passage for: {instruction}"
                        ),
                    ]
                )
            passage = str(reply.content).strip().strip("[]").strip()
            if re.search(r"(?:[A-ZÁÉÍÓÚÑ]{3,}|[IVXLC]+|\d+)\.\s*$", before):  # the paragraph label is already there
                passage = re.sub(r"^(?:[A-ZÁÉÍÓÚÑ]{3,}|[IVXLC]+|\d+)\.\s+", "", passage)
            if passage and not PLACEHOLDER.search(passage) and not BLOCK.search(passage):
                break
            passage = ""
        if not passage:
            say(f"Bloque {i}: el modelo no ha devuelto un texto válido; se deja la instrucción marcada")
            passage = f"[[{instruction}]]"
        else:
            written += 1
        out.append(text[position : m.start()])
        out.append(passage)
        position = m.end()
    out.append(text[position:])
    return "".join(out), written


def check_data_present(text: str, data: dict[str, str]) -> list[str]:
    """Data the user typed that appears nowhere in the draft, verbatim: a field the template never used."""
    from .citations import normalize

    body = normalize(text)
    return [k for k, v in data.items() if str(v).strip() and normalize(str(v)) not in body]


def draft(
    procedure: Procedure,
    data: dict[str, str],
    *,
    llm,
    template_text: str,
    guard=None,
    log=None,
) -> dict:
    """The filled document (Markdown-like text) and what to look at: missing data, blocks left, data not found."""
    say = log or (lambda _l: None)
    if "\n---\n" in template_text:  # what the template says about itself, above the rule, is not part of the document
        template_text = template_text.split("\n---\n", 1)[1].lstrip("\n")
    say(f"Procedimiento «{procedure.name}»: plantilla de {len(template_text):,} caracteres, {len(data)} datos")
    text, missing = fill(template_text, data)
    say(
        f"Campos sustituidos en código: {len(PLACEHOLDER.findall(template_text)) - len(missing)} · sin valor: {', '.join(missing) if missing else 'ninguno'}"
    )
    blocks = len(BLOCK.findall(text))
    written = 0
    if blocks and llm is not None:
        say(f"Bloques a redactar por el modelo: {blocks}")
        text, written = write_blocks(text, llm=llm, procedure=procedure, data=data, guard=guard, log=log)
    elif blocks:
        say(f"Bloques a redactar: {blocks} (sin modelo, quedan marcados)")
    unused = check_data_present(text, data)
    say(
        "Datos en el borrador: "
        + ("todos aparecen" if not unused else "no aparecen literalmente: " + ", ".join(unused))
    )
    return {
        "text": text,
        "missing_fields": missing,
        "blocks": blocks,
        "blocks_written": written,
        "data_unused": unused,
    }


def spec_dump(data: dict) -> str:
    return json.dumps(data, ensure_ascii=False)
