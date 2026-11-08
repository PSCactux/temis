"""Legal citations in a draft, checked against the firm's archive of the BOE, in code.

A brief cites "artículo 135.5 de la LEC" or "art. 9 de la Ley 29/1994". The model is not asked whether
that is right: the citation is extracted with patterns, the norm is located in the consolidated
legislation stored by the BOE archive, and the article's current text is read from it. Each result
says found / not found / not in the archive / not verifiable, with the official URL and when the
archive fetched the text. Case law (STS, ECLI) is recognised but not verified: the firm's archive holds
no judgments, and the public case-law database forbids bulk reuse."""

from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

LAW_REF = r"(?:Ley\s+Orgánica|Ley|Real\s+Decreto(?:-ley|\s+Legislativo)?|Decreto-ley|Decreto\s+Legislativo)\s+\d+/\d{4}"
ARTICLE = re.compile(
    r"\b(?:art(?:ículos?|ícul|s?\.)|artículos?)\s*"
    r"((?:\d+(?:\.\d+)?(?:\s*(?:bis|ter|quáter|quater|quinquies))?)(?:\s*(?:,|y|e)\s*\d+(?:\.\d+)?(?:\s*(?:bis|ter))?)*)"
    r"\s*(?:,\s*)?(?:de\s+la|del|de\s+el|de)?\s*(?P<law>"
    + LAW_REF
    + r"|[A-ZÁÉÍÓÚ][\w .áéíóúÁÉÍÓÚ-]{0,70}?)?(?=[.,;:()\n?!¿¡»\"']|\s+(?:y|e|o|que|en|para|por|con)\s|$)",
    re.IGNORECASE,
)
JUDGMENT = re.compile(
    r"\b(STS|STC|STSJ|SAP|SAN|ATS|ATC)\s*(?:n\.?º|núm\.?|número)?\s*\d+/\d{4}\b|\bECLI:ES:[A-Z]{2,4}:\d{4}:\d+[A-Z]?\b",
    re.IGNORECASE,
)
LAW_NUMBER = re.compile(LAW_REF, re.IGNORECASE)


@dataclass(frozen=True)
class Citation:
    raw: str
    kind: str  # article | law | judgment
    article: str | None = None  # "135" (apartado kept separately)
    apartado: str | None = None  # "5" in "135.5"
    law: str | None = None  # as written: "LEC", "Ley 29/1994", "Código Civil"


@dataclass
class Verification:
    citation: Citation
    status: str  # found | article_not_found | law_not_found | law_not_in_archive | unverifiable
    law_id: str | None = None
    law_title: str | None = None
    heading: str | None = None
    text: str | None = None
    vigencia: str | None = None
    fetched_at: str | None = None
    url: str | None = None
    note: str | None = None


class LawSource(Protocol):
    def find_norm(self, reference: str) -> tuple[str, str] | None: ...  # (norm id, title), or None
    def is_stored(self, norm_id: str) -> bool: ...
    def article(self, norm_id: str, number: str) -> dict | None: ...  # heading, text, vigencia, fetched_at
    def articles(self, norm_id: str) -> list[dict]: ...  # every current article: number, heading, text


@dataclass
class Aliases:
    """Known norms by abbreviation or short name, from data/laws.json."""

    by_alias: dict[str, tuple[str, str]] = field(default_factory=dict)  # lowercase alias -> (id, title)

    @classmethod
    def load(cls, path: Path) -> Aliases:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        table: dict[str, tuple[str, str]] = {}
        for norm_id, body in data.get("laws", {}).items():
            for alias in body.get("aliases", []):
                table[_norm(alias)] = (norm_id, body["title"])
        return cls(table)

    def resolve(self, reference: str) -> tuple[str, str] | None:
        return self.by_alias.get(_norm(reference))


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().strip(".,;:()").lower())


CONNECTORS = {"de", "del", "la", "las", "los", "el", "y", "e"}


def trim_law_name(raw: str) -> str:
    """«LEC procede» → «LEC»; «Código Civil procede» → «Código Civil»; «Ley 29/1994» stays. A norm's name is capitalised
    words, numbers and connectors; the first ordinary lowercase word ends it."""
    if not raw:
        return ""
    if LAW_NUMBER.match(raw):
        return LAW_NUMBER.match(raw).group(0)
    kept: list[str] = []
    for word in raw.split():
        if word[0].isupper() or word[0].isdigit() or word.lower() in CONNECTORS:
            kept.append(word)
        else:
            break
    while kept and kept[-1].lower() in CONNECTORS:
        kept.pop()
    return " ".join(kept)


def extract_citations(text: str) -> list[Citation]:
    """Every article, law and judgment the text cites, in order, without duplicates."""
    found: list[Citation] = []
    seen: set[tuple] = set()

    def add(c: Citation) -> None:
        key = (c.kind, c.article, c.apartado, _norm(c.law or ""), c.raw if c.kind == "judgment" else "")
        if key not in seen:
            seen.add(key)
            found.append(c)

    for m in ARTICLE.finditer(text):
        law = trim_law_name((m.group("law") or "").strip())
        for number in re.split(r"\s*(?:,|\by\b|\be\b)\s*", m.group(1)):
            number = number.strip()
            if not number:
                continue
            base, _, apartado = number.partition(".")
            add(
                Citation(
                    raw=m.group(0).strip(),
                    kind="article",
                    article=base.replace(" ", ""),
                    apartado=apartado or None,
                    law=law or None,
                )
            )
    for m in JUDGMENT.finditer(text):
        add(Citation(raw=m.group(0), kind="judgment"))
    for m in LAW_NUMBER.finditer(text):
        add(Citation(raw=m.group(0), kind="law", law=re.sub(r"\s+", " ", m.group(0))))
    return found


def verify(citations: list[Citation], source: LawSource, aliases: Aliases) -> list[Verification]:
    out: list[Verification] = []
    for c in citations:
        if c.kind == "judgment":
            out.append(
                Verification(
                    c,
                    "unverifiable",
                    note="jurisprudencia: no hay archivo local de sentencias (CENDOJ no permite la reutilización masiva); comprobar a mano",
                )
            )
            continue
        if not c.law:
            out.append(Verification(c, "law_not_found", note="la cita no dice de qué norma es el artículo"))
            continue
        resolved = aliases.resolve(c.law) or source.find_norm(c.law)
        if not resolved:
            out.append(Verification(c, "law_not_found", note=f"ninguna norma consolidada del BOE se llama «{c.law}»"))
            continue
        norm_id, title = resolved
        url = f"https://www.boe.es/buscar/act.php?id={norm_id}"
        if c.kind == "law":
            out.append(Verification(c, "found", norm_id, title, url=url))
            continue
        if not source.is_stored(norm_id):
            out.append(
                Verification(
                    c,
                    "law_not_in_archive",
                    norm_id,
                    title,
                    url=url,
                    note=f"descárgala con `boe consolidada fetch {norm_id}` y repite",
                )
            )
            continue
        art = source.article(norm_id, c.article or "")
        if art is None:
            out.append(
                Verification(
                    c,
                    "article_not_found",
                    norm_id,
                    title,
                    url=url,
                    note=f"la norma existe pero no tiene artículo {c.article}",
                )
            )
            continue
        note = None
        text = art["text"]
        if c.apartado:
            m = re.search(rf"(?ms)^\s*{re.escape(c.apartado)}\.\s.*?(?=^\s*\d+\.\s|\Z)", text)
            if m:
                text = f"{art['heading'] or ''}\n{m.group(0).strip()}"  # the cited apartado, not the article's start
            else:
                note = f"el artículo existe pero no se ve un apartado {c.apartado} en su texto vigente"
        out.append(
            Verification(
                c,
                "found",
                norm_id,
                title,
                art["heading"],
                text,
                art.get("vigencia"),
                art.get("fetched_at"),
                f"{url}#a{c.article}",
                note,
            )
        )
    return out


LABELS = {
    "found": "✔ existe",
    "article_not_found": "✘ no existe el artículo",
    "law_not_found": "✘ norma no encontrada",
    "law_not_in_archive": "? norma no descargada en el archivo",
    "unverifiable": "? no verificable automáticamente",
}


def format_report(results: list[Verification], *, snippet: int = 280) -> str:
    """Plain text for the model and the user: one block per citation, verdict first."""
    if not results:
        return "No se han encontrado citas legales (artículos, leyes o sentencias) en el texto."
    lines = [f"{len(results)} citas comprobadas contra el archivo del BOE:"]
    for v in results:
        head = f"- {LABELS[v.status]} · {v.citation.raw}"
        if v.law_title:
            head += f" → {v.law_title}"
        lines.append(head)
        if v.heading:
            lines.append(f"    {v.heading}" + (f" (vigente desde {v.vigencia})" if v.vigencia else ""))
        if v.text:
            body = re.sub(r"\s+", " ", v.text)
            if v.heading and body.startswith(v.heading):
                body = body[len(v.heading) :].strip().lstrip(".:- ").strip()
            lines.append(f"    «{body[:snippet]}{'…' if len(body) > snippet else ''}»")
        if v.note:
            lines.append(f"    {v.note}")
        if v.url:
            lines.append(f"    {v.url}" + (f" · texto descargado el {v.fetched_at[:10]}" if v.fetched_at else ""))
    bad = sum(1 for v in results if v.status in ("article_not_found", "law_not_found"))
    unknown = sum(1 for v in results if v.status in ("law_not_in_archive", "unverifiable"))
    lines.append(f"Resumen: {len(results) - bad - unknown} correctas, {bad} incorrectas, {unknown} sin verificar.")
    return "\n".join(lines)


class BoeArchive:
    """The BOE archive's catalog (a sibling project) read directly from its SQLite file."""

    def __init__(self, data_dir: Path):
        path = Path(data_dir) / "catalog.sqlite"
        if not path.exists():
            raise FileNotFoundError(f"no BOE archive catalog at {path}")
        self.conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=30)
        self.conn.row_factory = sqlite3.Row

    def find_norm(self, reference: str) -> tuple[str, str] | None:
        ref = re.sub(r"\s+", " ", reference.strip())
        for table in ("norma", "consolidada_item"):
            row = self.conn.execute(
                f"SELECT norma_id, titulo FROM {table} WHERE titulo LIKE ? ORDER BY fecha_actualizacion DESC LIMIT 1",
                (ref + ",%",),
            ).fetchone()
            if row:
                return row["norma_id"], row["titulo"]
        return None

    def is_stored(self, norm_id: str) -> bool:
        return self.conn.execute("SELECT 1 FROM norma WHERE norma_id = ? LIMIT 1", (norm_id,)).fetchone() is not None

    def article(self, norm_id: str, number: str) -> dict | None:
        block = "a" + re.sub(r"\s+", "", number.lower())
        row = self.conn.execute(
            """SELECT v.bloque_titulo, v.text, v.fecha_vigencia, f.fetched_at FROM norma_version v
               JOIN parse_run p ON p.id = v.parse_run_id AND p.current = 1
               JOIN fetch f ON f.id = v.fetch_id
               WHERE v.norma_id = ? AND v.bloque_id = ? ORDER BY v.fecha_vigencia DESC LIMIT 1""",
            (norm_id, block),
        ).fetchone()
        if row is None:
            return None
        return {
            "heading": row["bloque_titulo"],
            "text": row["text"] or "",
            "vigencia": row["fecha_vigencia"],
            "fetched_at": row["fetched_at"],
        }

    def articles(self, norm_id: str) -> list[dict]:
        """Every article of the norm in its current parse, latest version of each, for locating a quoted text."""
        rows = self.conn.execute(
            """SELECT v.bloque_id, v.bloque_titulo, v.text FROM norma_version v
               JOIN parse_run p ON p.id = v.parse_run_id AND p.current = 1
               WHERE v.norma_id = ? AND v.bloque_id LIKE 'a%' ORDER BY v.bloque_id, v.fecha_vigencia DESC""",
            (norm_id,),
        ).fetchall()
        out: dict[str, dict] = {}
        for row in rows:
            number = row["bloque_id"][1:]
            if number not in out:
                out[number] = {"number": number, "heading": row["bloque_titulo"], "text": row["text"] or ""}
        return list(out.values())


def check_text(text: str, source: LawSource, aliases: Aliases) -> tuple[list[Verification], str]:
    results = verify(extract_citations(text), source, aliases)
    return results, format_report(results)
