"""A draft reviewed in a pass of its own, separate from writing it.

The review takes the text of a draft and returns issues anchored to passages of the draft, each with
a reason and, when the code can tell what the text should say, a proposed change the user accepts or
rejects one by one. What is checked in code: paragraphs repeated elsewhere in the draft; quoted
passages («…») that are not verbatim in the firm's documents, with the closest original sentence as
the correction; legal citations against the BOE archive (norm and article exist, and the article cited
is the one that contains the text quoted next to it). What the model does: read the draft in windows
and point at legal issues; a pointer that does not quote the draft verbatim is dropped. Model findings
are always warnings, never errors: only the code says "wrong"."""

from __future__ import annotations

import difflib
import json
import re
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field

from langchain_core.messages import HumanMessage, SystemMessage

from . import lawcheck, prompts
from .citations import from_injection, normalize, quote_in
from .memory.chunking import Chunk
from .schemas import Issues

MIN_QUOTE_CHARS = 20
MIN_QUOTE_WORDS = 4
MIN_PARAGRAPH_WORDS = 8
REPEAT_SIMILARITY = 0.85  # Jaccard of content words above which two paragraphs are the same paragraph
CLOSEST_MATCH = 0.7  # SequenceMatcher ratio above which an original sentence is offered as the correction
WINDOW_CHARS = 5000  # draft text handed to the model per call

QUOTE = re.compile(r"«([^«»]+?)»|[“\"]([^“”\"]+?)[”\"]")
SENTENCE = re.compile(r"(?<=[.!?;:])\s+|\n+")

KINDS = {
    "parrafo_repetido": "párrafo repetido",
    "cita_no_literal": "cita no literal",
    "cita_no_encontrada": "cita no encontrada en los documentos",
    "articulo_inexistente": "artículo inexistente",
    "articulo_equivocado": "artículo equivocado",
    "norma_no_encontrada": "norma no encontrada",
    "norma_sin_descargar": "norma no descargada en el archivo",
    "jurisprudencia_sin_verificar": "jurisprudencia sin verificar",
    "cita_legal_no_literal": "cita legal no literal",
    "cuestion_legal": "cuestión legal",
}


@dataclass
class Change:
    before: str  # verbatim passage of the draft
    after: str  # what it becomes; empty deletes it
    occurrence: int = 1  # which occurrence of `before` in the draft (repeats make the same text appear twice)


@dataclass
class Issue:
    n: int
    kind: str
    severity: str  # error | aviso
    passage: str  # verbatim from the draft; the UI finds and marks it
    paragraph: int  # 1-based paragraph of the draft
    reason: str
    change: Change | None = None
    source: str | None = None  # where the original text lives (document label or BOE URL)
    quote: str | None = None  # the original text, when there is one
    file: str | None = None  # indexed document the original comes from (opens in the reader)


@dataclass
class Review:
    name: str
    paragraphs: int
    issues: list[Issue]
    quotes_checked: int
    citations_checked: int
    windows: int  # model calls for legal issues
    notes: list[str] = field(default_factory=list)
    seconds: float = 0.0

    def errors(self) -> int:
        return sum(1 for i in self.issues if i.severity == "error")

    def as_dict(self, draft: str) -> dict:
        return {**asdict(self), "draft": draft, "report": format_report(self)}


# ── the draft ──────────────────────────────────────────────────────────
def paragraphs_of(text: str) -> list[tuple[int, int, str]]:
    """(start, end, text) of every non-empty paragraph, split on blank lines."""
    out: list[tuple[int, int, str]] = []
    for m in re.finditer(r"(?:[^\n]|\n(?!\s*\n))+", text):
        body = m.group(0)
        lead = len(body) - len(body.lstrip())
        trail = len(body) - len(body.rstrip())
        if body.strip():
            out.append((m.start() + lead, m.end() - trail, body.strip()))
    return out


def paragraph_at(paragraphs: list[tuple[int, int, str]], position: int) -> int:
    for i, (start, end, _) in enumerate(paragraphs, start=1):
        if start <= position <= end:
            return i
    return max(1, len(paragraphs))


def occurrence_at(text: str, passage: str, position: int) -> int:
    return text.count(passage, 0, position) + 1


def _words(text: str) -> set[str]:
    return {w for w in re.findall(r"\w+", normalize(text)) if len(w) > 3}


# ── checks in code ─────────────────────────────────────────────────────
def find_repeats(text: str, paragraphs: list[tuple[int, int, str]]) -> list[Issue]:
    """A paragraph that says what an earlier paragraph said; the later one is the issue and the change deletes it."""
    issues: list[Issue] = []
    seen: list[tuple[int, str, set[str]]] = []
    for i, (start, _, body) in enumerate(paragraphs, start=1):
        words = _words(body)
        if len(body.split()) < MIN_PARAGRAPH_WORDS:
            continue
        norm = normalize(body)
        twin = next(
            (
                n
                for n, other_norm, other_words in seen
                if other_norm == norm
                or (words and other_words and len(words & other_words) / len(words | other_words) >= REPEAT_SIMILARITY)
            ),
            None,
        )
        if twin is not None:
            issues.append(
                Issue(
                    0,
                    "parrafo_repetido",
                    "error",
                    body,
                    i,
                    f"repite lo que ya dice el párrafo {twin}",
                    Change(body, "", occurrence_at(text, body, start)),
                )
            )
        else:
            seen.append((i, norm, words))
    return issues


def extract_quotes(text: str) -> list[tuple[str, int]]:
    """Quoted passages long enough to be a citation of a document, with the position of their text."""
    out: list[tuple[str, int]] = []
    for m in QUOTE.finditer(text):
        inner = m.group(1) or m.group(2)
        if len(inner) >= MIN_QUOTE_CHARS and len(inner.split()) >= MIN_QUOTE_WORDS:
            out.append((inner, m.start() + 1))
    return out


def _windows(sentences: list[str], size: int = 3):
    for i in range(len(sentences)):
        for j in range(i + 1, min(i + size, len(sentences)) + 1):
            yield " ".join(sentences[i:j])


def aligned(quote: str, original: str) -> str:
    """The span of `original` that corresponds to `quote`: from its first matching run to its last."""
    blocks = [
        b for b in difflib.SequenceMatcher(None, quote.lower(), original.lower()).get_matching_blocks() if b.size >= 3
    ]
    if not blocks:
        return original
    return original[blocks[0].b : blocks[-1].b + blocks[-1].size].strip(" ,;:.")


def closest_passage(quote: str, chunks: list[Chunk]) -> tuple[float, str, Chunk] | None:
    """The sentence (or run of up to three) in the chunks most similar to the quote, if similar enough."""
    target = normalize(quote)
    words = _words(quote)
    best: tuple[float, str, Chunk] | None = None
    for chunk in chunks:
        chunk_words = _words(chunk.text)
        if words and len(words & chunk_words) < len(words) * 0.5:
            continue
        sentences = [s.strip() for s in SENTENCE.split(chunk.text) if s.strip()]
        for window in _windows(sentences):
            if abs(len(window) - len(quote)) > max(len(quote), 40):
                continue
            matcher = difflib.SequenceMatcher(None, target, normalize(window))
            if matcher.quick_ratio() < CLOSEST_MATCH:
                continue
            ratio = matcher.ratio()
            if ratio >= CLOSEST_MATCH and (best is None or ratio > best[0]):
                best = (ratio, re.sub(r"\s+", " ", window).strip(), chunk)
    return best


def check_quotes(
    text: str, paragraphs, chunks: list[Chunk], *, skip: set[int] | None = None
) -> tuple[list[Issue], int]:
    """Every quoted passage must appear verbatim in some document; otherwise the closest original is the fix."""
    issues: list[Issue] = []
    quotes = extract_quotes(text)
    for quote, position in quotes:
        if skip and position in skip:
            continue
        if any(quote_in(quote, c.text) for c in chunks):
            continue
        found = closest_passage(quote, chunks)
        if found:
            ratio, original, chunk = found
            issues.append(
                Issue(
                    0,
                    "cita_no_literal",
                    "error",
                    quote,
                    paragraph_at(paragraphs, position),
                    f"la cita no es literal: en «{chunk.label()}» el texto dice otra cosa",
                    Change(quote, aligned(quote, original)),
                    chunk.label(),
                    original,
                    chunk.source,
                )
            )
        else:
            issues.append(
                Issue(
                    0,
                    "cita_no_encontrada",
                    "error",
                    quote,
                    paragraph_at(paragraphs, position),
                    "ninguno de los documentos contiene este texto ni nada parecido",
                )
            )
    return issues, len(quotes)


def check_law(
    text: str, paragraphs, source: lawcheck.LawSource, aliases: lawcheck.Aliases
) -> tuple[list[Issue], int, set[int]]:
    """Legal citations against the archive. A quote in the same paragraph as an article citation is taken as
    the text of that article (the nearest one when there are several) and must be found in it. Returns the
    issues, how many citations were checked and the positions of the quotes that were law text."""
    issues: list[Issue] = []
    law_quotes: set[int] = set()
    results = lawcheck.verify(lawcheck.extract_citations(text), source, aliases)
    found_articles: list[tuple[int, lawcheck.Verification]] = []
    for v in results:
        c = v.citation
        position = text.find(c.raw)
        paragraph = paragraph_at(paragraphs, max(position, 0))
        if v.status == "article_not_found":
            issues.append(Issue(0, "articulo_inexistente", "error", c.raw, paragraph, v.note or "", None, v.url))
        elif v.status == "law_not_found":
            issues.append(Issue(0, "norma_no_encontrada", "error", c.raw, paragraph, v.note or ""))
        elif v.status == "law_not_in_archive":
            issues.append(Issue(0, "norma_sin_descargar", "aviso", c.raw, paragraph, v.note or "", None, v.url))
        elif v.status == "unverifiable":
            issues.append(Issue(0, "jurisprudencia_sin_verificar", "aviso", c.raw, paragraph, v.note or ""))
        elif v.status == "found" and c.kind == "article" and position >= 0:
            found_articles.append((position, v))
    for quote, qpos in extract_quotes(text):
        paragraph = paragraph_at(paragraphs, qpos)
        nearby = [(abs(p - qpos), p, v) for p, v in found_articles if paragraph_at(paragraphs, p) == paragraph]
        if not nearby:
            continue
        _, position, v = min(nearby, key=lambda x: x[0])
        c = v.citation
        law_quotes.add(qpos)
        if quote_in(quote, v.text or ""):
            continue
        elsewhere = _article_with(source, v.law_id or "", quote)
        if elsewhere is None:
            issues.append(
                Issue(
                    0,
                    "cita_legal_no_literal",
                    "error",
                    quote,
                    paragraph,
                    f"el texto entrecomillado no aparece literalmente en el artículo {c.article} ni en ningún otro de la norma",
                    None,
                    v.url,
                )
            )
            continue
        number = elsewhere["number"]
        fixed = re.sub(r"\d+(?:\.\d+)?(?:\s*(?:bis|ter|quáter|quater|quinquies))?", number, c.raw, count=1)
        title = _article_title(elsewhere)
        issues.append(
            Issue(
                0,
                "articulo_equivocado",
                "error",
                c.raw,
                paragraph,
                f"el texto citado no está en el artículo {c.article}, sino en el {number}"
                + (f" ({title})" if title else ""),
                Change(c.raw, fixed, occurrence_at(text, c.raw, position)),
                f"https://www.boe.es/buscar/act.php?id={v.law_id}#a{number}",
                quote,
            )
        )
    return issues, len(results), law_quotes


def _article_title(art: dict) -> str:
    """«Cómputo de los plazos» from the first line «Artículo 133. Cómputo de los plazos.»"""
    first = (art.get("text") or "").strip().splitlines()[0] if art.get("text") else (art.get("heading") or "")
    return re.sub(r"^art[íi]culo\s+\S+\.?\s*", "", first, flags=re.IGNORECASE).strip(" .")


def _article_with(source: lawcheck.LawSource, norm_id: str, quote: str) -> dict | None:
    listing = getattr(source, "articles", None)
    if listing is None or not norm_id:
        return None
    for art in listing(norm_id):
        if quote_in(quote, art["text"]):
            return art
    return None


# ── the model's part ───────────────────────────────────────────────────
def windows_of(paragraphs: list[tuple[int, int, str]], size: int = WINDOW_CHARS) -> list[tuple[int, str]]:
    """Runs of whole paragraphs up to `size` characters, with the number of their first paragraph."""
    out: list[tuple[int, str]] = []
    first, parts, length = 1, [], 0
    for i, (_, _, body) in enumerate(paragraphs, start=1):
        if parts and length + len(body) > size:
            out.append((first, "\n\n".join(parts)))
            first, parts, length = i, [], 0
        parts.append(body)
        length += len(body) + 2
    if parts:
        out.append((first, "\n\n".join(parts)))
    return out


def find_legal_issues(llm, text: str, paragraphs, guard, progress=None, log=None) -> tuple[list[Issue], int]:
    """The model reads the draft in windows and points at legal issues; only pointers quoting the draft survive."""
    issues: list[Issue] = []
    finder = llm.with_structured_output(Issues, method="json_schema")
    windows = windows_of(paragraphs)
    for k, (first, window) in enumerate(windows, start=1):
        with guard:
            result: Issues = finder.invoke(
                [SystemMessage(prompts.REVIEW_ISSUES), HumanMessage(prompts.data_block("draft", window))]
            )
        before, dropped_here = len(issues), 0
        for item in result.issues:
            passage = item.quote.strip()
            if (
                not passage
                or not quote_in(passage, window)
                or from_injection(passage, window)
                or not item.reason.strip()
            ):
                dropped_here += 1
                continue
            position = text.find(passage)
            issues.append(
                Issue(
                    0,
                    "cuestion_legal",
                    "aviso",
                    passage,
                    paragraph_at(paragraphs, max(position, 0)),
                    item.reason.strip(),
                )
            )
        if progress:
            progress(k, len(windows))
        if log:
            log(
                f"Ventana {k}/{len(windows)} (desde el párrafo {first}): {len(issues) - before} cuestión(es) con pasaje literal"
                + (f", {dropped_here} descartada(s) por no citar el borrador" if dropped_here else "")
            )
    return issues, len(windows)


# ── the pass ───────────────────────────────────────────────────────────
def review(
    *,
    text: str,
    name: str = "borrador",
    chunks: list[Chunk] | None = None,
    law_source: lawcheck.LawSource | None = None,
    aliases: lawcheck.Aliases | None = None,
    llm=None,
    lock=None,
    progress=None,
    log=None,
) -> Review:
    """Review a draft: repeats and quotes in code, citations against the archive, legal issues by the model.
    `log(line)` receives a readable account of each phase as it finishes."""
    started = time.perf_counter()
    say = log or (lambda _line: None)
    if not text.strip():
        raise ValueError("the draft is empty")
    paragraphs = paragraphs_of(text)
    say(f"Borrador «{name}»: {len(paragraphs)} párrafos, {len(text):,} caracteres")
    notes: list[str] = []
    issues = find_repeats(text, paragraphs)
    say(f"Párrafos repetidos: {len(issues)}" + "".join(f" · el {i.paragraph} {i.reason}" for i in issues))
    law_quotes: set[int] = set()
    citations_checked = 0
    if law_source is not None and aliases is not None:
        law_issues, citations_checked, law_quotes = check_law(text, paragraphs, law_source, aliases)
        issues += law_issues
        say(
            f"Citas legales comprobadas contra el archivo del BOE: {citations_checked}"
            + "".join(f" · {KINDS.get(i.kind, i.kind)}: {i.passage[:80]}" for i in law_issues)
        )
    else:
        notes.append("citas legales no comprobadas: el archivo del BOE no está configurado")
        say("Citas legales: sin comprobar (archivo del BOE no configurado)")
    quotes_checked = 0
    if chunks:
        quote_issues, quotes_checked = check_quotes(text, paragraphs, chunks, skip=law_quotes)
        issues += quote_issues
        say(
            f"Citas entrecomilladas cotejadas con {len(chunks)} fragmentos de documentos: {quotes_checked}"
            + "".join(
                f" · {KINDS.get(i.kind, i.kind)} (párrafo {i.paragraph}): «{i.passage[:80]}»" for i in quote_issues
            )
        )
    else:
        notes.append("citas de documentos no comprobadas: no hay documentos indexados con los que cotejar")
        say("Citas de documentos: sin comprobar (no hay documentos indexados)")
    windows = 0
    if llm is not None:
        say("El modelo lee el borrador por ventanas en busca de cuestiones legales concretas…")
        guard = lock if lock is not None else nullcontext()
        legal, windows = find_legal_issues(llm, text, paragraphs, guard, progress, log=say)
        issues += legal
        say(f"Cuestiones legales señaladas por el modelo (avisos): {len(legal)}")
    issues.sort(key=lambda i: (i.paragraph, text.find(i.passage)))
    for n, issue in enumerate(issues, start=1):
        issue.n = n
    return Review(
        name,
        len(paragraphs),
        issues,
        quotes_checked,
        citations_checked,
        windows,
        notes,
        round(time.perf_counter() - started, 1),
    )


def apply_changes(text: str, changes: list[Change], accepted: set[int] | None = None) -> str:
    """The draft with the accepted changes applied (all of them when `accepted` is None), later positions first."""
    located: list[tuple[int, Change]] = []
    for i, change in enumerate(changes, start=1):
        if accepted is not None and i not in accepted:
            continue
        position = -1
        for _ in range(change.occurrence):
            position = text.find(change.before, position + 1)
            if position < 0:
                break
        if position >= 0:
            located.append((position, change))
    for position, change in sorted(located, key=lambda x: -x[0]):
        end = position + len(change.before)
        if change.after == "":
            while end < len(text) and text[end] == "\n":  # a deleted paragraph takes its blank line along
                end += 1
        text = text[:position] + change.after + text[end:]
    return text


def revealed(value, reveal):
    """`reveal` applied to every string inside a result, whatever its nesting."""
    if isinstance(value, str):
        return reveal(value)
    if isinstance(value, list):
        return [revealed(v, reveal) for v in value]
    if isinstance(value, dict):
        return {k: revealed(v, reveal) for k, v in value.items()}
    return value


def format_report(r: Review) -> str:
    lines = [
        f"# Revisión de «{r.name}»",
        "",
        f"Párrafos: {r.paragraphs} · citas de documentos comprobadas: {r.quotes_checked} · citas legales comprobadas: {r.citations_checked}"
        + (f" · ventanas leídas por el modelo: {r.windows}" if r.windows else ""),
        "",
    ]
    for note in r.notes:
        lines += [f"_{note}_", ""]
    for severity, title in (("error", "Errores"), ("aviso", "Avisos")):
        group = [i for i in r.issues if i.severity == severity]
        if not group:
            continue
        lines += [f"## {title}", ""]
        for i in group:
            lines.append(f"- [{i.n}] {KINDS.get(i.kind, i.kind)} (párrafo {i.paragraph}): {i.reason}")
            lines.append(f"  > «{i.passage[:300]}{'…' if len(i.passage) > 300 else ''}»")
            if i.quote and i.kind != "articulo_equivocado":
                lines.append(f"  > Original: «{i.quote[:300]}{'…' if len(i.quote) > 300 else ''}»")
            if i.change is not None:
                lines.append(
                    "  Cambio propuesto: "
                    + ("eliminar el párrafo" if i.change.after == "" else f"«{i.change.after[:300]}»")
                )
            if i.source:
                lines.append(f"  {i.source}")
        lines.append("")
    if not r.issues:
        lines.append(
            "Sin hallazgos: nada repetido, todas las citas literales y todas las referencias legales resueltas."
        )
    return "\n".join(lines).rstrip() + "\n"


# ── job plumbing (shares the jobs table with analysis) ─────────────────
def run_review_job(
    conn, job_id: str, *, memory, law_source_factory, aliases, llm, lock=None, anonymize=None, reveal=None
) -> None:
    """Execute a queued review job: anonymise the draft like the index, review it, reveal the result."""
    from .analysis import get_job, update_job

    job = get_job(conn, job_id)
    if job is None:
        return
    update_job(conn, job_id, status="running")
    from .analysis import JobLog

    log = JobLog(conn, job_id)
    try:
        spec = json.loads(job["goal"] or "{}")
        text = spec.get("text", "")
        if anonymize:
            log("Pseudonimizando el borrador como el índice (nombres e identificadores → marcadores)…")
            with lock if lock is not None else nullcontext():
                text = anonymize(text)
        sources = spec.get("documents") or [s for s in memory.indexed_sources()]
        exclude = spec.get("exclude")
        chunks = [c for s in sources if s != exclude for c in memory.document_chunks(s)]
        law_source = law_source_factory() if law_source_factory else None
        result = review(
            text=text,
            name=job["source"],
            chunks=chunks,
            law_source=law_source,
            aliases=aliases if law_source is not None else None,
            llm=llm if spec.get("legal", True) else None,
            lock=lock,
            progress=lambda done, total: update_job(conn, job_id, done=done, total=total),
            log=log,
        )
        log(
            f"Terminado en {result.seconds} s: {result.errors()} errores, {len(result.issues) - result.errors()} avisos"
        )
        data = result.as_dict(text)
        if reveal:
            data = revealed(data, reveal)
        update_job(conn, job_id, status="done", result=data, done=1, total=1)
    except Exception as exc:  # noqa: BLE001 — the job records its failure; the server keeps serving
        update_job(conn, job_id, status="failed", error=f"{type(exc).__name__}: {exc}")
