"""Deterministic grounding checks for an answer built from documents: every [n] must point to a fragment
that was provided and its verbatim quote must appear in that fragment. A quote taken from text that
addresses an AI assistant is rejected, so an injection inside a document never becomes a cited fact."""

import re
import unicodedata

from .memory.chunking import Chunk
from .schemas import DocumentAnswer

# Small models sometimes write [n1] for [1]; both are accepted and the rendered answer uses [1].
MARKER = re.compile(r"\[(?:n\s*=?\s*)?(\d+(?:\s*,\s*(?:n\s*=?\s*)?\d+)*)\]")  # [1], [1, 2], [n1], [n=1]
ELLIPSIS = re.compile(r"\.\.\.|…|\[\.\.\.\]")
INJECTION = re.compile(
    r"\b(ignora|ignore|olvida|disregard|forget)\b.{0,40}\b(instrucciones|instructions|indicaciones)\b"
    r"|\b(para|al|a)\s+(el\s+|un\s+)?(asistente|modelo)\s+de\s+(ia|inteligencia artificial)\b"
    r"|\b(to|for)\s+(the\s+|an?\s+)?(ai|llm)\s+(assistant|model)\b"
    r"|\bsystem prompt\b",
    re.IGNORECASE,
)


def cited_numbers(text: str) -> set[int]:
    return {int(_digits(n)) for group in MARKER.findall(text) for n in group.split(",")}


def tidy_markers(text: str) -> str:
    return MARKER.sub(lambda m: "[" + ", ".join(_digits(n) for n in m.group(1).split(",")) + "]", text)


def _digits(marker: str) -> str:
    return re.sub(r"\D", "", marker)


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text).lower()
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = re.sub(r"[«»\"“”‘’'*_]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip(" .,;:…")


def quote_in(quote: str, text: str) -> bool:
    haystack, position = normalize(text), 0
    for part in (normalize(p) for p in ELLIPSIS.split(quote)):
        if not part:
            continue
        found = haystack.find(part, position)
        if found < 0:
            return False
        position = found + len(part)
    return position > 0


def from_injection(quote: str, text: str) -> bool:
    sentences = re.split(r"(?<=[.!?])\s+|\n+", text)
    return any(INJECTION.search(s) and quote_in(quote, s) for s in sentences)


def citation_problems(answer: DocumentAnswer, sources: list[Chunk]) -> list[dict]:
    """What is wrong with each citation, as data: kind, fragment number and the quote involved."""
    cited = cited_numbers(answer.answer)
    problems: list[dict] = []
    if not cited:
        problems.append({"kind": "none"})
    unknown = sorted(n for n in cited if not 1 <= n <= len(sources))
    if unknown:
        problems.append({"kind": "unknown", "numbers": unknown})
    quoted = set()
    for quote in answer.quotes:
        if not 1 <= quote.source <= len(sources):
            continue
        chunk = sources[quote.source - 1]
        if from_injection(quote.text, chunk.text):
            problems.append({"kind": "injection", "n": quote.source, "label": chunk.label(), "quote": quote.text})
        elif quote_in(quote.text, chunk.text):
            quoted.add(quote.source)
        else:
            problems.append({"kind": "not_in_fragment", "n": quote.source, "label": chunk.label(), "quote": quote.text})
    for n in sorted(cited - quoted):
        if 1 <= n <= len(sources) and not any(p.get("n") == n for p in problems):
            problems.append({"kind": "no_quote", "n": n, "label": sources[n - 1].label()})
    return problems


def unverified_numbers(problems: list[dict]) -> set[int]:
    return {p["n"] for p in problems if "n" in p} | {n for p in problems for n in p.get("numbers", [])}


def check_answer(answer: DocumentAnswer, sources: list[Chunk], lang: str = "en") -> list[str]:
    """Problems found, one sentence each in the configured language; empty means the answer is grounded."""
    from .i18n import t

    out: list[str] = []
    for p in citation_problems(answer, sources):
        kind = p["kind"]
        if kind == "none":
            out.append(t(lang, "cite_none"))
        elif kind == "unknown":
            out.append(t(lang, "cite_unknown", numbers=", ".join(f"[{n}]" for n in p["numbers"])))
        else:
            quote = p.get("quote", "")
            short = quote if len(quote) <= 160 else quote[:160] + "…"
            out.append(t(lang, "cite_" + kind, n=p["n"], label=f"«{p['label']}»", quote=short))
    return out
