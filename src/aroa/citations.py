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


MEMORY_MARKER = re.compile(r"\[(m\d+(?:\s*,\s*m\d+)*)\]", re.IGNORECASE)


def memory_items(memories: dict) -> list[dict]:
    """Recalled facts then conversations, numbered m1, m2… in the order the prompt lists them."""
    items: list[dict] = []
    for f in memories.get("facts", []) or []:
        items.append(
            {
                "n": f"m{len(items) + 1}",
                "kind": "fact",
                "text": f["text"],
                "when": (f.get("ts") or "")[:10],
                "origin": f.get("origin", ""),
            }
        )
    for e in memories.get("episodes", []) or []:
        items.append(
            {
                "n": f"m{len(items) + 1}",
                "kind": "conversation",
                "user": e.get("user", ""),
                "assistant": e.get("assistant", ""),
                "when": (e.get("ts") or "")[:10],
                "thread_id": e.get("thread_id", ""),
            }
        )
    return items


def cite_memory(text: str, memories: dict, lang: str = "en") -> tuple[str, list[dict]]:
    """Markers like [m1] kept only when the block exists; returns the clean reply and the cited items as sources,
    in the same shape the documents route uses (n, label, quote) plus kind and when."""
    from .i18n import t

    items = {i["n"]: i for i in memory_items(memories)}
    cited: list[str] = []

    def sub(m: re.Match) -> str:
        ids = [x.strip().lower() for x in m.group(1).split(",")]
        kept = [x for x in ids if x in items]
        for x in kept:
            if x not in cited:
                cited.append(x)
        return "[" + ", ".join(kept) + "]" if kept else ""

    clean = MEMORY_MARKER.sub(sub, text)
    clean = re.sub(r"[ \t]+([.,;:!?])", r"\1", re.sub(r"[ \t]{2,}", " ", clean))
    sources = []
    for x in cited:
        item = items[x]
        if item["kind"] == "fact":
            label = t(lang, "mem_fact", when=item["when"])
            quote = item["text"]
        else:
            label = t(lang, "mem_conv", when=item["when"])
            quote = item["user"]
        sources.append(
            {
                "n": x,
                "kind": "memory",
                "label": label,
                "quote": quote,
                "when": item["when"],
                "file": None,
                "verified": True,
            }
        )
    return clean, sources


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
