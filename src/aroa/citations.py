"""Deterministic grounding checks for an answer built from documents: every [n] must point to a fragment
that was provided and its verbatim quote must appear in that fragment. A quote taken from text that
addresses an AI assistant is rejected, so an injection inside a document never becomes a cited fact."""

import re
import unicodedata

from .memory.chunking import Chunk
from .schemas import DocumentAnswer

# Small models sometimes write [n1] for [1]; both are accepted and the rendered answer uses [1].
MARKER = re.compile(r"\[n?(\d+(?:\s*,\s*n?\d+)*)\]")
ELLIPSIS = re.compile(r"\.\.\.|…|\[\.\.\.\]")
INJECTION = re.compile(
    r"\b(ignora|ignore|olvida|disregard|forget)\b.{0,40}\b(instrucciones|instructions|indicaciones)\b"
    r"|\b(para|al|a)\s+(el\s+|un\s+)?(asistente|modelo)\s+de\s+(ia|inteligencia artificial)\b"
    r"|\b(to|for)\s+(the\s+|an?\s+)?(ai|llm)\s+(assistant|model)\b"
    r"|\bsystem prompt\b",
    re.IGNORECASE,
)


def cited_numbers(text: str) -> set[int]:
    return {int(n.strip().lstrip("n")) for group in MARKER.findall(text) for n in group.split(",")}


def tidy_markers(text: str) -> str:
    return MARKER.sub(lambda m: "[" + ", ".join(n.strip().lstrip("n") for n in m.group(1).split(",")) + "]", text)


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


def check_answer(answer: DocumentAnswer, sources: list[Chunk]) -> list[str]:
    """Problems found; empty means the answer is grounded."""
    cited = cited_numbers(answer.answer)
    errors: list[str] = []
    if not cited:
        errors.append("The answer cites no fragment.")
    unknown = sorted(n for n in cited if not 1 <= n <= len(sources))
    if unknown:
        errors.append(f"It cites fragments that do not exist: {unknown}.")
    quoted = set()
    for quote in answer.quotes:
        if not 1 <= quote.source <= len(sources):
            continue
        source_text = sources[quote.source - 1].text
        if from_injection(quote.text, source_text):
            errors.append(f"The quote for [{quote.source}] comes from text that instructs an AI assistant.")
        elif quote_in(quote.text, source_text):
            quoted.add(quote.source)
        else:
            errors.append(f"The quote attributed to [{quote.source}] does not appear in that fragment.")
    missing = sorted(n for n in cited - quoted if 1 <= n <= len(sources))
    if missing:
        errors.append(f"Missing verifiable quotes for {missing}.")
    return errors
