"""Pseudonymisation of personal data before anything reaches a model, a store or a trace.

Identifiers (ID numbers, phones, emails, IBANs, cards, plates) are found with patterns in code.
Names of people and organisations are found by a short structured model call when privacy mode
is "full". Every original is mapped to a stable placeholder such as [PERSON_3] in SQLite, so the
same person gets the same placeholder in every conversation and document, and the mapping only
lives on this machine. The user interface reveals the placeholders; the model never sees the originals.
"""

import re
import sqlite3
from collections.abc import Callable

from pydantic import BaseModel, Field

from .telemetry.metrics import now_iso

PATTERNS: list[tuple[str, re.Pattern]] = [
    ("EMAIL", re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")),
    ("IBAN", re.compile(r"\b[A-Z]{2}\d{2}(?:\s?[A-Z0-9]{4}){2,7}(?:\s?[A-Z0-9]{1,4})?\b")),
    ("CARD", re.compile(r"\b(?:\d{4}[ -]?){3}\d{4}\b")),
    ("NIE", re.compile(r"\b[XYZ][ -]?\d{7}[ -]?[A-Z]\b", re.IGNORECASE)),
    ("DNI", re.compile(r"\b\d{8}[ -]?[A-Z]\b")),
    ("SSN", re.compile(r"\b\d{2}[ /-]?\d{8}[ /-]?\d{2}\b")),
    (
        "PHONE",
        re.compile(r"(?<![\w/])(?:\+\d{1,3}[ -]?)?(?:\(?\d{2,3}\)?[ -]?)?\d{3}[ -]?\d{2,3}[ -]?\d{2,3}(?![\w/])"),
    ),
    ("PLATE", re.compile(r"\b\d{4}[ -]?[B-DF-HJ-NP-TV-Z]{3}\b")),
]
PLACEHOLDER = re.compile(r"\[(PERSON|ORG|EMAIL|IBAN|CARD|NIE|DNI|SSN|PHONE|PLATE)_(\d+)\]")

SCHEMA = """
CREATE TABLE IF NOT EXISTS pseudonyms (
    placeholder TEXT PRIMARY KEY,
    original TEXT NOT NULL,
    kind TEXT NOT NULL,
    ts TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS pseudonyms_original ON pseudonyms(original);
"""


class Names(BaseModel):
    people: list[str] = Field(
        default_factory=list,
        description="Natural persons only (first name and surnames as written), plus any shorter form used for them.",
    )
    organizations: list[str] = Field(
        default_factory=list,
        description="Companies, law firms and other private organisations as written (S.L., S.A., LLP...).",
    )


NAMES_PROMPT = """List every natural person and every organisation named in the text, exactly as written.
Include shorter forms that refer to the same one (surname alone, initials, trade name). Do not include
courts, public bodies, laws, places, job titles or generic words. Empty lists if there are none."""

NamesExtractor = Callable[[str], Names]

# The model proposes names; the code decides what is a name. Role words and public bodies are never
# pseudonymised: replacing "el cliente" with a placeholder would wreck every later question.
ROLE_WORDS = {
    "cliente",
    "clienta",
    "clientes",
    "proveedor",
    "proveedora",
    "arrendador",
    "arrendadora",
    "arrendatario",
    "arrendataria",
    "parte",
    "partes",
    "demandante",
    "demandado",
    "demandada",
    "abogado",
    "abogada",
    "letrado",
    "letrada",
    "procurador",
    "procuradora",
    "juez",
    "jueza",
    "magistrado",
    "tribunal",
    "juzgado",
    "empresa",
    "sociedad",
    "usuario",
    "usuaria",
    "asistente",
    "firma",
    "despacho",
    "bufete",
    "contraparte",
    "deudor",
    "acreedor",
    "comprador",
    "vendedor",
    "vendedora",
    "trabajador",
    "trabajadora",
    "empleado",
    "empleada",
    "empleador",
    "testigo",
    "perito",
    "responsable",
    "encargado",
    "interesado",
    "titular",
    "the",
    "client",
    "provider",
    "landlord",
    "tenant",
    "party",
    "counsel",
    "lawyer",
    "court",
    "company",
    "user",
    "assistant",
}
PUBLIC_BODY = re.compile(
    r"\b(instituto|agencia|ministerio|juzgados?|audiencia|tribunal(?:es)?|ayuntamiento|registro|bolet[ií]n|comisi[oó]n|"
    r"direcci[oó]n general|generalitat|junta|consejer[ií]a|delegaci[oó]n|hacienda|seguridad social|"
    r"court|ministry|agency|office of|department of)\b",
    re.IGNORECASE,
)
ARTICLES = {
    "el",
    "la",
    "los",
    "las",
    "un",
    "una",
    "the",
    "a",
    "an",
    "sr",
    "sra",
    "sr.",
    "sra.",
    "d",
    "d.",
    "dña",
    "dña.",
}
CORPORATE = re.compile(
    r"\b(s\.?l\.?p?\.?u?|s\.?a\.?u?|s\.?c\.?|s\.?coop\.?|llp|llc|ltd|inc|gmbh|plc|sas|srl|bv)\b\.?$", re.IGNORECASE
)


def clean_name(candidate: str) -> str:
    """Strip articles, honorifics and surrounding punctuation."""
    words = candidate.strip(" ,;:«»\"'()").lstrip(".").split()
    while words and words[0].lower() in ARTICLES:
        words = words[1:]
    name = " ".join(words)
    return name if CORPORATE.search(name) else name.rstrip(".")


def plausible_name(candidate: str, kind: str) -> str | None:
    """The cleaned name when it looks like a real person or private organisation, else None."""
    name = clean_name(candidate)
    words = name.split()
    if not words or PUBLIC_BODY.search(name):
        return None
    if all(w.lower().strip(".,") in ROLE_WORDS for w in words):
        return None
    if kind == "PERSON":
        # a person is at least a capitalised word of 3+ letters; single words are accepted as surnames
        return name if all(w[:1].isupper() and len(w.strip(".,")) >= 2 for w in words) and len(words[0]) >= 3 else None
    if CORPORATE.search(name) or len([w for w in words if w[:1].isupper()]) >= 2:
        return name
    return None


class Anonymizer:
    def __init__(self, conn: sqlite3.Connection, extract_names: NamesExtractor | None = None):
        self.conn = conn
        self.extract_names = extract_names
        conn.executescript(SCHEMA)

    # ── mapping ────────────────────────────────────────────────────────
    def placeholder_for(self, original: str, kind: str) -> str:
        row = self.conn.execute(
            "SELECT placeholder FROM pseudonyms WHERE original = ? AND kind = ?", (original, kind)
        ).fetchone()
        if row:
            return row["placeholder"]
        if kind in ("PERSON", "ORG") and len(original) >= 5:
            # "Talleres Ribera" and "Talleres Ribera, S.L." are the same party: reuse the placeholder
            for known in self.conn.execute("SELECT placeholder, original FROM pseudonyms WHERE kind = ?", (kind,)):
                a, b = original.lower(), known["original"].lower()
                if (a in b or b in a) and min(len(a), len(b)) >= 5:
                    return known["placeholder"]
        count = self.conn.execute("SELECT COUNT(*) AS n FROM pseudonyms WHERE kind = ?", (kind,)).fetchone()["n"]
        placeholder = f"[{kind}_{count + 1}]"
        self.conn.execute(
            "INSERT INTO pseudonyms (placeholder, original, kind, ts) VALUES (?,?,?,?)",
            (placeholder, original, kind, now_iso()),
        )
        self.conn.commit()
        return placeholder

    def mapping(self) -> dict[str, str]:
        return {
            r["placeholder"]: r["original"] for r in self.conn.execute("SELECT placeholder, original FROM pseudonyms")
        }

    def known_names(self) -> list[tuple[str, str]]:
        rows = self.conn.execute("SELECT original, kind FROM pseudonyms WHERE kind IN ('PERSON', 'ORG')").fetchall()
        return [(r["original"], r["kind"]) for r in rows]

    # ── anonymise ──────────────────────────────────────────────────────
    def anonymize(self, text: str, *, names: bool = True) -> str:
        """Replace identifiers and known names; with `names`, ask the extractor for new names first."""
        if not text:
            return text
        found: list[tuple[str, str]] = []
        if names and self.extract_names:
            result = self.extract_names(text)
            proposed = [(n, "PERSON") for n in result.people] + [(n, "ORG") for n in result.organizations]
            for candidate, kind in proposed:
                kind = "ORG" if CORPORATE.search(clean_name(candidate)) else kind
                if (name := plausible_name(candidate, kind)) is not None:
                    found.append((name, kind))
        candidates = [(n, k) for n, k in found + self.known_names() if len(n.strip()) >= 3]
        for name, kind in sorted(set(candidates), key=lambda x: -len(x[0])):
            if name.lower() in text.lower():
                text = re.sub(
                    r"(?<!\w)" + re.escape(name) + r"(?!\w)",
                    self.placeholder_for(name, kind),
                    text,
                    flags=re.IGNORECASE,
                )
        for kind, pattern in PATTERNS:
            text = pattern.sub(lambda m, k=kind: self.placeholder_for(m.group(0), k), text)
        return text

    def reveal(self, text: str) -> str:
        if not text:
            return text
        mapping = self.mapping()
        return PLACEHOLDER.sub(lambda m: mapping.get(m.group(0), m.group(0)), text)

    def redact(self, text: str, *, names: bool = True) -> str:
        """Irreversible: placeholders become black boxes, for documents that leave the firm."""
        return PLACEHOLDER.sub(lambda m: "█" * 8, self.anonymize(text, names=names))


def contains_personal_data(text: str) -> list[str]:
    """Kinds of identifiers present in the text (used by tests and evals to prove nothing leaked)."""
    return [kind for kind, pattern in PATTERNS if pattern.search(text)]
