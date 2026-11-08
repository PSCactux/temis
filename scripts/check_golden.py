"""Verify evals/golden.jsonl against the sample corpus without running any model.

For every case: the schema is the evaluation `Case`, ids are unique, kinds are valid and the id prefix
matches the kind. For documents cases: each `expected` entry matches at least one chunk produced by the
project's own chunker over data/samples, every `must_include` term (any `|` alternative) appears in the
text of one of those chunks, and `documents` entries name existing sample files. Privacy cases must carry
at least one identifier the anonymiser detects, and their `must_include` terms must appear verbatim in the
message or the setup turns (the reply is revealed exactly as written). Project cases need `expect_file`.

Usage: VIRTUAL_ENV= uv run python scripts/check_golden.py [evals/golden.jsonl] [data/samples]
Exits 1 and lists every problem when something is wrong.
"""

import sys
from collections import Counter, defaultdict
from pathlib import Path

from aroa.config import Settings
from aroa.evaluation import Case, load_cases, matches
from aroa.memory.loaders import document_files, load_file
from aroa.privacy import contains_personal_data

KINDS = {"identity", "chat", "memory", "wall", "privacy", "policy", "documents", "web", "alarms", "project"}
PREFIX_KINDS = {
    "docs": {"documents"},
    "selected": {"documents"},
    "memory": {"memory"},
    "wall": {"wall"},
    "privacy": {"privacy"},
    "policy": {"policy", "chat"},  # policy-allowed-* cases are ordinary work and use kind chat
    "alarm": {"alarms"},
    "project": {"project"},
    "web": {"web"},
    "chat": {"chat"},
    "identity": {"identity"},
}


def chunk_samples(samples: Path, max_chars: int) -> list:
    chunks = []
    for path in document_files(samples):
        chunks += load_file(path, max_chars)
    return chunks


def check_case(case: Case, chunks: list, files: set[str], problems: list[str]) -> None:
    def problem(text: str) -> None:
        problems.append(f"{case.id}: {text}")

    if case.kind not in KINDS:
        problem(f"unknown kind {case.kind!r}")
    allowed = PREFIX_KINDS.get(case.id.split("-", 1)[0])
    if allowed is None:
        problem("id prefix is not one of " + ", ".join(sorted(PREFIX_KINDS)))
    elif case.kind not in allowed:
        problem(f"id prefix does not match kind {case.kind!r}")
    if not case.question.strip():
        problem("empty question")
    for term in case.must_include:
        if not term.strip() or any(not alt.strip() for alt in term.split("|")):
            problem(f"empty must_include alternative in {term!r}")

    if case.kind == "documents":
        for name in case.documents:
            if name not in files:
                problem(f"documents entry {name!r} is not a sample file")
        if case.answerable:
            if not case.expected:
                problem("answerable documents case without `expected`")
            if not case.must_include:
                problem("answerable documents case without `must_include`")
        elif case.expected:
            problem("unanswerable case must not list `expected` sections")
        hits = []
        for expected in case.expected:
            found = [c for c in chunks if matches(c.to_payload(), expected)]
            if not found:
                problem(f"expected {expected!r} matches no chunk")
            sections = {(c.doc_id, c.section) for c in found}
            if len(sections) > 1:
                problem(f"expected {expected!r} is ambiguous, matches {len(sections)} sections")
            if case.documents and found and not {c.source for c in found} <= set(case.documents):
                problem(f"expected {expected!r} is outside the selected documents")
            hits += found
        corpus = "\n".join(f"{c.section}\n{c.text}" for c in hits).lower()
        for term in case.must_include:
            if hits and not any(alt.lower() in corpus for alt in term.split("|")):
                problem(f"must_include {term!r} is not in the text of the expected chunks")
    elif case.kind == "privacy":
        if not contains_personal_data(case.question + "\n" + "\n".join(case.setup)):
            problem("privacy case carries no detectable identifier")
        if not case.must_include:
            problem("privacy case without `must_include` (nothing to prove the reveal)")
        written = case.question + "\n" + "\n".join(case.setup)
        for term in case.must_include:
            if not any(alt in written for alt in term.split("|")):
                problem(f"must_include {term!r} does not appear verbatim in the message or setup")
    elif case.kind == "policy":
        if case.must_include:
            problem("a blocked case cannot require words in the reply")
    elif case.kind == "project":
        if not case.expect_file:
            problem("project case without `expect_file`")
        if not case.matter:
            problem("project case without `matter`")
    elif case.kind == "memory":
        if not case.setup:
            problem("memory case without `setup` turns")
        if not case.must_include:
            problem("memory case without `must_include`")
    elif case.kind == "wall":
        if not case.must_not_include:
            problem("wall case without `must_not_include`")
        if not case.matter:
            problem("wall case without `matter`")
    elif case.kind in {"web", "chat", "identity"} and not (case.must_include or case.must_not_include):
        if case.kind != "identity":
            problem(f"{case.kind} case with nothing to check")


def main() -> int:
    dataset = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("evals/golden.jsonl")
    samples = Path(sys.argv[2]) if len(sys.argv) > 2 else Settings().samples_dir
    problems: list[str] = []
    try:
        cases = load_cases(dataset)
    except TypeError as exc:  # an unknown field in a row
        print(f"{dataset}: a row does not fit the Case schema: {exc}")
        return 1
    chunks = chunk_samples(samples, Settings().chunk_max_chars)
    files = {p.name for p in document_files(samples)}

    counts = Counter(c.id for c in cases)
    for case_id, n in sorted(counts.items()):
        if n > 1:
            problems.append(f"{case_id}: id used {n} times")
    for case in cases:
        check_case(case, chunks, files, problems)

    # Wall cases must come after a memory case in another matter already stored what they must not recall.
    stored_matters: set[str] = set()
    for case in cases:
        if case.kind == "wall" and not (stored_matters - {case.matter}):
            problems.append(f"{case.id}: no earlier memory case stored facts in another matter")
        if case.setup and case.kind in {"memory", "privacy"}:
            stored_matters.add(case.matter)

    by_kind = Counter(c.kind for c in cases)
    covered = defaultdict(set)
    for case in cases:
        for expected in case.expected:
            for c in chunks:
                if matches(c.to_payload(), expected):
                    covered[c.doc_id].add(c.section)
    print(f"{len(cases)} cases in {dataset}: " + ", ".join(f"{k} {n}" for k, n in sorted(by_kind.items())))
    for doc_id in sorted({c.doc_id for c in chunks}):
        sections = {c.section for c in chunks if c.doc_id == doc_id}
        missing = sorted(sections - covered[doc_id])
        print(
            f"  {doc_id}: {len(covered[doc_id])}/{len(sections)} sections covered"
            + (f"; missing {missing}" if missing else "")
        )
        if missing:
            problems.append(f"{doc_id}: sections without a case: {missing}")

    if problems:
        print(f"\n{len(problems)} problem(s):")
        for p in problems:
            print(f"- {p}")
        return 1
    print("\nOK: every expected section exists, every must_include term is in its chunks, ids and kinds are valid.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
