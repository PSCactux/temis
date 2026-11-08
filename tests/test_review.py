"""The review pass: repeats, quotes and legal citations checked in code; the model's pointers kept only when verbatim."""

import sqlite3
import threading
from pathlib import Path

from aroa.analysis import create_job, get_job
from aroa.lawcheck import Aliases
from aroa.memory.chunking import Chunk
from aroa.models import FakeChatModel
from aroa.review import (
    Change,
    aligned,
    apply_changes,
    extract_quotes,
    format_report,
    paragraphs_of,
    review,
    run_review_job,
)
from aroa.schemas import IssueItem, Issues
from aroa.telemetry.db import SCHEMA
from tests.conftest import FakeMemory

ALIASES = Aliases.load(Path(__file__).parent.parent / "data/laws.json")

LEASE = Chunk(
    id="lease:0",
    doc_id="lease",
    title="Contrato de arrendamiento",
    section="Cláusula 8.ª",
    text=(
        "La Arrendataria podrá desistir del contrato una vez transcurridos al menos seis meses desde su inicio. "
        "En ese caso, indemnizará a la Arrendadora con una cantidad equivalente a una mensualidad de renta por cada "
        "año de contrato que reste por cumplir; los periodos inferiores al año darán lugar a la parte proporcional."
    ),
    source="lease.md",
)

DRAFT = """PRIMERO. La Arrendataria comunicó su voluntad de desistir con más de treinta días de antelación, por lo que la indemnización reclamada carece de fundamento y la demanda debe ser desestimada.

SEGUNDO. El contrato dice que «indemnizará a la Arrendadora con una cantidad equivalente a dos mensualidades de renta por cada año de contrato que reste por cumplir», y la demanda pide más.

TERCERO. Según el contrato, «los periodos inferiores al año darán lugar a la parte proporcional», lo que la demanda ignora.

I. Conforme al artículo 999 de la LEC el escrito está en plazo. Conforme al artículo 130 de la LEC, «En el cómputo de los plazos señalados por días se excluirán los inhábiles». Cita la STS 1/2020.

II. La Arrendataria comunicó su voluntad de desistir con más de treinta días de antelación, por lo que la indemnización reclamada carece de fundamento y la demanda debe ser desestimada.

SUPLICO que se desestime la demanda.
"""


class FakeLaw:
    norms = {"BOE-A-2000-323": "Ley 1/2000, de 7 de enero, de Enjuiciamiento Civil."}
    table = {
        "130": "Artículo 130. Días y horas hábiles.\n1. Las actuaciones judiciales habrán de practicarse en días y horas hábiles.",
        "133": "Artículo 133. Cómputo de los plazos.\n1. Los plazos comenzarán a correr desde el día siguiente.\n2. En el cómputo de los plazos señalados por días se excluirán los inhábiles.",
    }

    def find_norm(self, reference):
        return None

    def is_stored(self, norm_id):
        return norm_id in self.norms

    def article(self, norm_id, number):
        text = self.table.get(number)
        return {"heading": f"Artículo {number}", "text": text, "vigencia": None, "fetched_at": None} if text else None

    def articles(self, norm_id):
        return [{"number": n, "heading": f"Artículo {n}", "text": t} for n, t in self.table.items()]


def test_the_planted_problems_are_flagged_and_nothing_else():
    result = review(text=DRAFT, chunks=[LEASE], law_source=FakeLaw(), aliases=ALIASES)
    kinds = [(i.kind, i.severity, i.paragraph) for i in result.issues]
    assert kinds == [
        ("cita_no_literal", "error", 2),
        ("articulo_inexistente", "error", 4),
        ("articulo_equivocado", "error", 4),
        ("jurisprudencia_sin_verificar", "aviso", 4),
        ("parrafo_repetido", "error", 5),
    ]
    assert result.quotes_checked == 3 and result.citations_checked == 3 and result.errors() == 4
    by_kind = {i.kind: i for i in result.issues}
    assert by_kind["cita_no_literal"].change.after == (
        "indemnizará a la Arrendadora con una cantidad equivalente a una mensualidad de renta por cada año de contrato que reste por cumplir"
    )
    assert by_kind["cita_no_literal"].file == "lease.md" and "una mensualidad" in by_kind["cita_no_literal"].quote
    assert by_kind["articulo_equivocado"].change == Change("artículo 130 de la LEC", "artículo 133 de la LEC", 1)
    assert "Cómputo de los plazos" in by_kind["articulo_equivocado"].reason
    assert by_kind["parrafo_repetido"].reason == "repite lo que ya dice el párrafo 1"
    assert by_kind["parrafo_repetido"].change.after == "" and by_kind["parrafo_repetido"].change.occurrence == 1


def test_applying_the_changes_gives_the_corrected_draft():
    result = review(text=DRAFT, chunks=[LEASE], law_source=FakeLaw(), aliases=ALIASES)
    fixed = apply_changes(DRAFT, [i.change for i in result.issues if i.change])
    assert "dos mensualidades" not in fixed and "una mensualidad de renta por cada año" in fixed
    assert "artículo 133 de la LEC, «En el cómputo" in fixed and "artículo 130" not in fixed
    assert fixed.count("comunicó su voluntad de desistir") == 1 and "\n\n\n" not in fixed
    assert fixed.endswith("SUPLICO que se desestime la demanda.\n")
    # the user rejects the deletion: only the accepted changes are applied
    partial = apply_changes(DRAFT, [i.change for i in result.issues if i.change], accepted={1, 2})
    assert partial.count("comunicó su voluntad de desistir") == 2 and "una mensualidad de renta por cada año" in partial


def test_without_an_archive_or_documents_the_report_says_what_was_not_checked():
    result = review(text="Un párrafo cualquiera con «una cita de veinte caracteres o más» y el artículo 1 de la LEC.")
    assert result.issues == [] and result.quotes_checked == 0 and result.citations_checked == 0
    assert any("BOE" in n for n in result.notes) and any("documentos" in n for n in result.notes)
    assert "Sin hallazgos" in format_report(result)


def test_quotes_paragraphs_and_alignment_helpers():
    assert [
        q
        for q, _ in extract_quotes(
            'Dice «uno dos tres cuatro cinco seis» y "corta" y “otra cita con cuatro palabras más”.'
        )
    ] == [
        "uno dos tres cuatro cinco seis",
        "otra cita con cuatro palabras más",
    ]
    paragraphs = paragraphs_of("A\n\n  B b\n\n\nC\n")
    assert [p[2] for p in paragraphs] == ["A", "B b", "C"] and paragraphs[1][0] == 5
    assert aligned("gato come pescado", "El gato come pescado fresco.") == "gato come pescado"


def test_the_model_pointer_survives_only_when_it_quotes_the_draft():
    llm = FakeChatModel(
        script=[
            Issues(
                issues=[
                    IssueItem(
                        quote="la indemnización reclamada carece de fundamento",
                        reason="no se acredita la fecha de la comunicación",
                    ),
                    IssueItem(quote="una frase que no está en el borrador", reason="inventada"),
                ]
            )
        ]
    )
    result = review(text=DRAFT, llm=llm)
    model = [i for i in result.issues if i.kind == "cuestion_legal"]
    assert len(model) == 1 and model[0].severity == "aviso" and model[0].paragraph == 1
    assert result.windows == 1
    prompt = llm.calls[0][-1].content
    assert prompt.startswith("<draft>") and "SUPLICO" in prompt


def test_the_review_job_anonymises_reviews_and_reveals():
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    memory = FakeMemory()
    memory.chunks = [LEASE]
    memory.indexed_sources = lambda: {"lease.md": {"title": "Contrato de arrendamiento"}}
    import json

    spec = {"text": DRAFT.replace("Arrendataria", "Marta Soler"), "documents": [], "exclude": None, "legal": False}
    job_id = create_job(conn, "review", "escrito.md", json.dumps(spec))
    lock = threading.Lock()
    run_review_job(
        conn,
        job_id,
        memory=memory,
        law_source_factory=FakeLaw,
        aliases=ALIASES,
        llm=None,
        lock=lock,
        anonymize=lambda t: t.replace("Marta Soler", "Arrendataria"),
        reveal=lambda t: t.replace("Arrendataria", "Marta Soler"),
    )
    job = get_job(conn, job_id)
    assert job["status"] == "done" and not lock.locked()
    result = job["result"]
    assert [i["kind"] for i in result["issues"]] == [
        "cita_no_literal",
        "articulo_inexistente",
        "articulo_equivocado",
        "jurisprudencia_sin_verificar",
        "parrafo_repetido",
    ]
    assert "Marta Soler" in result["draft"] and "Arrendataria" not in result["draft"]
    assert "Marta Soler" in result["issues"][4]["change"]["before"] and "Marta Soler" in result["report"]
