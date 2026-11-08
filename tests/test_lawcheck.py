"""Legal citations: extracted with patterns, verified against a law source, reported verdict first."""

import os
from pathlib import Path

import pytest

from aroa.lawcheck import Aliases, BoeArchive, Citation, check_text, extract_citations, format_report, verify

ALIASES = Aliases.load(Path(__file__).parent.parent / "data/laws.json")


class FakeSource:
    norms = {
        "BOE-A-2000-323": "Ley 1/2000, de 7 de enero, de Enjuiciamiento Civil.",
        "BOE-A-1994-26003": "Ley 29/1994, de Arrendamientos Urbanos.",
    }
    stored = {"BOE-A-2000-323"}
    articles = {
        ("BOE-A-2000-323", "135"): {
            "heading": "Artículo 135. Presentación de escritos",
            "text": "Artículo 135. Presentación de escritos\n1. Cuando las oficinas...\n5. La presentación de escritos... hasta las quince horas del día hábil siguiente.",
            "vigencia": "2024-03-20",
            "fetched_at": "2026-11-08T12:24:50Z",
        },
        ("BOE-A-2000-323", "133"): {
            "heading": "Artículo 133. Cómputo de los plazos.",
            "text": "Artículo 133. Cómputo de los plazos.\n1. Los plazos...",
            "vigencia": "2010-05-04",
            "fetched_at": "2026-11-08T12:24:50Z",
        },
    }

    def find_norm(self, reference):
        for norm_id, title in self.norms.items():
            if title.lower().startswith(reference.lower() + ","):
                return norm_id, title
        return None

    def is_stored(self, norm_id):
        return norm_id in self.stored

    def article(self, norm_id, number):
        return self.articles.get((norm_id, number))


def test_extraction_covers_abbreviations_numbers_lists_and_judgments():
    text = (
        "Conforme al artículo 135.5 de la LEC y a los arts. 130 y 133 LEC, y según el art. 9 de la Ley 29/1994, "
        "procede. Así lo dijo la STS 1234/2020 (ECLI:ES:TS:2020:1234) y el artículo 1964 del Código Civil."
    )
    cites = extract_citations(text)
    arts = [(c.article, c.apartado, c.law) for c in cites if c.kind == "article"]
    assert ("135", "5", "LEC") in arts and ("130", None, "LEC") in arts and ("133", None, "LEC") in arts
    assert ("9", None, "Ley 29/1994") in arts and ("1964", None, "Código Civil") in arts
    assert [c.raw for c in cites if c.kind == "judgment"] == ["STS 1234/2020", "ECLI:ES:TS:2020:1234"]
    assert any(c.kind == "law" and c.law == "Ley 29/1994" for c in cites)


def test_verdicts_cover_every_outcome():
    text = "art. 135.5 LEC; artículo 999 de la LEC; art. 9 de la Ley 29/1994; artículo 3 de la Ley 12345/1999; art. 7 LOPJ; STS 1/2020; artículo 135.9 LEC"
    results = verify(extract_citations(text), FakeSource(), ALIASES)
    by_raw = {v.citation.raw: v for v in results}
    assert by_raw["art. 135.5 LEC"].status == "found" and by_raw["art. 135.5 LEC"].url.endswith("#a135")
    assert by_raw["artículo 999 de la LEC"].status == "article_not_found"
    assert by_raw["art. 9 de la Ley 29/1994"].status == "law_not_in_archive"  # known norm, not downloaded yet
    assert by_raw["artículo 3 de la Ley 12345/1999"].status == "law_not_found"
    assert by_raw["art. 7 LOPJ"].status == "law_not_in_archive"  # alias resolves, archive lacks it in this fake
    assert by_raw["STS 1/2020"].status == "unverifiable"
    assert by_raw["artículo 135.9 LEC"].status == "found" and "apartado 9" in by_raw["artículo 135.9 LEC"].note


def test_report_puts_the_verdict_first_and_counts():
    results, report = check_text(
        "Según el artículo 135.5 de la LEC y el artículo 999 LEC, y la STS 1/2020.", FakeSource(), ALIASES
    )
    assert report.splitlines()[1].startswith("- ✔ existe · artículo 135.5 de la LEC → Ley 1/2000")
    assert "✘ no existe el artículo · artículo 999 LEC" in report
    assert "quince horas" in report and "https://www.boe.es/buscar/act.php?id=BOE-A-2000-323#a135" in report
    assert report.rstrip().endswith("1 correctas, 1 incorrectas, 1 sin verificar.")
    assert format_report([]).startswith("No se han encontrado citas")


def test_citations_without_a_law_are_reported_not_guessed():
    results = verify([Citation(raw="artículo 24", kind="article", article="24")], FakeSource(), ALIASES)
    assert results[0].status == "law_not_found" and "no dice de qué norma" in results[0].note


@pytest.mark.skipif(not os.environ.get("BOE_ARCHIVE_DIR"), reason="needs the BOE archive (BOE_ARCHIVE_DIR)")
def test_against_the_real_archive():
    source = BoeArchive(Path(os.environ["BOE_ARCHIVE_DIR"]))
    results, report = check_text("artículo 135.5 de la LEC y artículo 1 de la Ley 29/1994", source, ALIASES)
    assert all(v.status == "found" for v in results), report
    assert "quince horas" in report


def test_the_model_hands_the_message_to_the_checker(make_assistant, workdir, monkeypatch):
    from aroa.schemas import Claim, Claims, Facts
    from tests.test_graph import routing, tool_call

    monkeypatch.setattr("aroa.lawcheck.BoeArchive", lambda _dir: FakeSource())
    assistant = make_assistant(
        [
            routing("review"),
            tool_call("check_citations", {}),
            "El artículo 135.5 LEC existe; el 999 no.",
            Claims(claims=[Claim(kind="other", target="citas comprobadas")]),
            Facts(facts=[]),
        ]
    )
    assistant.settings.boe_archive_dir = workdir
    result = assistant.ask("t1", "Revisa las citas: según el artículo 135.5 de la LEC y el artículo 999 LEC procede.")
    assert result.kind == "review"
    seen = assistant.model.calls[2][-1].content
    assert "✔ existe · artículo 135.5 de la LEC" in seen and "✘ no existe el artículo · artículo 999 LEC" in seen


def test_law_names_stop_at_the_first_ordinary_word():
    from aroa.lawcheck import trim_law_name

    assert trim_law_name("LEC procede") == "LEC"
    assert trim_law_name("Código Civil procede la nulidad") == "Código Civil"
    assert trim_law_name("Ley de Enjuiciamiento Civil") == "Ley de Enjuiciamiento Civil"
    assert trim_law_name("Ley 29/1994 de Arrendamientos") == "Ley 29/1994"
    assert trim_law_name("de la") == ""


def test_a_message_citing_an_article_is_routed_to_the_archive_not_the_web(make_assistant, workdir, monkeypatch):
    from aroa.graph import cites_law
    from aroa.schemas import Claim, Claims, Facts
    from tests.test_graph import routing, tool_call

    assert cites_law("¿qué dice el artículo 9 de la LAU?") and not cites_law("¿qué plazo tengo para apelar?")
    from aroa.graph import wants_web

    assert wants_web("busca en el BOE el artículo 1 de la LAU") and not wants_web("comprueba el artículo 1 de la LAU")
    monkeypatch.setattr("aroa.lawcheck.BoeArchive", lambda _dir: FakeSource())
    assistant = make_assistant(
        [
            routing("web"),  # the router wants to search the web; the archive has the law
            tool_call("check_citations", {}),
            "El artículo 135.5 LEC dice...",
            Claims(claims=[Claim(kind="other", target="citas")]),
            Facts(facts=[]),
        ]
    )
    assistant.settings.boe_archive_dir = workdir
    result = assistant.ask("t1", "¿qué dice el artículo 135.5 de la LEC?")
    assert result.kind == "review" and "✔ existe · artículo 135.5 de la LEC" in assistant.model.calls[2][-1].content
