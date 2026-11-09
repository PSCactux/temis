"""Procedures: structured records checked before saving, data filled in code, marked blocks written by the model,
a checklist in the review."""

from pathlib import Path

from langchain_core.messages import AIMessage

from aroa.models import FakeChatModel
from aroa.procedures import (
    Field,
    check,
    check_data_present,
    draft,
    fill,
    from_dict,
    load_all,
    parse,
    read_procedure,
    save,
    slugify,
    write_blocks,
)
from aroa.review import review
from aroa.schemas import Checklist, ChecklistItem

SAMPLE = Path(__file__).parent.parent / "data/procedures_samples/reclamacion-patrimonial.json"
TEMPLATE = Path(__file__).parent.parent / "data/samples/plantilla_reclamacion_patrimonial.md"


def test_the_sample_procedure_loads_from_its_record():
    proc = read_procedure(SAMPLE)
    assert proc.slug == "reclamacion-patrimonial" and proc.name == "Reclamación patrimonial"
    assert proc.description.startswith("Reclamación de responsabilidad patrimonial")
    assert proc.template_ref == "plantilla_reclamacion_patrimonial.md" and proc.template_text == ""
    assert [f.name for f in proc.fields][:3] == [
        "Administración reclamada",
        "Reclamante",
        "Domicilio a efectos de notificaciones",
    ]
    assert proc.fields[0].hint.startswith("órgano al que se dirige") and proc.fields[5].long
    assert "Registro formal" in proc.instructions and len(proc.checklist) == 9
    assert proc.checklist[5] == "Cita el artículo 32 de la Ley 40/2015 y el artículo 67 de la Ley 39/2015"


def test_headings_are_matched_without_accents_and_an_inline_template_is_kept():
    proc = parse(
        "# Carta\n\nIntro.\n\n## PLANTILLA\n\nEstimado {{Nombre}}:\n\n[[Saluda]]\n\n## Datos\n\n- Nombre\n\n## Revisión\n\n- [ ] Saluda por su nombre\n",
        "carta",
    )
    assert proc.template_text.startswith("Estimado {{Nombre}}") and proc.fields[0].name == "Nombre"
    assert (
        proc.checklist == ["Saluda por su nombre"] and slugify("Reclamación Patrimonial") == "reclamacion-patrimonial"
    )


def test_placeholders_are_filled_in_code_and_missing_ones_reported():
    text, missing = fill(
        "A {{Administración reclamada}}, firmado {{Reclamante}} en {{Lugar}}.",
        {"administracion reclamada": "Ayuntamiento de Valencia", "Reclamante": "Marta Soler"},
    )
    assert text == "A Ayuntamiento de Valencia, firmado Marta Soler en {{Lugar}}." and missing == ["Lugar"]
    assert check_data_present(text, {"Reclamante": "Marta Soler", "Cuantía": "3.200 euros"}) == ["Cuantía"]


def test_blocks_are_written_by_the_model_and_bad_replies_rejected():
    proc = read_procedure(SAMPLE)
    llm = FakeChatModel(
        script=[
            AIMessage("{{sin rellenar}}"),
            AIMessage("El reclamante sufrió una fractura de muñeca (documento n.º 1)."),
            AIMessage(""),
            AIMessage("   "),
        ]
    )
    text, written = write_blocks(
        "PRIMERO. Hecho.\n\nSEGUNDO. [[Describe el daño]]\n\nTERCERO. [[Causalidad]]\n",
        llm=llm,
        procedure=proc,
        data={"Daños sufridos": "fractura de muñeca"},
    )
    assert written == 1 and "SEGUNDO. El reclamante sufrió una fractura de muñeca (documento n.º 1)." in text
    assert "TERCERO. [[Causalidad]]" in text  # two bad replies: the instruction stays visible, nothing is invented
    prompt = llm.calls[0][-1].content
    assert prompt.startswith('<procedure name="Reclamación patrimonial">') and "<facts>" in prompt and "[[…]]" in prompt


def test_draft_fills_the_sample_template_and_checks_the_data_back():
    proc = read_procedure(SAMPLE)
    data = {
        "Administración reclamada": "Ayuntamiento de Valencia",
        "Reclamante": "Marta Soler Vidal, DNI 11223344A",
        "Domicilio a efectos de notificaciones": "calle Colón 8, Valencia",
        "Fecha del hecho lesivo": "3 de febrero de 2026",
        "Lugar del hecho": "la calle de la Paz, frente al número 12",
        "Descripción del hecho": "tropezó con una baldosa levantada de la acera y cayó al suelo",
        "Servicio público implicado": "conservación de la vía pública",
        "Daños sufridos": "fractura del radio izquierdo",
        "Cuantía reclamada": "4.850 euros (cuatro mil ochocientos cincuenta euros)",
        "Documentos que se aportan": "1. Informe de urgencias. 2. Fotografías de la acera.",
    }
    llm = FakeChatModel(script=[AIMessage(f"Pasaje {i} redactado con fractura del radio izquierdo.") for i in range(3)])
    result = draft(proc, data, llm=llm, template_text=TEMPLATE.read_text(encoding="utf-8"))
    assert result["missing_fields"] == [] and result["blocks"] == 3 and result["blocks_written"] == 3
    assert result["data_unused"] == ["Servicio público implicado"]  # only named inside a block's instructions
    assert "A Ayuntamiento de Valencia" in result["text"] and "{{" not in result["text"] and "[[" not in result["text"]
    assert "Fdo.: Marta Soler Vidal, DNI 11223344A" in result["text"]


def test_the_review_checks_the_procedure_checklist_with_verbatim_quotes():
    text = "PRIMERO. El día 3 de febrero de 2026, en la calle de la Paz, la reclamante cayó.\n\nSOLICITO que se reconozca la indemnización.\n"
    from aroa.schemas import Issues

    llm = FakeChatModel(
        script=[
            Issues(),
            Checklist(
                items=[
                    ChecklistItem(
                        item="Fecha y lugar del hecho constan",
                        met=True,
                        evidence="El día 3 de febrero de 2026, en la calle de la Paz",
                    ),
                    ChecklistItem(item="Cuantifica el daño en euros", met=False, note="no hay cuantía"),
                    ChecklistItem(
                        item="Termina con una petición", met=True, evidence="texto que no está en el borrador"
                    ),
                ]
            ),
        ]
    )
    result = review(text=text, llm=None)
    assert result.checklist == [] and any("hace falta el modelo" in n for n in result.notes) is False
    result = review(
        text=text,
        llm=llm,
        checklist=["Fecha y lugar del hecho constan", "Cuantifica el daño en euros", "Termina con una petición"],
    )
    met = {v["item"]: v["met"] for v in result.checklist}
    assert met == {
        "Fecha y lugar del hecho constan": True,
        "Cuantifica el daño en euros": False,
        "Termina con una petición": False,
    }
    unmet = [i for i in result.issues if i.kind == "lista_revision"]
    assert [i.passage for i in unmet] == ["Cuantifica el daño en euros", "Termina con una petición"] and unmet[
        0
    ].reason == "no hay cuantía"
    from aroa.review import format_report

    report = format_report(result)
    assert (
        "## Lista de revisión" in report
        and "✔ Fecha y lugar del hecho constan — «El día 3 de febrero de 2026, en la calle de la Paz»" in report
    )


def test_samples_are_copied_on_first_load(tmp_path):
    procs = load_all(tmp_path / "procedures", SAMPLE.parent)
    assert [p.slug for p in procs] == ["reclamacion-patrimonial"] and (
        tmp_path / "procedures/reclamacion-patrimonial.json"
    ).is_file()


def test_the_sample_passes_its_check_against_its_template():
    result = check(read_procedure(SAMPLE), TEMPLATE.read_text(encoding="utf-8"))
    assert result["errors"] == [] and result["blocks"] == 3 and "Cuantía reclamada" in result["placeholders"]
    assert result["warnings"] == []  # «Daños sufridos» and «Servicio público implicado» are named by the blocks


def test_check_stops_a_procedure_that_cannot_be_followed():
    proc = from_dict(
        {
            "name": "Carta",
            "template": "../fuera/carta.md",
            "fields": [
                {"name": "Nombre"},
                {"name": " nombre "},
                {"name": "Fecha {x}"},
                {"name": "", "hint": "sin nombre"},
            ],
            "checklist": ["  ", "Saluda"],
        },
        "carta",
    )
    assert proc.template_ref == "carta.md" and proc.checklist == ["Saluda"]  # no paths, no blank rows
    result = check(proc, "Estimado {{Nombre}} y {{Cargo}}:\n\n[[Saluda]]", taken={"carta"})
    assert result["errors"] == [
        {"code": "name_taken", "name": "Carta"},
        {"code": "field_repeated", "name": "nombre"},
        {"code": "field_brackets", "name": "Fecha {x}"},
        {"code": "field_unnamed", "n": 4},
        {"code": "placeholder_without_field", "name": "Cargo"},
    ]
    assert {"code": "no_instructions"} in result["warnings"]
    gone = check(from_dict({"name": "Carta", "template": "no_esta.md"}, "carta"), None)
    assert gone["errors"] == [{"code": "template_missing", "name": "no_esta.md"}]
    assert check(from_dict({}, "x"), None)["errors"] == [{"code": "no_name"}, {"code": "no_template"}]


def test_saving_writes_the_record_and_replaces_a_markdown_procedure(tmp_path):
    md = tmp_path / "carta.md"
    md.write_text(
        "# Carta\n\nPrimera línea\ncortada.\n\n## Plantilla\n\ndocumento: carta.md\n\n## Datos\n\n- Nombre\n",
        encoding="utf-8",
    )
    proc = read_procedure(md)
    assert proc.description == "Primera línea cortada."  # hard-wrapped Markdown comes back as a paragraph
    proc.fields.append(Field("Hechos", "qué pasó", True))
    path = save(tmp_path, proc)
    assert path.name == "carta.json" and not md.exists() and proc.source == "carta.json"
    again = read_procedure(path)
    assert again.name == "Carta" and again.template_ref == "carta.md"
    assert [(f.name, f.long) for f in again.fields] == [("Nombre", False), ("Hechos", True)]


def test_the_api_refuses_to_save_or_follow_a_procedure_with_errors(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from fastapi.testclient import TestClient

    from aroa import server

    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "carta.md").write_text("Nota interna\n\n---\n\nEstimado {{Nombre}}:\n\n[[Saluda]]\n", encoding="utf-8")
    settings = SimpleNamespace(procedures_dir=tmp_path / "procs", procedures_samples_dir=None, docs_dir=docs)
    monkeypatch.setattr(server, "runtime", SimpleNamespace(assistant=SimpleNamespace(settings=settings)))
    client = TestClient(server.app)  # no context manager: the real runtime is never started
    record = {"name": "Carta", "template": "carta.md", "fields": [], "instructions": "Breve.", "checklist": ["Saluda"]}
    r = client.post("/api/procedures", json=record)
    assert r.status_code == 422 and r.json()["detail"]["check"]["errors"] == [
        {"code": "placeholder_without_field", "name": "Nombre"}
    ]
    assert not (tmp_path / "procs/carta.json").exists()
    record["fields"] = [{"name": "Nombre", "hint": "como firma"}]
    r = client.post("/api/procedures", json=record)
    assert r.status_code == 200 and r.json()["slug"] == "carta" and (tmp_path / "procs/carta.json").is_file()
    assert client.post("/api/procedures", json=record).json()["detail"]["check"]["errors"][0]["code"] == "name_taken"
    assert client.put("/api/procedures/carta", json=record).status_code == 200  # the same one, saved again
    (docs / "carta.md").write_text("Estimado {{Nombre}} {{Apellido}}:\n", encoding="utf-8")  # the template changed
    assert client.get("/api/procedures/carta").json()["check"]["errors"] == [
        {"code": "placeholder_without_field", "name": "Apellido"}
    ]
    r = client.post("/api/procedures/carta/draft", json={"data": {"Nombre": "Ana"}})
    assert r.status_code == 409
