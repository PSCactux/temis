"""Editable documents: HTML to text for the index, Word in and out, versions on save, names kept inside the folder."""

from io import BytesIO

import pytest

from aroa.documents import (
    docx_to_html,
    html_to_docx,
    html_to_text,
    restore,
    safe_name,
    save_html,
    text_to_html,
    version_html,
    versions,
)
from aroa.memory.loaders import document_files, load_file, read_text

HTML = (
    "<h1>Contrato de servicios</h1><p>Entre <strong>Acme, S.L.</strong> y <em>la Cliente</em>.</p>"
    "<h2>Cláusula 1.ª Precio</h2><p>El precio es de 4.500 euros.<br>Pagadero a treinta días.</p>"
    "<ul><li>Primera</li><li>Segunda</li></ul><ol><li>Uno</li><li>Dos</li></ol>"
    "<blockquote><p>Texto citado.</p></blockquote>"
    "<table><tr><th>Concepto</th><th>Importe</th></tr><tr><td>Licencia</td><td>1.000</td></tr></table>"
    '<p style="text-align: center;">Firmado.</p><script>alert(1)</script>'
)


def test_html_becomes_the_markdown_like_text_the_chunker_understands():
    text = html_to_text(HTML)
    assert text.startswith("# Contrato de servicios\n\nEntre Acme, S.L. y la Cliente.\n\n## Cláusula 1.ª Precio\n\n")
    assert "El precio es de 4.500 euros.\nPagadero a treinta días." in text
    assert "- Primera\n\n- Segunda" in text and "1. Uno\n\n2. Dos" in text
    assert "> Texto citado." in text and "Concepto | Importe\n\nLicencia | 1.000" in text
    assert "alert" not in text and text.endswith("Firmado.\n")


def test_word_round_trip_keeps_the_structure(tmp_path):
    path = tmp_path / "contrato.docx"
    path.write_bytes(html_to_docx(HTML, title="Contrato"))
    from docx import Document

    doc = Document(BytesIO(path.read_bytes()))
    styles = [p.style.name for p in doc.paragraphs if p.text.strip()]
    assert styles[0] == "Heading 1" and "Heading 2" in styles and "List Bullet" in styles and "List Number" in styles
    assert "Quote" in styles
    bold = [r.text for p in doc.paragraphs for r in p.runs if r.bold]
    assert "Acme, S.L." in bold
    assert len(doc.tables) == 1 and doc.tables[0].cell(1, 0).text == "Licencia"
    assert doc.core_properties.title == "Contrato"
    back = docx_to_html(path)
    assert "<h1>Contrato de servicios</h1>" in back and "<strong>Acme, S.L.</strong>" in back and "<table>" in back
    assert read_text(path).startswith("# Contrato de servicios")
    chunks = load_file(path, max_chars=1200)
    assert chunks[0].title == "Contrato de servicios" and any("4.500" in c.text for c in chunks)


def test_saving_keeps_versions_and_restores_them(tmp_path):
    docs = tmp_path / "docs"
    first = save_html(docs, "Escrito de oposición", "<p>v1</p>")
    assert first.name == "Escrito_de_oposición.html" and versions(docs, first.name) == []
    save_html(docs, first.name, "<p>v2</p>")
    kept = versions(docs, first.name)
    assert len(kept) == 1 and version_html(docs, first.name, kept[0]["version"]) == "<p>v1</p>"
    assert restore(docs, first.name, kept[0]["version"]).read_text() == "<p>v1</p>"
    assert len(versions(docs, first.name)) == 2  # the restore kept v2 as a version too
    assert restore(docs, first.name, "nope") is None
    # the index never sees the versions folder
    assert [p.name for p in document_files(docs)] == [first.name]
    assert read_text(first) == "v1\n"


def test_names_stay_inside_the_folder():
    assert safe_name("../../etc/passwd") == "passwd.html"
    assert safe_name("Demanda (borrador) 2026.html") == "Demanda_borrador_2026.html"
    assert safe_name("informe", ".docx") == "informe.docx"
    with pytest.raises(ValueError):
        safe_name("..")


def test_text_to_html_for_the_editor():
    html = text_to_html("# Título\n\nUn **párrafo**\ncon dos líneas.\n\n- a\n- b\n\n1. uno\n\n> cita\n")
    assert html.startswith("<h1>Título</h1>\n<p>Un <strong>párrafo</strong><br>con dos líneas.</p>")
    assert "<ul>\n<li>a</li>\n<li>b</li>\n</ul>" in html and "<ol>\n<li>uno</li>\n</ol>" in html
    assert html.endswith("<blockquote><p>cita</p></blockquote>")
    assert html_to_text(text_to_html("<b>no es html</b>")) == "<b>no es html</b>\n"
