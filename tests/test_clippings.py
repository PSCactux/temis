"""A clipping is a small document: file in the documents folder, header with its origin, text verbatim."""

from pathlib import Path

import pytest

from aroa.clippings import is_clipping, write
from aroa.memory.chunking import chunk_document


def test_clipping_file_keeps_origin_note_and_verbatim_text(tmp_path: Path):
    name = write(
        tmp_path,
        source="contrato.md",
        source_title="Contrato de arrendamiento",
        text="  La renta anual es de 13.200 euros.  ",
        note="para la demanda",
        matter="2026-014",
    )
    assert is_clipping(name) and name.startswith("recorte-contrato-de-arrendamiento-") and name.endswith(".md")
    body = (tmp_path / name).read_text()
    assert body.startswith("# Recorte · Contrato de arrendamiento: «La renta anual es de 13.200 euros.»")
    assert "(`contrato.md`)" in body and "Expediente 2026-014" in body and "**Nota:** para la demanda" in body
    assert "\nLa renta anual es de 13.200 euros.\n" in body
    chunks = chunk_document(body, title="x", source=name, max_chars=1800)
    assert any("13.200" in c.text for c in chunks)  # indexed like any document, so it can be cited


def test_empty_clipping_is_refused(tmp_path: Path):
    with pytest.raises(ValueError):
        write(tmp_path, source="a.md", source_title="A", text="   ")
