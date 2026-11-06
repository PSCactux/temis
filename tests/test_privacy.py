"""Pseudonymisation: stable placeholders, reveal, redaction, names from the extractor, identifiers in code."""

import sqlite3

from aroa.privacy import Anonymizer, Names, contains_personal_data


def make(extractor=None) -> Anonymizer:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    return Anonymizer(conn, extractor)


def test_identifiers_get_stable_placeholders():
    a = make()
    first = a.anonymize("DNI 12345678Z, móvil 612 345 678, correo ana@example.com, IBAN ES91 2100 0418 4502 0005 1332")
    assert contains_personal_data(first) == []
    assert "[DNI_1]" in first and "[PHONE_1]" in first and "[EMAIL_1]" in first and "[IBAN_1]" in first
    again = a.anonymize("Otra vez el DNI 12345678Z y otro 87654321X")
    assert "[DNI_1]" in again and "[DNI_2]" in again
    assert a.reveal(again) == "Otra vez el DNI 12345678Z y otro 87654321X"


def test_names_come_from_the_extractor_and_are_remembered():
    calls = []

    def extractor(text):
        calls.append(text)
        return Names(people=["Ana García López", "García López"], organizations=["Talleres Ribera S.L."])

    a = make(extractor)
    text = a.anonymize("Ana García López, de Talleres Ribera S.L., llamó. La Sra. García López insiste.")
    # "García López" is part of "Ana García López": same person, same placeholder
    assert text == "[PERSON_1], de [ORG_1], llamó. La Sra. [PERSON_1] insiste."
    # a later text with no extractor call still replaces the names already known
    later = a.anonymize("Reunión con Talleres Ribera S.L. mañana", names=False)
    assert later == "Reunión con [ORG_1] mañana" and len(calls) == 1
    # one placeholder per person, so a short form is revealed as the full name
    assert a.reveal(text) == "Ana García López, de Talleres Ribera S.L., llamó. La Sra. Ana García López insiste."


def test_redaction_is_irreversible():
    a = make()
    redacted = a.redact("Teléfono 600111222 y DNI 11111111H.")
    assert redacted == "Teléfono ████████ y DNI ████████."
    assert a.reveal(redacted) == redacted


def test_dates_and_amounts_are_not_touched():
    a = make()
    assert a.anonymize("El 12/05/2026 pagó 4.500 euros y el plazo es de 20 días.") == (
        "El 12/05/2026 pagó 4.500 euros y el plazo es de 20 días."
    )


def test_role_words_and_public_bodies_are_never_names():
    from aroa.privacy import plausible_name

    assert plausible_name("la Arrendadora", "PERSON") is None
    assert plausible_name("El Cliente", "ORG") is None
    assert plausible_name("Instituto Nacional de Estadística", "ORG") is None
    assert plausible_name("Juzgado de Primera Instancia n.º 3", "ORG") is None
    assert plausible_name("Sra. García López", "PERSON") == "García López"
    assert plausible_name("Lucía Ferrer Gómez", "PERSON") == "Lucía Ferrer Gómez"
    assert plausible_name("Talleres Ribera, S.L.", "ORG") == "Talleres Ribera, S.L."
    assert plausible_name("NubeSegura Tecnologías", "ORG") == "NubeSegura Tecnologías"
    assert plausible_name("proveedor", "ORG") is None
