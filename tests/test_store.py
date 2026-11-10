"""The real memory store: needs the embedding models on disk, so it runs only with AROA_INTEGRATION=1."""

import os

import pytest

from aroa.config import Settings
from aroa.memory.store import MemoryStore

pytestmark = pytest.mark.skipif(not os.environ.get("AROA_INTEGRATION"), reason="set AROA_INTEGRATION=1 to run")


@pytest.fixture
def store(tmp_path):
    return MemoryStore(Settings(data_dir=tmp_path / ".data", _env_file=None))


def test_recall_respects_the_wall_between_matters(store):
    store.add_facts(["[PERSON_1] es el director financiero de [ORG_1]"], thread_id="a", turn_id="1", matter="m-a")
    store.add_facts(["El despacho cierra en agosto"], thread_id="b", turn_id="2", matter="")
    store.add_facts(["[PERSON_9] lleva la contabilidad de [ORG_9]"], thread_id="c", turn_id="3", matter="m-b")
    texts = [f["text"] for f in store.recall("¿quién lleva la contabilidad o las finanzas?", matter="m-b")["facts"]]
    assert "[PERSON_1] es el director financiero de [ORG_1]" not in texts
    assert "[PERSON_9] lleva la contabilidad de [ORG_9]" in texts
    texts = [f["text"] for f in store.recall("¿en qué mes cierra el despacho?", matter="m-b")["facts"]]
    assert texts == ["El despacho cierra en agosto"]  # general memory crosses the wall


def test_unrelated_memory_is_not_recalled(store):
    store.add_facts(
        ["El titular de la finca es [PERSON_2]", "[ORG_3] tiene una hipoteca sobre la finca"],
        thread_id="a",
        turn_id="1",
    )
    assert store.recall("hola, ¿qué tal?")["facts"] == []
    assert store.recall("¿qué plazo hay para recurrir la sentencia?")["facts"] == []
    texts = [f["text"] for f in store.recall("¿quién es el titular de la finca?")["facts"]]
    assert texts[0] == "El titular de la finca es [PERSON_2]"


def test_duplicate_facts_are_skipped(store):
    stored, skipped = store.add_facts(["El cliente prefiere un acuerdo"], thread_id="a", turn_id="1")
    assert stored and not skipped
    stored, skipped = store.add_facts(["El cliente prefiere llegar a un acuerdo"], thread_id="a", turn_id="2")
    assert skipped and not stored
