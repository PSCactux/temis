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
    texts = [f["text"] for f in store.recall("¿quién es el director financiero?", matter="m-b")["facts"]]
    assert "[PERSON_1] es el director financiero de [ORG_1]" not in texts
    assert "El despacho cierra en agosto" in texts and "[PERSON_9] lleva la contabilidad de [ORG_9]" in texts


def test_duplicate_facts_are_skipped(store):
    stored, skipped = store.add_facts(["El cliente prefiere un acuerdo"], thread_id="a", turn_id="1")
    assert stored and not skipped
    stored, skipped = store.add_facts(["El cliente prefiere llegar a un acuerdo"], thread_id="a", turn_id="2")
    assert skipped and not stored
