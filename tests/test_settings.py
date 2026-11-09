"""Model settings: the local provider asks for the structure the server can honour; .env edits keep the rest."""

from aroa.models import make_chat_model
from aroa.server import write_env


def test_local_provider_points_at_the_server_and_forces_the_structured_method(monkeypatch):
    model = make_chat_model(
        "local:mlx-community/Qwen3-8B-4bit", 512, base_url="http://127.0.0.1:8082/v1", structured="function_calling"
    )
    assert model.model_name == "mlx-community/Qwen3-8B-4bit" and str(model.openai_api_base).startswith(
        "http://127.0.0.1:8082"
    )
    assert model.extra_body == {"chat_template_kwargs": {"enable_thinking": False}}
    seen = {}

    def fake(self, schema, **kwargs):
        seen.update(kwargs)
        return "runnable"

    monkeypatch.setattr("langchain_openai.ChatOpenAI.with_structured_output", fake)
    model.with_structured_output(dict, method="json_schema")
    assert seen["method"] == "function_calling" and seen["include_raw"] is True
    thinking = make_chat_model("local:x", 64, base_url="http://h/v1", reasoning=True)
    assert not thinking.extra_body


def test_write_env_replaces_values_activates_examples_and_keeps_the_rest(tmp_path):
    env = tmp_path / ".env"
    env.write_text("# comment\nLLM_MODEL=ollama:qwen3:8b\n# LLM_BASE_URL=http://x/v1\nOTHER=1\n", encoding="utf-8")
    write_env(
        {"LLM_MODEL": "local:qwen3-8b", "LLM_BASE_URL": "http://127.0.0.1:8080/v1", "LLM_STRUCTURED": "json_schema"},
        env,
    )
    assert env.read_text(encoding="utf-8") == (
        "# comment\nLLM_MODEL=local:qwen3-8b\nLLM_BASE_URL=http://127.0.0.1:8080/v1\nOTHER=1\nLLM_STRUCTURED=json_schema\n"
    )


def test_a_prose_reply_is_read_for_json_and_otherwise_retried_once():
    from langchain_core.messages import AIMessage, HumanMessage

    from aroa.models import _structured_with_retry
    from aroa.privacy import Names

    class Inner:
        def __init__(self, outs):
            self.outs, self.calls = list(outs), []

        def invoke(self, messages):
            self.calls.append(messages)
            return self.outs.pop(0)

    prose = Inner(
        [{"parsed": None, "raw": AIMessage('Here you go: {"people": ["Ana Ruiz"], "organizations": []} hope it helps')}]
    )
    got = _structured_with_retry(prose, Names, [HumanMessage("x")])
    assert got.people == ["Ana Ruiz"] and len(prose.calls) == 1
    retry = Inner([{"parsed": None, "raw": AIMessage("No names here.")}, {"parsed": Names(people=["B"]), "raw": None}])
    assert _structured_with_retry(retry, Names, [HumanMessage("x")]).people == ["B"]
    assert "calling the Names tool" in retry.calls[1][-1].content
    nothing = Inner([{"parsed": None, "raw": AIMessage("no")}, {"parsed": None, "raw": AIMessage("still no")}])
    assert _structured_with_retry(nothing, Names, [HumanMessage("x")]) is None


def test_name_detection_without_an_answer_still_replaces_identifiers(tmp_path):
    import sqlite3

    from aroa.privacy import Anonymizer

    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    out = Anonymizer(conn, extract_names=lambda text: None).anonymize("Ana Ruiz, DNI 12345678Z, tel. 612345678")
    assert "12345678Z" not in out and "612345678" not in out and "[DNI_1]" in out
