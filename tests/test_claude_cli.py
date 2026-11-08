"""The claude-cli provider: one prompt per call, structured output through the CLI's schema, tool calls emulated."""

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

from aroa.models import ClaudeCliChatModel, transcript
from aroa.schemas import Routing


@tool
def create_alarm(text: str, due: str) -> str:
    """Create an alarm."""
    return "ok"


class Scripted(ClaudeCliChatModel):
    seen: list = []
    answer: dict = {}

    def _run(self, prompt, system, schema):
        self.seen.append({"prompt": prompt, "system": system, "schema": schema})
        return {"result": "hola", "structured_output": self.answer, "usage": {"input_tokens": 3, "output_tokens": 2}}


def test_transcript_flattens_roles_and_tool_calls():
    system, prompt = transcript(
        [
            SystemMessage("Be brief."),
            HumanMessage("pon una alarma"),
            AIMessage("", tool_calls=[{"name": "create_alarm", "args": {"text": "x"}, "id": "c1"}]),
            ToolMessage('<tool_result tool="create_alarm">\nok\n</tool_result>', tool_call_id="c1"),
        ]
    )
    assert system == "Be brief."
    assert prompt.startswith("User:\npon una alarma\n\nAssistant:\n\n[called create_alarm")
    assert "Tool result (c1):" in prompt and prompt.endswith("Assistant:")


def test_structured_output_goes_through_the_schema():
    answer = {"route": "alarms", "search_query": "alarma", "project": "", "policy": "allowed", "policy_reason": ""}
    model = Scripted(answer=answer)
    result = model.with_structured_output(Routing, method="json_schema").invoke([HumanMessage("avísame")])
    assert isinstance(result, Routing) and result.route == "alarms"
    assert model.seen[-1]["schema"]["title"] == "Routing"


def test_tool_calls_are_emulated_and_unknown_tools_dropped():
    calls = [{"name": "create_alarm", "arguments": {"text": "x", "due": "y"}}, {"name": "rm", "arguments": {}}]
    model = Scripted(answer={"reply": "", "tool_calls": calls})
    reply = model.bind_tools([create_alarm]).invoke([HumanMessage("pon una alarma")])
    assert [c["name"] for c in reply.tool_calls] == ["create_alarm"]
    assert reply.tool_calls[0]["args"] == {"text": "x", "due": "y"} and reply.tool_calls[0]["id"].startswith("call_")
    assert "## Tool calling" in model.seen[-1]["system"] and "create_alarm" in model.seen[-1]["system"]
    assert model.seen[-1]["schema"]["properties"]["tool_calls"]


def test_plain_reply_and_usage():
    model = Scripted()
    reply = model.invoke([HumanMessage("hola")])
    assert reply.content == "hola" and reply.usage_metadata["input_tokens"] == 3
