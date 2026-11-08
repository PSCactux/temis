"""Chat models from a "<provider>:<model>" spec, plus a scripted fake for tests."""

import json
import os
import subprocess
import uuid
from collections.abc import Sequence
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import BaseModel

API_KEYS = {"anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY"}


def make_chat_model(spec: str, max_tokens: int, *, reasoning: bool = False) -> BaseChatModel:
    """Build a chat model: "anthropic:claude-sonnet-5-5", "openai:<name>" (OPENAI_BASE_URL), "ollama:qwen3:8b"
    or "claude-cli:sonnet" (the Claude Code CLI on a subscription, for evaluation on one's own machine)."""
    provider, _, name = spec.partition(":")
    key = API_KEYS.get(provider)
    if key and not os.environ.get(key):
        raise RuntimeError(f"{spec} needs {key}. Copy .env.example to .env and set it.")
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(model=name, max_tokens=max_tokens, temperature=0)
    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=name, max_tokens=max_tokens, temperature=0)
    if provider == "ollama":
        from langchain_ollama import ChatOllama

        # Thinking models (qwen3...) spend the output budget on reasoning and then return empty structured
        # output; off by default, LLM_REASONING=true turns it on.
        # Temperature 0 everywhere: routing, grading and claims must be reproducible, and the eval numbers too.
        return ChatOllama(model=name, num_predict=max_tokens, reasoning=reasoning, temperature=0)
    if provider == "claude-cli":
        return ClaudeCliChatModel(model=name or "sonnet", max_tokens=max_tokens)
    raise ValueError(f"Unknown provider in {spec!r}: use anthropic:, openai:, ollama: or claude-cli:")


TOOL_CALL_SCHEMA = {
    "type": "object",
    "properties": {
        "reply": {"type": "string", "description": "What to say to the user; empty when calling tools"},
        "tool_calls": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"name": {"type": "string"}, "arguments": {"type": "object"}},
                "required": ["name", "arguments"],
            },
        },
    },
    "required": ["reply", "tool_calls"],
}


class ClaudeCliChatModel(BaseChatModel):
    """Claude through the Claude Code CLI in print mode: one process per call, no API key.

    Structured output uses the CLI's `--json-schema`; tool calls are emulated with a schema that asks
    for `reply` or `tool_calls`, since the CLI exposes no function-calling interface. Meant to run the
    evals on a subscription from one's own machine; a deployment uses the `anthropic:` provider."""

    model: str = "sonnet"
    max_tokens: int = 4096
    timeout_seconds: int = 600
    bound_tools: list = []
    expected_schema: Any = None

    @property
    def _llm_type(self) -> str:
        return "claude-cli"

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any):
        clone = self.model_copy()
        clone.bound_tools = list(tools)
        return clone

    def with_structured_output(self, schema, **kwargs: Any):
        clone = self.model_copy()
        clone.expected_schema = schema
        return clone | (lambda message: schema(**message.additional_kwargs["parsed"]))

    def _run(self, prompt: str, system: str, schema: dict | None) -> dict:
        args = ["claude", "-p", prompt, "--output-format", "json", "--model", self.model, "--tools", ""]
        args += ["--no-session-persistence", "--setting-sources", ""]  # no hooks, plugins or project settings
        if system:
            args += ["--system-prompt", system]
        if schema:
            args += ["--json-schema", json.dumps(schema)]
        done = subprocess.run(args, capture_output=True, text=True, timeout=self.timeout_seconds)
        if done.returncode != 0:
            raise RuntimeError(f"claude CLI failed ({done.returncode}): {done.stderr.strip()[:500]}")
        data = json.loads(done.stdout)
        if data.get("is_error"):
            raise RuntimeError(f"claude CLI error: {str(data.get('result'))[:500]}")
        return data

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        system, prompt = transcript(messages)
        schema = None
        if self.expected_schema is not None:
            schema = self.expected_schema.model_json_schema()
        elif self.bound_tools:
            schema = TOOL_CALL_SCHEMA
            system += "\n\n" + describe_tools(self.bound_tools)
        data = self._run(prompt, system, schema)
        usage = data.get("usage") or {}
        if self.expected_schema is not None:
            message = AIMessage("", additional_kwargs={"parsed": _structured(data)})
        elif self.bound_tools:
            out = _structured(data)
            names = {t.name for t in self.bound_tools}
            calls = [
                {"name": c["name"], "args": c.get("arguments") or {}, "id": f"call_{uuid.uuid4().hex[:8]}"}
                for c in out.get("tool_calls") or []
                if c.get("name") in names
            ]
            message = AIMessage(out.get("reply") or "", tool_calls=calls)
        else:
            message = AIMessage(str(data.get("result") or ""))
        message.usage_metadata = {
            "input_tokens": int(usage.get("input_tokens", 0))
            + int(usage.get("cache_read_input_tokens", 0))
            + int(usage.get("cache_creation_input_tokens", 0)),
            "output_tokens": int(usage.get("output_tokens", 0)),
            "total_tokens": 0,
        }
        message.usage_metadata["total_tokens"] = (
            message.usage_metadata["input_tokens"] + message.usage_metadata["output_tokens"]
        )
        message.response_metadata = {"cost_usd": data.get("total_cost_usd"), "model": self.model}
        return ChatResult(generations=[ChatGeneration(message=message)])


def _structured(data: dict) -> dict:
    """The CLI's structured output; when the field is missing, the result text itself if it is JSON."""
    out = data.get("structured_output")
    if isinstance(out, dict) and out:
        return out
    try:
        parsed = json.loads(str(data.get("result") or ""))
    except ValueError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def transcript(messages: Sequence[BaseMessage]) -> tuple[str, str]:
    """System text and a flat transcript of the conversation for a model that takes one prompt."""
    system, lines = [], []
    for m in messages:
        if isinstance(m, SystemMessage):
            system.append(str(m.content))
        elif isinstance(m, HumanMessage):
            lines.append(f"User:\n{m.content}")
        elif isinstance(m, ToolMessage):
            lines.append(f"Tool result ({m.tool_call_id}):\n{m.content}")
        elif isinstance(m, AIMessage):
            text = str(m.content)
            if m.tool_calls:
                text += "\n" + "\n".join(
                    f"[called {c['name']} {json.dumps(c['args'], ensure_ascii=False)} -> {c['id']}]"
                    for c in m.tool_calls
                )
            lines.append(f"Assistant:\n{text}")
    lines.append("Assistant:")
    return "\n\n".join(system), "\n\n".join(lines)


def describe_tools(tools: Sequence[Any]) -> str:
    rows = [f"- {t.name}: {t.description}\n  arguments: {json.dumps(t.args, ensure_ascii=False)}" for t in tools]
    return (
        "## Tool calling\nYou can call these tools by returning them in `tool_calls` (name and arguments); "
        "leave `reply` empty when you call tools, and put the final answer in `reply` with no tool calls "
        "when the work is done.\n" + "\n".join(rows)
    )


class FakeChatModel(BaseChatModel):
    """Replies from a script: each entry is a string, an AIMessage, or a callable(messages) -> either.

    Structured outputs are honoured when the script entry is a Pydantic instance of the requested schema.
    The graph's control flow is tested with this model, so tests need neither network nor weights.
    """

    script: list[Any]
    calls: list[list[BaseMessage]] = []
    bound_tools: list = []
    expected_schema: Any = None

    @property
    def _llm_type(self) -> str:
        return "fake"

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any):
        clone = self.model_copy()
        clone.bound_tools = list(tools)
        return clone

    def with_structured_output(self, schema, **kwargs: Any):
        clone = self.model_copy()
        clone.expected_schema = schema
        return clone | _unwrap_structured

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.calls.append(messages)
        if not self.script:
            raise AssertionError(f"FakeChatModel ran out of script; last messages: {messages[-1].content!r}")
        entry = self.script.pop(0)
        if callable(entry) and not isinstance(entry, BaseModel):
            entry = entry(messages)
        if isinstance(entry, str):
            entry = AIMessage(entry)
        elif isinstance(entry, BaseModel) and not isinstance(entry, BaseMessage):
            entry = AIMessage("", additional_kwargs={"parsed": entry})
        entry.usage_metadata = {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
        return ChatResult(generations=[ChatGeneration(message=entry)])


def _unwrap_structured(message: AIMessage):
    parsed = message.additional_kwargs.get("parsed")
    if parsed is None:
        raise AssertionError("FakeChatModel structured call needs a Pydantic entry in the script")
    return parsed
