"""Chat models from a "<provider>:<model>" spec, plus a scripted fake for tests."""

import os
from collections.abc import Sequence
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import BaseModel

API_KEYS = {"anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY"}


def make_chat_model(spec: str, max_tokens: int, *, reasoning: bool = False) -> BaseChatModel:
    """Build a chat model: "anthropic:claude-sonnet-5-5", "openai:<name>" (OPENAI_BASE_URL) or "ollama:qwen3:8b"."""
    provider, _, name = spec.partition(":")
    key = API_KEYS.get(provider)
    if key and not os.environ.get(key):
        raise RuntimeError(f"{spec} needs {key}. Copy .env.example to .env and set it.")
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(model=name, max_tokens=max_tokens)
    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=name, max_tokens=max_tokens)
    if provider == "ollama":
        from langchain_ollama import ChatOllama

        # Thinking models (qwen3...) spend the output budget on reasoning and then return empty structured
        # output; off by default, LLM_REASONING=true turns it on.
        return ChatOllama(model=name, num_predict=max_tokens, reasoning=reasoning)
    raise ValueError(f"Unknown provider in {spec!r}: use anthropic:, openai: or ollama:")


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
