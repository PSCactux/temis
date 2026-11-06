"""Entry point for `langgraph dev` / LangGraph Studio, which supplies its own checkpointer."""

from aroa.app import Assistant
from aroa.config import get_settings

graph = Assistant(get_settings()).graph
