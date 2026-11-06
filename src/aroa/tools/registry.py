"""Which tools each route gets. Fewer tools per call is what keeps small models on track."""

from langchain_core.tools import tool

from . import alarms, project, web
from .base import ToolContext

TOOL_ROUTES = ("alarms", "web", "project")


def make_shared(ctx: ToolContext) -> list:
    @tool
    def ask_user(question: str) -> str:
        """Ask the user something you need to continue. The turn pauses until they answer."""
        return question  # never executed: the executor turns this call into an interrupt

    @tool
    def remember(fact: str) -> str:
        """Store a durable fact about the user in memory, in their language."""
        stored, skipped = ctx.memory.add_facts([fact], thread_id=ctx.thread_id, turn_id=ctx.turn_id, matter=ctx.matter)
        return f"Remembered: {stored[0]}" if stored else f"Already known: {skipped[0]}"

    return [ask_user, remember]


def tools_for(route: str, ctx: ToolContext) -> list:
    if route == "alarms":
        specific = alarms.make_tools(ctx)
    elif route == "web":
        specific = web.make_tools(ctx)
    elif route == "project":
        specific = project.make_tools(ctx)
    else:
        return []
    return specific + make_shared(ctx)
