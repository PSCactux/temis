"""The citation checker as a tool: the model points at a text, the code verifies the citations."""

from langchain_core.tools import tool

from .. import lawcheck
from .base import ToolContext, ToolError
from .project import safe_path


def make_tools(ctx: ToolContext, *, files: bool = False) -> list:
    """`check_citations` reads the user's message; with `files=True` (matter folder) a second tool reads a file.
    The message tool takes no arguments on purpose: a small model given a `path` parameter invents one."""
    settings = ctx.settings

    def run(text: str) -> str:
        if not text.strip():
            raise ToolError("Nothing to check: the text is empty.")
        if settings.boe_archive_dir is None:
            raise ToolError("BOE_ARCHIVE_DIR is not configured: point it at the data folder of the BOE archive.")
        try:
            source = lawcheck.BoeArchive(settings.boe_archive_dir)
            aliases = lawcheck.Aliases.load(settings.laws_path)
        except (OSError, ValueError) as exc:
            raise ToolError(str(exc)) from exc
        _, report = lawcheck.check_text(text, source, aliases)
        return report

    @tool
    def check_citations() -> str:
        """Verify every legal citation in the user's current message (attachments included) against the firm's
        archive of the BOE: whether each norm and article exists and its current text. Takes no arguments.
        ALWAYS use this instead of judging citations or quoting articles yourself; report its verdicts as they are."""
        return run(ctx.message)

    @tool
    def check_citations_in_file(path: str) -> str:
        """Verify every legal citation in a file of the matter folder against the firm's archive of the BOE."""
        target = safe_path(ctx.project_root, path)
        if not target.is_file():
            raise ToolError(f"{path} does not exist.")
        return run(target.read_text(encoding="utf-8", errors="replace"))

    return [check_citations, check_citations_in_file] if files else [check_citations]
