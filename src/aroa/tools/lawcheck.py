"""The citation checker as a tool: the model points at a text, the code verifies the citations."""

from langchain_core.tools import tool

from .. import lawcheck
from .base import ToolContext, ToolError
from .project import safe_path


def make_tools(ctx: ToolContext) -> list:
    settings = ctx.settings

    @tool
    def check_citations(path: str = "") -> str:
        """Verify every legal citation (artículo X de la Ley Y, LEC, Código Civil, STS...) in a text against the
        firm's archive of the BOE: whether the norm exists, whether the article exists, and its current text.
        Without `path`, checks the user's current message (attachments included); with `path`, a file of the
        matter folder. ALWAYS use this instead of judging citations yourself; report its verdicts as they are."""
        if path:
            target = safe_path(ctx.project_root, path)
            if not target.is_file():
                raise ToolError(f"{path} does not exist.")
            text = target.read_text(encoding="utf-8", errors="replace")
        else:
            text = ctx.message
        if not text.strip():
            raise ToolError("Nothing to check: no text in the message and no path given.")
        if settings.boe_archive_dir is None:
            raise ToolError("BOE_ARCHIVE_DIR is not configured: point it at the data folder of the BOE archive.")
        try:
            source = lawcheck.BoeArchive(settings.boe_archive_dir)
            aliases = lawcheck.Aliases.load(settings.laws_path)
        except (OSError, ValueError) as exc:
            raise ToolError(str(exc)) from exc
        _, report = lawcheck.check_text(text, source, aliases)
        return report

    return [check_citations]
