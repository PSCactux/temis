"""The deadline tool: the model asks, the code counts (see `aroa.deadlines`)."""

from datetime import date

from langchain_core.tools import tool

from .. import deadlines
from .base import ToolContext, ToolError


def make_tools(ctx: ToolContext) -> list:
    settings = ctx.settings

    @tool
    def compute_deadline(
        notified_on: str,
        amount: int,
        unit: str = "días",
        jurisdiction: str = "civil",
        day_kind: str = "hábiles",
        urgent: bool = False,
    ) -> str:
        """Compute when a procedural term ends. ALWAYS use this instead of counting days yourself.
        notified_on: date of notification, YYYY-MM-DD. amount + unit: the term ("20", "días"; "1", "meses"; "1", "años").
        jurisdiction: civil | contencioso | social | penal | administrativo. day_kind: "hábiles" (the default and the
        rule for procedural terms) or "naturales" (only when the user or the law says "días naturales").
        urgent: true only when the law declares the proceeding urgent (August and Christmas then count).
        Returns the due date, the filing grace limit, every day excluded and the rules applied."""
        try:
            start = date.fromisoformat(notified_on)
        except ValueError as exc:
            raise ToolError(f"notified_on must be YYYY-MM-DD, got {notified_on!r}") from exc
        kind = day_kind.strip().lower()
        if kind not in ("hábiles", "habiles", "naturales"):
            raise ToolError('day_kind must be "hábiles" or "naturales"')
        try:
            calendar = deadlines.Calendar.load(settings.calendar_path, local=settings.local_holidays_list())
            result = deadlines.compute(
                start,
                amount,
                unit,
                jurisdiction,
                calendar=calendar,
                natural=kind == "naturales",
                region=settings.court_region,
                urgent=urgent,
            )
        except (ValueError, OSError) as exc:
            raise ToolError(str(exc)) from exc
        return result.as_text()

    return [compute_deadline]
