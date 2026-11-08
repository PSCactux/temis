"""Procedural deadlines: the arithmetic the law prescribes, with every excluded day accounted for."""

from datetime import date, datetime
from pathlib import Path

import pytest

from aroa.deadlines import Calendar, compute

CAL = Calendar.load(Path(__file__).parent.parent / "data/calendar/holidays.json")


def test_civil_days_skip_weekends_and_august():
    # Notified Friday 2026-07-24; 20 working days: five in July, August does not count, fifteen in September.
    result = compute(date(2026, 7, 24), 20, "días", "civil", calendar=CAL)
    assert result.due == date(2026, 9, 21)
    assert result.grace_until == datetime(2026, 9, 22, 15, 0)
    reasons = {why for _, why in result.excluded}
    assert "agosto inhábil" in reasons and "sábado" in reasons and "domingo" in reasons
    assert sum(1 for d, why in result.excluded if why == "agosto inhábil") == 21  # the working days of August 2026


def test_administrative_days_count_august_but_not_the_national_holiday():
    # Notified Friday 2026-08-07; 10 working days under Ley 39/2015: August counts, weekends do not.
    result = compute(date(2026, 8, 7), 10, "días", "administrativo", calendar=CAL)
    assert result.due == date(2026, 8, 21)
    assert result.grace_until is None  # no art. 135.5 LEC grace in administrative procedure


def test_month_term_without_equivalent_day_ends_on_the_last_day_then_next_working_day():
    # Notified 2026-01-30; one month: February has no 30th, so the 28th, a Saturday, so Monday 2 March.
    result = compute(date(2026, 1, 30), 1, "meses", "civil", calendar=CAL)
    assert result.due == date(2026, 3, 2)
    assert any("no tiene día 30" in n for n in result.notes)
    assert [why for _, why in result.excluded] == ["sábado", "domingo"]


def test_term_ending_on_a_holiday_moves_to_the_next_working_day():
    # 2026-05-01 is a national holiday (Friday): a month from 2026-04-01 ends on Monday 4 May.
    result = compute(date(2026, 4, 1), 1, "meses", "civil", calendar=CAL)
    assert result.due == date(2026, 5, 4)
    assert result.excluded[0] == (date(2026, 5, 1), "festivo nacional")


def test_christmas_break_is_skipped_in_court_terms_but_not_in_administrative_ones():
    civil = compute(date(2026, 12, 21), 5, "días", "civil", calendar=CAL)
    assert civil.due == date(2027, 1, 11)  # 22, 23 Dec; 24 Dec to 6 Jan excluded; 7, 8 Jan (Thu, Fri); Monday 11 Jan
    admin = compute(date(2026, 12, 21), 5, "días", "administrativo", calendar=CAL)
    assert admin.due == date(2026, 12, 29)  # 22, 23, 24, 28, 29 (25 Dec holiday, weekend)


def test_natural_days_count_everything_but_a_non_working_end_moves():
    # 15 natural days from Monday 2026-06-15 ends Tuesday 30 June: no move.
    assert compute(date(2026, 6, 15), 15, "días", "civil", calendar=CAL, natural=True).due == date(2026, 6, 30)
    # 5 natural days from Monday 2026-06-15 ends Saturday 20 June: moves to Monday 22.
    moved = compute(date(2026, 6, 15), 5, "días", "civil", calendar=CAL, natural=True)
    assert moved.due == date(2026, 6, 22) and moved.excluded == [
        (date(2026, 6, 20), "sábado"),
        (date(2026, 6, 21), "domingo"),
    ]


def test_regional_and_local_holidays_come_from_the_calendar():
    # 2026-09-11 (Diada) is a Friday: counts in Madrid, not in Catalonia.
    madrid = compute(date(2026, 9, 9), 2, "días", "civil", calendar=CAL, region="MD")
    catalonia = compute(date(2026, 9, 9), 2, "días", "civil", calendar=CAL, region="CT")
    assert madrid.due == date(2026, 9, 11) and catalonia.due == date(2026, 9, 14)
    local = Calendar.load(Path(__file__).parent.parent / "data/calendar/holidays.json", local=["2026-09-10"])
    assert compute(date(2026, 9, 9), 2, "días", "civil", calendar=local, region="CT").due == date(2026, 9, 15)


def test_urgent_proceedings_count_august():
    result = compute(date(2026, 7, 24), 20, "días", "social", calendar=CAL, urgent=True)
    assert result.due == date(2026, 8, 21)
    assert any("urgente" in n for n in result.notes)


def test_missing_calendar_year_is_flagged_not_silently_ignored():
    result = compute(date(2031, 3, 3), 3, "días", "civil", calendar=CAL)
    assert any("sin calendario de festivos para 2031" in n for n in result.notes)


@pytest.mark.parametrize("bad", [("x", 5, "días"), ("civil", 0, "días"), ("civil", 5, "semanas")])
def test_bad_inputs_are_refused(bad):
    jurisdiction, amount, unit = bad
    with pytest.raises(ValueError):
        compute(date(2026, 1, 12), amount, unit, jurisdiction, calendar=CAL)


def test_text_report_lists_due_date_rules_and_exclusions():
    text = compute(date(2026, 7, 24), 20, "días", "civil", calendar=CAL).as_text()
    assert "VENCE: 2026-09-21 (lunes)" in text and "15:00 del 2026-09-22" in text
    assert "art. 133 LEC" in text and "agosto inhábil" in text and "COURT_REGION" in text


def test_the_model_asks_the_tool_and_the_claim_is_verified(make_assistant):
    from aroa.schemas import Claim, Claims, Facts
    from tests.test_graph import routing, tool_call

    assistant = make_assistant(
        [
            routing("alarms"),
            tool_call(
                "compute_deadline", {"notified_on": "2026-07-24", "amount": 20, "unit": "días", "jurisdiction": "civil"}
            ),
            "Vence el 21 de septiembre de 2026; puedes presentar hasta las 15:00 del 22.",
            Claims(claims=[Claim(kind="other", target="plazo calculado")]),
            Facts(facts=[]),
        ]
    )
    result = assistant.ask(
        "t1", "me notificaron la sentencia el 24 de julio de 2026, ¿hasta cuándo puedo apelar? 20 días"
    )
    assert result.kind == "alarms" and result.extras.get("verified") is not False
    tool_output = assistant.model.calls[2][-1].content  # what the model saw from the tool
    assert "VENCE: 2026-09-21" in tool_output and "agosto inhábil" in tool_output


def test_a_date_and_a_term_force_the_tool_route_even_when_the_router_says_chat(make_assistant):
    from aroa.graph import deadline_in
    from aroa.schemas import Claim, Claims, Facts
    from tests.test_graph import routing, tool_call

    assert deadline_in("Auto notificado el 30 de enero de 2026 con plazo de un mes para recurrir. ¿Último día?")
    assert not deadline_in("¿cuál es el plazo para apelar según la guía?")  # no concrete date: a documents question
    assistant = make_assistant(
        [
            routing("chat"),
            tool_call("compute_deadline", {"notified_on": "2026-01-30", "amount": 1, "unit": "meses"}),
            "Vence el 2 de marzo de 2026.",
            Claims(claims=[Claim(kind="other", target="plazo")]),
            Facts(facts=[]),
        ]
    )
    result = assistant.ask(
        "t1", "Auto notificado el 30 de enero de 2026 con plazo de un mes para recurrir. ¿Último día?"
    )
    assert result.kind == "alarms" and "VENCE: 2026-03-02" in assistant.model.calls[2][-1].content
