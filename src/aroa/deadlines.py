"""Procedural deadlines computed in code, never by the model.

The rules are the Spanish ones: which days do not count (weekends, holidays, August, the Christmas
break), how days, months and years are counted, what happens when the term ends on a non-working day,
and the grace period for filings. Every result lists the days it excluded and the rule it applied, so a
lawyer can check the arithmetic and the sources; the firm's own calendar (regional and local holidays)
is data, not code. Nothing here replaces the lawyer's judgement about which rule governs a case."""

from __future__ import annotations

import calendar as cal
import json
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from pathlib import Path

JURISDICTIONS = {
    "civil": {
        "name": "civil (LEC)",
        "august": True,
        "christmas": True,
        "grace": True,
        "rules": [
            "art. 130 LEC: inhábiles sábados, domingos, 24 y 31 de diciembre, festivos nacionales, autonómicos y locales",
            "art. 183 LOPJ: inhábil el mes de agosto, y del 24 de diciembre al 6 de enero, salvo actuaciones urgentes",
            "art. 133 LEC: los plazos de días se cuentan desde el día siguiente a la notificación, excluidos los inhábiles; "
            "los de meses o años, de fecha a fecha; si el último día es inhábil, se prorroga al siguiente hábil",
            "art. 135.5 LEC: los escritos sujetos a plazo pueden presentarse hasta las 15:00 del día hábil siguiente al vencimiento",
        ],
    },
    "contencioso": {
        "name": "contencioso-administrativo (LJCA)",
        "august": True,
        "christmas": True,
        "grace": True,
        "rules": [
            "art. 128 LJCA: plazos improrrogables; agosto inhábil salvo el procedimiento de protección de derechos fundamentales",
            "art. 183 LOPJ y art. 130 LEC (supletoria): días inhábiles",
            "art. 135.5 LEC (supletoria): presentación hasta las 15:00 del día hábil siguiente",
        ],
    },
    "social": {
        "name": "social (LRJS)",
        "august": True,
        "christmas": True,
        "grace": True,
        "rules": [
            "art. 43 LRJS: agosto inhábil salvo las modalidades que la ley declara urgentes (despido, extinción, movilidad, "
            "modificación sustancial, vacaciones, conciliación, derechos fundamentales, entre otras)",
            "art. 183 LOPJ y art. 130 LEC (supletoria): días inhábiles",
            "art. 135.5 LEC (supletoria): presentación hasta las 15:00 del día hábil siguiente",
        ],
    },
    "penal": {
        "name": "penal (LECrim)",
        "august": False,
        "christmas": False,
        "grace": True,
        "rules": [
            "art. 201 LECrim: para la instrucción todos los días y horas son hábiles; en las demás fases rigen los días inhábiles "
            "del art. 183 LOPJ. Confirmar la fase antes de fiarse de este cómputo",
            "art. 135.5 LEC (supletoria): presentación hasta las 15:00 del día hábil siguiente",
        ],
    },
    "administrativo": {
        "name": "administrativo (Ley 39/2015)",
        "august": False,
        "christmas": False,
        "grace": False,
        "rules": [
            "art. 30 Ley 39/2015: inhábiles sábados, domingos y festivos; agosto es hábil; los plazos de días se cuentan desde el "
            "día siguiente a la notificación; los de meses, de fecha a fecha, y si no hay día equivalente, el último del mes; "
            "si el último día es inhábil, se entiende prorrogado al primer día hábil siguiente",
        ],
    },
}

UNITS = {
    "días": "days",
    "dias": "days",
    "days": "days",
    "meses": "months",
    "months": "months",
    "años": "years",
    "anos": "years",
    "years": "years",
}


@dataclass
class Calendar:
    """Holidays by year: national plus the firm's region and locality. Loaded from JSON, never guessed."""

    national: dict[int, set[date]] = field(default_factory=dict)
    regions: dict[str, dict[int, set[date]]] = field(default_factory=dict)
    region_names: dict[str, str] = field(default_factory=dict)
    local: set[date] = field(default_factory=set)
    source: str = ""

    @classmethod
    def load(cls, path: Path, *, local: list[str] = ()) -> Calendar:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        national = {int(y): {date.fromisoformat(d) for d in days} for y, days in data.get("national", {}).items()}
        regions, names = {}, {}
        for code, body in data.get("regions", {}).items():
            names[code] = body.get("name", code)
            regions[code] = {int(y): {date.fromisoformat(d) for d in days} for y, days in body.items() if y != "name"}
        return cls(national, regions, names, {date.fromisoformat(d) for d in local}, data.get("source", ""))

    def years(self) -> set[int]:
        return set(self.national)

    def holiday(self, day: date, region: str = "") -> str | None:
        if day in self.national.get(day.year, ()):
            return "festivo nacional"
        if region and day in self.regions.get(region, {}).get(day.year, ()):
            return f"festivo autonómico ({self.region_names.get(region, region)})"
        if day in self.local:
            return "festivo local"
        return None


@dataclass
class Deadline:
    start: date
    due: date
    grace_until: datetime | None
    counted: int
    unit: str
    jurisdiction: str
    natural: bool
    excluded: list[tuple[date, str]]
    rules: list[str]
    notes: list[str]

    def as_text(self) -> str:
        lines = [
            f"Notificación: {self.start.isoformat()} ({_weekday(self.start)})",
            f"Plazo: {self.counted} {self.unit}{' naturales' if self.natural else ' hábiles' if self.unit == 'días' else ''}"
            f", jurisdicción {JURISDICTIONS[self.jurisdiction]['name']}",
            f"VENCE: {self.due.isoformat()} ({_weekday(self.due)})",
        ]
        if self.grace_until:
            lines.append(
                f"Presentación posible hasta las 15:00 del {self.grace_until.date().isoformat()} (art. 135.5 LEC)"
            )
        if self.excluded:
            shown = [f"{d.isoformat()} {why}" for d, why in self.excluded[:12]]
            more = f" y {len(self.excluded) - 12} más" if len(self.excluded) > 12 else ""
            lines.append("Días no computados: " + "; ".join(shown) + more)
        lines.append("Reglas aplicadas: " + " | ".join(self.rules))
        for note in self.notes:
            lines.append("Aviso: " + note)
        return "\n".join(lines)


def why_inhabil(day: date, jurisdiction: str, calendar: Calendar, region: str, *, urgent: bool) -> str | None:
    """The reason a day does not count, or None when it does."""
    spec = JURISDICTIONS[jurisdiction]
    if day.weekday() == 5:
        return "sábado"
    if day.weekday() == 6:
        return "domingo"
    holiday = calendar.holiday(day, region)
    if holiday:
        return holiday
    if not urgent:
        if spec["august"] and day.month == 8:
            return "agosto inhábil"
        if spec["christmas"] and ((day.month == 12 and day.day >= 24) or (day.month == 1 and day.day <= 6)):
            return "periodo navideño inhábil (24 dic a 6 ene)"
    if jurisdiction == "civil" and day.month == 12 and day.day == 31:
        return "31 de diciembre inhábil (art. 130 LEC)"
    return None


def compute(
    start: date,
    amount: int,
    unit: str,
    jurisdiction: str = "civil",
    *,
    calendar: Calendar,
    natural: bool = False,
    region: str = "",
    urgent: bool = False,
) -> Deadline:
    """The day a term ends, counted the way the law says, with every excluded day and the rules applied."""
    if jurisdiction not in JURISDICTIONS:
        raise ValueError(f"jurisdicción desconocida: {jurisdiction}; usa {', '.join(JURISDICTIONS)}")
    kind = UNITS.get(unit.lower())
    if kind is None:
        raise ValueError(f"unidad desconocida: {unit}; usa días, meses o años")
    if amount <= 0:
        raise ValueError("el plazo debe ser un número positivo")
    spec = JURISDICTIONS[jurisdiction]
    excluded: list[tuple[date, str]] = []
    notes: list[str] = []

    def blocked(day: date) -> str | None:
        return why_inhabil(day, jurisdiction, calendar, region, urgent=urgent)

    if kind == "days":
        day, counted = start, 0
        while counted < amount:
            day += timedelta(days=1)
            reason = None if natural else blocked(day)
            if reason:
                excluded.append((day, reason))
            else:
                counted += 1
        due = day
    else:
        months = amount * (12 if kind == "years" else 1)
        year, month = divmod(start.month - 1 + months, 12)
        year, month = start.year + year, month + 1
        last = cal.monthrange(year, month)[1]
        due = date(year, month, min(start.day, last))
        if start.day > last:
            notes.append(
                f"el mes de vencimiento no tiene día {start.day}: vence el último día del mes (art. 133.3 LEC / art. 30.4 Ley 39/2015)"
            )
    # A term that ends on a non-working day runs to the next working day (art. 133.4 LEC, art. 30.5 Ley 39/2015).
    while (reason := blocked(due)) is not None:
        excluded.append((due, reason))
        due += timedelta(days=1)

    grace = None
    if spec["grace"]:
        nxt = due + timedelta(days=1)
        while blocked(nxt):
            nxt += timedelta(days=1)
        grace = datetime.combine(nxt, time(15, 0))

    years_needed = {start.year, due.year}
    missing = sorted(y for y in years_needed if y not in calendar.years())
    if missing:
        notes.append(
            f"sin calendario de festivos para {', '.join(map(str, missing))}: solo se han excluido fines de semana y periodos legales"
        )
    if not region:
        notes.append(
            "sin festivos autonómicos ni locales configurados (COURT_REGION, LOCAL_HOLIDAYS): compruébalos para el juzgado concreto"
        )
    if urgent:
        notes.append("cómputo como actuación urgente: agosto y el periodo navideño se han contado como hábiles")
    if jurisdiction == "social" and not urgent:
        notes.append(
            "si la modalidad procesal es de las urgentes del art. 43.4 LRJS (despido, etc.), repite el cálculo como urgente"
        )
    return Deadline(
        start,
        due,
        grace,
        amount,
        "días" if kind == "days" else ("meses" if kind == "months" else "años"),
        jurisdiction,
        natural,
        excluded,
        list(spec["rules"]),
        notes,
    )


def _weekday(day: date) -> str:
    return ("lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo")[day.weekday()]
