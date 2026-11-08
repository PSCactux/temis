"""Command line: chat, ask, ingest, memory, alarms, eval, report, metrics, serve."""

import argparse
import json
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path

from rich.console import Console
from rich.markdown import Markdown
from rich.table import Table

from .config import Settings, get_settings

console = Console()


def _assistant(settings: Settings | None = None):
    from .app import Assistant

    return Assistant(settings or get_settings())


def print_trace(trace: list[dict]) -> None:
    for step in trace:
        seconds = f" ({step['seconds']}s)" if step.get("seconds") else ""
        console.print(f"[dim]· {step['title']}{seconds}: {step['summary']}[/dim]")
        for item in step.get("items", []):
            console.print(f"[dim]    - {item}[/dim]")


def cmd_chat(args) -> None:
    assistant = _assistant()
    thread = args.thread or uuid.uuid4().hex[:8]
    console.print(
        f"[bold]{assistant.settings.assistant_name}[/bold] · thread {thread}"
        + (f" · matter {args.matter}" if args.matter else "")
    )
    console.print("[dim]Ctrl-D or /quit to leave. /path shows how the last reply was built.[/dim]")
    last = None
    while True:
        try:
            text = console.input("[bold cyan]you>[/bold cyan] ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not text:
            continue
        if text == "/quit":
            break
        if text == "/path":
            print_trace(last.trace if last else [])
            continue
        last = _run(assistant, thread, text, matter=args.matter, verbose=args.verbose)


def _run(assistant, thread: str, text: str, *, matter: str, verbose: bool):
    result = None
    for kind, item in assistant.stream(thread, text, matter=matter):
        if kind == "node" and verbose:
            console.print(f"[dim]· {item['title']}: {item['summary']}[/dim]")
        elif kind == "done":
            result = item
    label = "[bold yellow]question>[/bold yellow]" if result.kind == "question" else "[bold green]aroa>[/bold green]"
    console.print(label)
    console.print(Markdown(result.reply))
    m = result.metrics
    console.print(
        f"[dim]{m['route']} · {m['seconds']}s · {m['llm_calls']} model calls"
        f" · {m['input_tokens']}+{m['output_tokens']} tokens · {m['tool_calls']} tools"
        f" · claims {m['claims'] - m['claims_failed']}/{m['claims']} · turn {result.turn_id}[/dim]"
    )
    return result


def cmd_ask(args) -> None:
    assistant = _assistant()
    thread = args.thread or uuid.uuid4().hex[:8]
    result = _run(assistant, thread, args.text, matter=args.matter, verbose=args.verbose)
    if args.verbose:
        print_trace(result.trace)


def cmd_ingest(args) -> None:
    assistant = _assistant()
    console.print("Indexing documents" + (" (pseudonymising first)" if assistant.anonymizer else "") + "...")
    result = assistant.ingest()
    console.print(
        f"Added {len(result['added'])}, removed {len(result['removed'])}, {result['total']} documents indexed."
    )
    for name in result["added"]:
        console.print(f"  + {name}")


def cmd_memory(args) -> None:
    assistant = _assistant()
    if args.action == "list":
        table = Table("date", "matter", "fact")
        for fact in assistant.memory.list_facts():
            table.add_row(fact["ts"][:10], fact.get("matter") or "-", assistant.reveal(fact["text"]))
        console.print(table)
        console.print(assistant.memory.stats())
    elif args.action == "forget":
        removed = assistant.memory.forget(args.text)
        console.print(f"Removed {removed} fact(s).")
    elif args.action == "wipe":
        assistant.memory.wipe()
        console.print("Facts and episodes wiped. Documents kept.")


def cmd_alarms(args) -> None:
    from .tools import alarms

    assistant = _assistant()
    table = Table("id", "due", "thread", "text")
    for alarm in alarms.pending(assistant.conn):
        table.add_row(str(alarm["id"]), alarm["due_ts"], alarm["thread_id"], assistant.reveal(alarm["text"]))
    console.print(table)


def cmd_report(args) -> None:
    from .telemetry.db import connect
    from .telemetry.metrics import report

    data = report(connect(get_settings().db_path), days=args.days)
    console.print(json.dumps(data, indent=2, ensure_ascii=False))


def cmd_metrics(args) -> None:
    from .telemetry.db import connect
    from .telemetry.metrics import prometheus_text

    sys.stdout.write(prometheus_text(connect(get_settings().db_path)))


def cmd_eval(args) -> None:
    from .evaluation import run_evals

    base = get_settings()
    settings = base.model_copy(update={"data_dir": base.data_dir / "eval", "docs_dir": base.data_dir / "eval-docs"})
    results = run_evals(settings, Path(args.dataset), only=args.only, on_case=lambda r: console.print(_case_line(r)))
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    out = Path("evals/results") / f"{stamp}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    table = Table("kind", "passed", "cases", "median s")
    for kind, row in results["by_kind"].items():
        table.add_row(kind, str(row["passed"]), str(row["cases"]), str(row["median_seconds"]))
    console.print(table)
    console.print(
        f"Total {results['passed']}/{results['cases']} ({results['pass_rate']:.0%}) · model {results['model']} · {out}"
    )


def _case_line(row: dict) -> str:
    mark = "[green]PASS[/green]" if row["passed"] else "[red]FAIL[/red]"
    failed = ", ".join(k for k, v in row["checks"].items() if not v)
    return f"{mark} {row['id']} ({row['seconds']}s)" + (f" [red]{failed}[/red]" if failed else "")


def cmd_citas(args) -> None:
    """Check the legal citations of a file against the BOE archive, without a model."""
    from .lawcheck import Aliases, BoeArchive, check_text

    settings = get_settings()
    if settings.boe_archive_dir is None:
        sys.exit("BOE_ARCHIVE_DIR is not set (the data folder of the BOE archive)")
    text = Path(args.file).read_text(encoding="utf-8", errors="replace")
    _, report = check_text(text, BoeArchive(settings.boe_archive_dir), Aliases.load(settings.laws_path))
    console.print(report)


def cmd_serve(args) -> None:
    import uvicorn

    uvicorn.run("aroa.server:app", host=args.host, port=args.port, log_level="info")


def main() -> None:
    parser = argparse.ArgumentParser(prog="aroa", description="A law firm's assistant on your own machine.")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("chat", help="interactive conversation")
    p.add_argument("--thread", help="continue a thread")
    p.add_argument("--matter", default="", help="matter the conversation belongs to (memory wall)")
    p.add_argument("-v", "--verbose", action="store_true")
    p.set_defaults(func=cmd_chat)

    p = sub.add_parser("ask", help="one message")
    p.add_argument("text")
    p.add_argument("--thread")
    p.add_argument("--matter", default="")
    p.add_argument("-v", "--verbose", action="store_true", help="print the path of the turn")
    p.set_defaults(func=cmd_ask)

    sub.add_parser("ingest", help="index the documents folder").set_defaults(func=cmd_ingest)

    p = sub.add_parser("memory", help="list, forget or wipe facts")
    p.add_argument("action", choices=["list", "forget", "wipe"])
    p.add_argument("text", nargs="?", default="")
    p.set_defaults(func=cmd_memory)

    sub.add_parser("alarms", help="pending alarms").set_defaults(func=cmd_alarms)

    p = sub.add_parser("report", help="aggregated metrics")
    p.add_argument("--days", type=int, default=1)
    p.set_defaults(func=cmd_report)

    sub.add_parser("metrics", help="Prometheus text").set_defaults(func=cmd_metrics)

    p = sub.add_parser("eval", help="run the evaluation dataset against the configured models")
    p.add_argument("--dataset", default="evals/dataset.jsonl")
    p.add_argument("--only", help="run only the cases of this kind or with this id prefix")
    p.set_defaults(func=cmd_eval)

    p = sub.add_parser("citas", help="check the legal citations of a file against the BOE archive")

    p.add_argument("file")

    p.set_defaults(func=cmd_citas)

    p = sub.add_parser("serve", help="web app and API")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.set_defaults(func=cmd_serve)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
