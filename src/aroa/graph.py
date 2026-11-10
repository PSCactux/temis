"""The assistant as a LangGraph state machine.

    recall            facts and past episodes relevant to the message (embeddings only)
    route             one structured call: route, standalone query, project
    chat              reply from persona + memories + history
    retrieve/grade/answer_documents/check_citations   corrective RAG over the user's documents
    act ⇄ tools       tool loop bound to the route's tools; ask_user pauses the thread (interrupt)
    verify            the reply's claims about actions, checked in code against the turn's evidence
    memorize          durable facts extracted from the turn, deduplicated, plus the episode

The model decides what to say and which tool to call. The code decides retries, limits,
when a loop is a loop and what counts as proven.
"""

import json
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Annotated, Literal, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, RemoveMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import Checkpointer, interrupt

from . import prompts
from .citations import check_answer, citation_problems, cite_memory, cited_numbers, tidy_markers, unverified_numbers
from .config import Settings
from .i18n import t
from .memory.chunking import Chunk
from .memory.store import MemoryStore
from .privacy import Anonymizer
from .schemas import Claims, DocumentAnswer, Facts, Grading, PolicyReview, Routing
from .telemetry.audit import log_memory_write, log_tool_call
from .tools.base import ToolContext, ToolError
from .tools.project import agents_md
from .tools.registry import TOOL_ROUTES, tools_for
from .tools.web import fetch_text
from .trace import Steps
from .verify import check_claims, unread_urls

MAX_DOCUMENT_ANSWERS = 2
REPEAT_WARN = 2  # the same tool call with the same arguments is answered from memory from here on
REPEAT_STOP = 5  # and from here on the loop is forced to conclude
MAX_VERIFY_RETRIES = 1


class AgentState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    # per-turn fields, reset in `recall`
    route: str
    query: str
    project: str
    memories: dict
    doc_hints: list[str]
    retrieved: list[dict]
    sources: list[dict]
    answer: dict | None
    citation_errors: list[str]
    generations: int
    evidence: list[dict]
    tool_turns: int
    repeats: dict[str, int]
    verify_feedback: list[str]
    verify_attempts: int
    verification: dict
    facts_written: list[str]
    trace: list[dict]


def _chunks(items: list[dict]) -> list[Chunk]:
    return [Chunk(**item) for item in items]


ATTACHMENT = re.compile(r'\n\n<attachment name="([^"\n]+)">\n(.*?)\n</attachment>', re.S)
ATTACHMENT_PREFIX = "adjunto:"
MAX_ATTACHMENT_CHUNKS = 12


def strip_attachments(text: str) -> str:
    return ATTACHMENT.sub("", text).strip()


def attachment_chunks(text: str, max_chars: int, query: str = "") -> list[Chunk]:
    """The files attached to the message as citable fragments, the ones closest to the question first.
    They are never indexed: they exist for this turn, like the message itself."""
    from .memory.chunking import chunk_document

    chunks: list[Chunk] = []
    for name, body in ATTACHMENT.findall(text):
        chunks += chunk_document(body, title=name, source=ATTACHMENT_PREFIX + name, max_chars=max_chars)
    if len(chunks) <= MAX_ATTACHMENT_CHUNKS:
        return chunks
    words = {w for w in re.findall(r"\w+", query.lower()) if len(w) > 3}
    ranked = sorted(chunks, key=lambda c: -len(words & set(re.findall(r"\w+", c.text.lower()))))
    kept = ranked[:MAX_ATTACHMENT_CHUNKS]
    return [c for c in chunks if c in kept]  # reading order kept


def is_attachment(chunk: Chunk) -> bool:
    return chunk.source.startswith(ATTACHMENT_PREFIX)


def _timed(node):
    def run(state: AgentState, config: RunnableConfig) -> dict:
        started = time.perf_counter()
        update = node(state, config)
        if update.get("trace") and update["trace"] is not state.get("trace"):
            update["trace"][-1]["seconds"] = round(time.perf_counter() - started, 2)
        return update

    run.__name__ = node.__name__
    return run


def window(messages: list[AnyMessage], n: int) -> list[AnyMessage]:
    """The last n messages, extended backwards so the window starts at a user message (tool pairs stay whole)."""
    start = max(0, len(messages) - n)
    while start > 0 and not isinstance(messages[start], HumanMessage):
        start -= 1
    return messages[start:]


def turn_messages(messages: list[AnyMessage]) -> list[AnyMessage]:
    """Messages since the last user message: this turn's tool calls and results."""
    for i in range(len(messages) - 1, -1, -1):
        if isinstance(messages[i], HumanMessage):
            return messages[i:]
    return messages


def load_persona(folder: Path, name: str) -> str:
    """The persona files joined, with {name} replaced by the assistant's name."""
    parts = [p.read_text(encoding="utf-8") for p in sorted(folder.glob("*.md"))] if folder.is_dir() else []
    return "\n\n".join(parts).replace("{name}", name) or f"# Who you are\n\nYou are {name}, a personal assistant."


def build_graph(
    llm: BaseChatModel,
    fast_llm: BaseChatModel,
    memory: MemoryStore,
    settings: Settings,
    conn,
    *,
    persona: str,
    anonymizer: Anonymizer | None = None,
    checkpointer: Checkpointer = None,
):
    router = fast_llm.with_structured_output(Routing, method="json_schema")
    grader = fast_llm.with_structured_output(Grading, method="json_schema")
    writer = llm.with_structured_output(DocumentAnswer, method="json_schema")
    claimer = fast_llm.with_structured_output(Claims, method="json_schema")
    reviewer = fast_llm.with_structured_output(PolicyReview, method="json_schema")
    extractor = fast_llm.with_structured_output(Facts, method="json_schema")
    lang = settings.assistant_language
    steps = Steps(lang)

    def ctx_of(state: AgentState, config: RunnableConfig) -> ToolContext:
        conf = config.get("configurable", {})
        return ToolContext(
            settings=settings,
            conn=conn,
            thread_id=conf.get("thread_id", "default"),
            turn_id=conf.get("turn_id", "no-turn"),
            project=state.get("project", ""),
            message=next((m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), ""),
            memory=memory,
            anonymizer=anonymizer,
            matter=conf.get("matter", ""),
        )

    def projects() -> list[str]:
        folder = settings.projects_dir
        return sorted(p.name for p in folder.iterdir() if p.is_dir()) if folder.is_dir() else []

    def system_prompt(state: AgentState, *, tools_note: str | None = None) -> SystemMessage:
        text = prompts.PERSONA_FRAME.format(
            persona=persona, today=datetime.now().astimezone().strftime("%A %Y-%m-%d %H:%M")
        )
        text += "\n\n" + prompts.format_memories(state.get("memories") or {})
        if tools_note is not None:
            text += "\n\n" + prompts.TOOLS_FRAME.format(extra=tools_note)
        if state.get("route") == "project" and state.get("project"):
            root = settings.projects_dir / state["project"]
            text += "\n\n" + prompts.PROJECT_FRAME.format(project=state["project"], agents_md=agents_md(root))
        if state.get("verify_feedback"):
            text += "\n\n" + prompts.VERIFY_FEEDBACK.format(
                failures="\n".join(f"- {f}" for f in state["verify_feedback"])
            )
        return SystemMessage(text)

    # ── recall & route ─────────────────────────────────────────────────
    def recall(state: AgentState, config: RunnableConfig) -> dict:
        conf = config.get("configurable", {})
        incoming = state["messages"][-1]
        question = incoming.content
        messages = []
        if anonymizer:
            question = anonymizer.anonymize(question)
            if question != incoming.content:
                # Same id: add_messages replaces the message, so the original text never reaches a model or a store.
                messages = [HumanMessage(question, id=incoming.id)]
        memories = memory.recall(question, exclude_thread=conf.get("thread_id"), matter=conf.get("matter", ""))
        selected = conf.get("documents") or None
        hints = (
            [f"{c.label()}: {c.text[:220]}" for c in memory.search_documents(question, k=3, sources=selected)]
            if memory.document_titles()
            else []
        )
        return {
            "messages": messages,
            "doc_hints": hints,
            "route": "",
            "query": question,
            "project": "",
            "memories": memories,
            "retrieved": [],
            "sources": [],
            "answer": None,
            "citation_errors": [],
            "generations": 0,
            "evidence": [],
            "tool_turns": 0,
            "repeats": {},
            "verify_feedback": [],
            "verify_attempts": 0,
            "verification": {},
            "facts_written": [],
            "trace": [steps.recalled(memories, anonymized=bool(messages))],
        }

    def route(state: AgentState, config: RunnableConfig) -> dict:
        history = window(state["messages"], settings.history_messages)
        system = prompts.ROUTE.format(
            titles=", ".join(memory.document_titles()) or "(none)",
            projects=", ".join(projects()) or "(none)",
            policy=prompts.POLICY,
        )
        # Document fragments and memory are untrusted: they travel in the user turn, tagged, never in
        # the system prompt that also decides the policy.
        facts = [f["text"] for f in state["memories"].get("facts", [])]
        context = HumanMessage(prompts.route_context(history[-1].content, state.get("doc_hints") or [], facts))
        result: Routing = router.invoke([SystemMessage(system), *history[:-1], context])
        if result.policy == "blocked":
            # Blocking is rare and costly when wrong (a lawyer refused for asking a question), so a block
            # is confirmed by a second, narrower call before it stands.
            review: PolicyReview = reviewer.invoke(
                [SystemMessage(prompts.POLICY_REVIEW.format(policy=prompts.POLICY)), state["messages"][-1]]
            )
            if review.verdict == "blocked":
                reason = review.reason.strip() or result.policy_reason.strip() or "it is outside what the firm allows."
                return {
                    "route": "blocked",
                    "trace": [*state["trace"], steps.blocked(reason)],
                    "messages": [_final(AIMessage(t(lang, "blocked_reply", reason=reason)), config, kind="blocked")],
                }
        chosen = result.route
        project = result.project.strip() if chosen == "project" else ""
        mentioned = project_in(state["messages"][-1].content, projects())
        if mentioned and (chosen == "project" or (chosen == "chat" and wants_files(state["messages"][-1].content))):
            chosen, project = "project", mentioned  # naming a matter folder plus file work is work in that folder
        elif chosen == "project" and project not in projects():
            chosen, project = "chat", ""  # an unknown project is answered in conversation, not with tools
        if chosen == "chat" and config.get("configurable", {}).get("documents"):
            chosen = "documents"  # the user chose documents to work on: a question goes to them first
        if chosen == "chat" and deadline_in(state["messages"][-1].content):
            chosen = "alarms"  # a term from a concrete date is arithmetic: the deadline tool, never the model's count
        text = state["messages"][-1].content
        if chosen in ("chat", "web") and settings.boe_archive_dir and cites_law(text) and not wants_web(text):
            chosen = "review"  # the law lives in the BOE archive: the checker reads it, the model does not recall it
        if chosen == "documents" and not memory.document_titles() and not ATTACHMENT.search(text):
            chosen = "chat"
        query = result.search_query.strip() or state["query"]
        return {
            "route": chosen,
            "query": query,
            "project": project,
            "trace": [*state["trace"], steps.routed(chosen, query, project)],
        }

    def after_route(state: AgentState) -> str:
        if state["route"] == "blocked":
            return END
        if state["route"] == "documents":
            return "retrieve"
        if state["route"] in TOOL_ROUTES:
            return "act"
        return "chat"

    # ── chat ───────────────────────────────────────────────────────────
    def with_memory(reply: AIMessage, state: AgentState) -> tuple[AIMessage, list[dict]]:
        """Markers like [m1] kept only when the remembered block exists; the cited items are listed under the reply
        like the sources of a document answer, so an answer from memory shows what it rests on."""
        text, sources = cite_memory(str(reply.content), state.get("memories") or {}, lang)
        if sources:
            lines = [text.strip(), "", f"**{t(lang, 'memory_sources')}**", ""]
            for s in sources:
                lines.append(f"- [{s['n']}] {s['label']}")
                lines.append(f"  > «{s['quote'][:300]}»")
            text = "\n".join(lines)
        return AIMessage(text, additional_kwargs=dict(reply.additional_kwargs), id=reply.id), sources

    def chat(state: AgentState, config: RunnableConfig) -> dict:
        history = window(state["messages"], settings.history_messages)
        reply = llm.invoke([system_prompt(state), *history])
        message, sources = with_memory(reply, state)
        return {
            "messages": [_final(message, config, kind="chat", sources=sources)],
            "trace": [*state["trace"], steps.chatted(len(sources))],
        }

    # ── documents ──────────────────────────────────────────────────────
    def retrieve(state: AgentState, config: RunnableConfig) -> dict:
        message = state["messages"][-1].content
        question = strip_attachments(message)
        selected = config.get("configurable", {}).get("documents") or None
        attached = attachment_chunks(message, settings.chunk_max_chars, state["query"] or question)
        chunks = memory.search_documents(state["query"], sources=selected) if memory.document_titles() else []
        if state["query"] != question and chunks:
            chunks = _fuse(chunks, memory.search_documents(question, sources=selected))
        chunks = attached + chunks  # what the user just attached comes first and is cited like any fragment
        return {
            "retrieved": [c.to_payload() for c in chunks],
            "trace": [*state["trace"], steps.retrieved(chunks, [state["query"], question])],
        }

    def grade(state: AgentState, config: RunnableConfig) -> dict:
        retrieved = _chunks(state["retrieved"])
        if not retrieved:
            return {"sources": [], "trace": [*state["trace"], steps.graded([], 0)]}
        result: Grading = grader.invoke(
            [
                SystemMessage(prompts.GRADE),
                HumanMessage(f"Question: {state['query']}\n\n{prompts.format_fragments(retrieved)}"),
            ]
        )
        keep = [retrieved[n - 1] for n in sorted(set(result.relevant)) if 1 <= n <= len(retrieved)]
        if keep and retrieved[0] not in keep:
            keep.insert(0, retrieved[0])  # the grader narrows the search, it does not overrule its best hit
        return {
            "sources": [c.to_payload() for c in keep],
            "trace": [*state["trace"], steps.graded(keep, len(retrieved))],
        }

    def after_grade(state: AgentState) -> Literal["answer_documents", "chat"]:
        return "answer_documents" if state["sources"] else "chat"  # nothing in the documents: answer from memory

    def answer_documents(state: AgentState, config: RunnableConfig) -> dict:
        sources = _chunks(state["sources"])
        question = state["messages"][-1].content  # the user's words; the standalone query only resolves references
        content = f"{prompts.format_fragments(sources)}\n\nQuestion: {question}\n(Standalone form: {state['query']})"
        if state["citation_errors"]:
            content += "\n\nYour previous answer failed the citation check:\n" + "\n".join(
                f"- {e}" for e in state["citation_errors"]
            )
        result: DocumentAnswer = writer.invoke([SystemMessage(prompts.ANSWER_DOCUMENTS), HumanMessage(content)])
        attempt = state["generations"] + 1
        return {
            "answer": result.model_dump(),
            "generations": attempt,
            "trace": [*state["trace"], steps.answered(result.coverage, len(cited_numbers(result.answer)), attempt)],
        }

    def check_citations(state: AgentState, config: RunnableConfig) -> dict:
        answer = DocumentAnswer(**state["answer"])
        errors = [] if answer.coverage == "none" else check_answer(answer, _chunks(state["sources"]), lang)
        return {"citation_errors": errors, "trace": [*state["trace"], steps.citations_checked(errors)]}

    def after_citations(state: AgentState) -> Literal["finalize_documents", "answer_documents"]:
        if state["citation_errors"] and state["generations"] < MAX_DOCUMENT_ANSWERS:
            return "answer_documents"
        return "finalize_documents"

    def finalize_documents(state: AgentState, config: RunnableConfig) -> dict:
        answer = DocumentAnswer(**state["answer"])
        sources = _chunks(state["sources"])
        if answer.coverage == "none":
            reviewed = "; ".join(sorted({c.title for c in sources}))
            text = f"{t(lang, 'abstain')}\n\n{t(lang, 'documents_reviewed')}: {reviewed}."
            return {"messages": [_final(AIMessage(text), config, kind="abstain", sources=[])]}
        quotes = {q.source: q.text for q in answer.quotes}
        bad = unverified_numbers(citation_problems(answer, sources)) if state["citation_errors"] else set()
        cited = []
        for n in sorted(cited_numbers(answer.answer)):
            if not 1 <= n <= len(sources):
                continue
            c = sources[n - 1]
            item = {"n": n, "label": c.label(), "file": c.source, "quote": quotes.get(n), "verified": n not in bad}
            if is_attachment(c):  # not in the library: the fragment itself travels with the answer
                where = "" if c.section in ("Start", c.title) else f" — {c.section}"
                item.update(
                    kind="attachment", file=None, label=f"{t(lang, 'attachment')} · {c.title}{where}", text=c.text
                )
            cited.append(item)
        lines = [tidy_markers(answer.answer.strip()), "", f"**{t(lang, 'sources')}**", ""]
        for s in cited:
            mark = "" if s["verified"] else f" — _{t(lang, 'unverified_mark')}_"
            where = f" (`{s['file']}`)" if s["file"] else ""
            lines.append(f"- [{s['n']}] {s['label']}{where}{mark}")
            if s["quote"]:
                lines.append(f"  > «{s['quote']}»")
        if state["citation_errors"]:
            lines += ["", f"_{t(lang, 'unverified_note')}_", ""] + [f"- {e}" for e in state["citation_errors"]]
        message = _final(
            AIMessage("\n".join(lines)), config, kind="documents", sources=cited, verified=not state["citation_errors"]
        )
        return {"messages": [message]}

    # ── tools ──────────────────────────────────────────────────────────
    def act(state: AgentState, config: RunnableConfig) -> dict:
        ctx = ctx_of(state, config)
        tools = tools_for(state["route"], ctx)
        exhausted = (
            state["tool_turns"] >= settings.max_tool_turns or max(state["repeats"].values(), default=0) >= REPEAT_STOP
        )
        note = ""
        if exhausted:
            note = (
                "You have used all the tool calls allowed for this turn. Reply now with what was done and what was not."
            )
        model = llm if exhausted else llm.bind_tools(tools)
        history = window(state["messages"], settings.history_messages)
        reply = model.invoke([system_prompt(state, tools_note=note), *history])
        calls = reply.tool_calls if not exhausted else []
        if not calls:
            reply, sources = with_memory(reply, state)
            reply = _final(reply, config, kind=state["route"], sources=sources)
        return {"messages": [reply], "trace": [*state["trace"], steps.thought(calls, concluded=not calls)]}

    def after_act(state: AgentState) -> Literal["tools", "verify"]:
        last = state["messages"][-1]
        return "tools" if isinstance(last, AIMessage) and last.tool_calls else "verify"

    def tools(state: AgentState, config: RunnableConfig) -> dict:
        ctx = ctx_of(state, config)
        by_name = {t.name: t for t in tools_for(state["route"], ctx)}
        last: AIMessage = state["messages"][-1]
        calls = last.tool_calls
        results: list[ToolMessage] = []
        evidence: list[dict] = []
        repeats = dict(state["repeats"])
        question = next((c for c in calls if c["name"] == "ask_user"), None)
        if question:
            # The node re-runs from the top when the thread resumes, so nothing else is executed alongside a question.
            answer = interrupt({"question": question["args"].get("question", "")})
            for call in calls:
                if call is question:
                    results.append(ToolMessage(f"The user answered: {answer}", tool_call_id=call["id"]))
                else:
                    results.append(
                        ToolMessage(
                            "Skipped: a question to the user was pending. Call again if still needed.",
                            tool_call_id=call["id"],
                        )
                    )
            return {
                "messages": results,
                "tool_turns": state["tool_turns"] + 1,
                "trace": [*state["trace"], steps.acted([{"tool": "ask_user", "ok": True, "summary": "answered"}])],
            }
        for call in calls:
            name, args = call["name"], call["args"] or {}
            signature = name + json.dumps(args, sort_keys=True, ensure_ascii=False)
            repeats[signature] = repeats.get(signature, 0) + 1
            started = time.perf_counter()
            if repeats[signature] > REPEAT_WARN:
                output = (
                    f"You already made this exact call {repeats[signature] - 1} times; its result has not changed. "
                    "Do something different or conclude."
                )
                ok = False
            elif name not in by_name:
                output, ok = f"Unknown tool {name!r}. Available: {', '.join(by_name)}.", False
            else:
                try:
                    output, ok = str(by_name[name].invoke(args)), True
                except ToolError as exc:
                    output, ok = f"Error: {exc}", False
                except Exception as exc:  # noqa: BLE001 — a tool crash is evidence of failure, not the end of the turn
                    output, ok = f"Error: {type(exc).__name__}: {exc}", False
            seconds = time.perf_counter() - started
            if ok and anonymizer and name in ("read_file", "grep", "read_page", "search_web", "run_command"):
                output = anonymizer.anonymize(output)  # what a tool brings in is pseudonymised like any input
            log_tool_call(conn, ctx.turn_id, name, args, ok, output, seconds)
            evidence.append({"tool": name, "args": args, "ok": ok, "summary": _summary(args), "output": output[:500]})
            where = {k: str(args[k]) for k in ("url", "path", "query") if args.get(k)}
            results.append(
                ToolMessage(prompts.data_block("tool_result", output, tool=name, **where), tool_call_id=call["id"])
            )
        return {
            "messages": results,
            "evidence": [*state["evidence"], *evidence],
            "tool_turns": state["tool_turns"] + 1,
            "repeats": repeats,
            "trace": [*state["trace"], steps.acted(evidence)],
        }

    def verify(state: AgentState, config: RunnableConfig) -> dict:
        ctx = ctx_of(state, config)
        reply: AIMessage = state["messages"][-1]
        history = turn_messages(state["messages"])
        if not state["evidence"]:
            if config.get("configurable", {}).get("source") == "alarm":
                # The turn that announces a due alarm has nothing to do but say so.
                return _verdict(state, reply, config, 0, [], False)
            # A tool route that never called a tool did nothing, whatever the reply says: the one case
            # that needs no model call to judge. Announcements and questions asked in prose land here.
            checked, failures = 1, [prompts.NO_TOOL_USED]
            retrying = state["verify_attempts"] < MAX_VERIFY_RETRIES
            return _verdict(state, reply, config, checked, failures, retrying)
        result: Claims = claimer.invoke(
            [SystemMessage(prompts.CLAIMS), HumanMessage(f"User asked: {history[0].content}\n\nReply: {reply.content}")]
        )
        root = (settings.projects_dir / state["project"]).resolve() if state.get("project") else None
        evidence = list(state["evidence"])
        fetched = []
        if state["route"] == "web" and state["verify_attempts"] < MAX_VERIFY_RETRIES:
            # The one unproven claim the code can settle itself: a page the reply says it read. Reading it
            # here turns the retry into "answer from this", instead of hoping the model calls read_page.
            for url in unread_urls(result.claims, evidence)[:2]:
                started = time.perf_counter()
                try:
                    text = (
                        anonymizer.anonymize(fetch_text(url, settings.web_max_chars))
                        if anonymizer
                        else fetch_text(url, settings.web_max_chars)
                    )
                    ok = True
                except ToolError as exc:
                    text, ok = f"Error: {exc}", False
                log_tool_call(conn, ctx.turn_id, "read_page", {"url": url}, ok, text, time.perf_counter() - started)
                evidence.append(
                    {
                        "tool": "read_page",
                        "args": {"url": url},
                        "ok": ok,
                        "summary": f"url={url!r}",
                        "output": text[:500],
                    }
                )
                if ok:
                    fetched.append(f"Content of {url} (read for you; answer from it and cite it):\n{text[:6000]}")
        checked, failures = check_claims(result.claims, evidence, project_root=root, conn=conn, thread_id=ctx.thread_id)
        if fetched:
            failures = [*failures, *fetched]  # the retry sees the page; the claim it supports now holds
        retrying = bool(failures) and state["verify_attempts"] < MAX_VERIFY_RETRIES
        update = _verdict(state, reply, config, checked, failures, retrying)
        update["evidence"] = evidence
        if fetched:
            update["trace"][-1] = steps.verified(checked, [f[:120] for f in failures], retrying)
        return update

    def _verdict(state, reply, config, checked, failures, retrying) -> dict:
        update = {
            "verification": {"checked": checked, "failed": len(failures), "failures": failures},
            "verify_attempts": state["verify_attempts"] + (1 if failures else 0),
            "verify_feedback": failures if retrying else [],
            "trace": [*state["trace"], steps.verified(checked, failures, retrying)],
        }
        if retrying:
            update["messages"] = [RemoveMessage(id=reply.id)]  # the unproven reply never reaches the user
        elif failures:
            marked = AIMessage(reply.content + "\n\n_" + t(lang, "unverified", failures="; ".join(failures)) + "_")
            update["messages"] = [
                _final(
                    marked,
                    config,
                    kind=state["route"],
                    verified=False,
                    sources=reply.additional_kwargs.get("sources", []),
                ),
                RemoveMessage(id=reply.id),
            ]
        return update

    def after_verify(state: AgentState) -> Literal["act", "memorize"]:
        return "act" if state["verify_feedback"] else "memorize"

    # ── memorize ───────────────────────────────────────────────────────
    def memorize(state: AgentState, config: RunnableConfig) -> dict:
        conf = config.get("configurable", {})
        if conf.get("source") == "alarm":
            return {"trace": [*state["trace"], steps.entry("memorize", "Alarm turns are not memorized.")]}
        turn = turn_messages(state["messages"])
        user, reply = turn[0].content, state["messages"][-1].content
        known = "\n".join(f"- {f['text']}" for f in state["memories"].get("facts", [])) or "- (none)"
        # Only the user's words are offered to the extractor, on every route. What Aroa replied is either
        # document content (cited there, never memory) or the model's own text (not a fact). This also keeps a
        # planted sentence in a document or page from becoming a durable fact.
        spoken = strip_attachments(user)  # a file attached to the message is not something the user said
        if not spoken.strip():
            return {"facts_written": [], "trace": [*state["trace"], steps.memorized([], [])]}
        result: Facts = extractor.invoke(
            [SystemMessage(prompts.FACTS), HumanMessage(f"Known facts:\n{known}\n\nUser: {spoken}")]
        )
        thread_id, turn_id = conf.get("thread_id", "default"), conf.get("turn_id", "no-turn")
        matter = conf.get("matter", "")
        origin = f"{state['route']}:user"
        stored, skipped = memory.add_facts(
            result.facts, thread_id=thread_id, turn_id=turn_id, matter=matter, origin=origin
        )
        for fact in stored:
            log_memory_write(conn, turn_id, "fact", fact, True)
        for fact in skipped:
            log_memory_write(conn, turn_id, "fact", fact, False)
        memory.add_episode(user, reply, thread_id=thread_id, turn_id=turn_id, matter=matter)
        return {"facts_written": stored, "trace": [*state["trace"], steps.memorized(stored, skipped)]}

    graph = StateGraph(AgentState)
    for name, node in [
        ("recall", recall),
        ("route", route),
        ("chat", chat),
        ("retrieve", retrieve),
        ("grade", grade),
        ("answer_documents", answer_documents),
        ("check_citations", check_citations),
        ("finalize_documents", finalize_documents),
        ("act", act),
        ("tools", tools),
        ("verify", verify),
        ("memorize", memorize),
    ]:
        graph.add_node(name, _timed(node))
    graph.add_edge(START, "recall")
    graph.add_edge("recall", "route")
    graph.add_conditional_edges("route", after_route, ["chat", "retrieve", "act", END])
    graph.add_edge("chat", "memorize")
    graph.add_edge("retrieve", "grade")
    graph.add_conditional_edges("grade", after_grade)
    graph.add_edge("answer_documents", "check_citations")
    graph.add_conditional_edges("check_citations", after_citations)
    graph.add_edge("finalize_documents", "memorize")
    graph.add_conditional_edges("act", after_act)
    graph.add_edge("tools", "act")
    graph.add_conditional_edges("verify", after_verify)
    graph.add_edge("memorize", END)
    return graph.compile(checkpointer=checkpointer)


DATE_WORDS = re.compile(
    r"\b\d{4}-\d{2}-\d{2}\b|\b\d{1,2}/\d{1,2}/\d{2,4}\b|\b\d{1,2}\s+de\s+(?:enero|febrero|marzo|abril|mayo|junio|julio|agosto|"
    r"septiembre|setiembre|octubre|noviembre|diciembre)\b",
    re.IGNORECASE,
)
TERM_WORDS = re.compile(
    r"\b(plazo|vence|vencimiento|recurrir|recurso|alegaciones|contestar|días?|mes(?:es)?|año|años)\b", re.IGNORECASE
)


def cites_law(text: str) -> bool:
    """The message names an article of a norm («artículo 9 de la LAU», «art. 135.5 LEC»)."""
    from .lawcheck import extract_citations

    return any(c.kind == "article" and c.law for c in extract_citations(text))


WEB_WORDS = re.compile(
    r"\b(busca|buscar|búscalo|internet|web|en línea|online|boe\.es|https?://|noticias|actualidad)\b", re.IGNORECASE
)


def wants_web(text: str) -> bool:
    """The user explicitly asks for a search or a page: the web route stays."""
    return bool(WEB_WORDS.search(text))


def deadline_in(text: str) -> bool:
    """A message that names a concrete date and a term is asking when something ends."""
    return bool(DATE_WORDS.search(text)) and bool(TERM_WORDS.search(text))


FILE_WORK = re.compile(
    r"\b(crea|crear|guarda|guardar|escribe|escribir|anota|apunta|archivo|archivos|fichero|ficheros|carpeta|nota|notas|"
    r"lee|leer|abre|abrir|lista|listar|borra|borrar|elimina|eliminar|modifica|renombra|ejecuta|redacta|"
    r"file|files|folder|create|write|save|list|read|delete|draft)\b",
    re.IGNORECASE,
)


def wants_files(text: str) -> bool:
    """The message asks for work on files, not just a question that names a matter."""
    return bool(FILE_WORK.search(text))


def project_in(text: str, names: list[str]) -> str:
    """The matter folder the text names, if any ("2026-014" also matches "2026/014"); longest name wins."""
    flat = text.lower().replace("/", "-")
    for name in sorted(names, key=len, reverse=True):
        if name.lower() in flat:
            return name
    return ""


def _final(message: AIMessage, config: RunnableConfig, **meta) -> AIMessage:
    """Mark the turn's reply with its turn id and kind, so clients can fetch its path and audit."""
    conf = config.get("configurable", {})
    message.additional_kwargs = {**message.additional_kwargs, "turn_id": conf.get("turn_id"), **meta}
    return message


def _summary(args: dict) -> str:
    text = ", ".join(f"{k}={str(v)[:60]!r}" for k, v in args.items())
    return text[:160]


def _fuse(*rankings: list[Chunk], k: int = 60) -> list[Chunk]:
    scores: dict[str, float] = {}
    chunks: dict[str, Chunk] = {}
    for ranking in rankings:
        for rank, chunk in enumerate(ranking, start=1):
            scores[chunk.id] = scores.get(chunk.id, 0) + 1 / (k + rank)
            chunks[chunk.id] = chunk
    ordered = sorted(scores, key=scores.get, reverse=True)
    return [chunks[i] for i in ordered[: max(map(len, rankings))]]
