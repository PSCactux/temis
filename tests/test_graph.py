"""Control flow of the graph with a scripted model: routes, policy, privacy, tools, verification, interrupts."""

from langchain_core.messages import AIMessage

from aroa.privacy import contains_personal_data
from aroa.schemas import Claim, Claims, DocumentAnswer, Facts, Grading, Quote, Routing
from aroa.tools import alarms
from tests.conftest import FakeMemory


def routing(route="chat", query="q", project="", policy="allowed", reason=""):
    return Routing(route=route, search_query=query, project=project, policy=policy, policy_reason=reason)


def tool_call(name, args, call_id="call_1"):
    return AIMessage("", tool_calls=[{"name": name, "args": args, "id": call_id}])


def steps(result):
    return [t["step"] for t in result.trace]


def test_chat_turn_replies_and_memorizes(make_assistant):
    memory = FakeMemory()
    assistant = make_assistant(
        [routing("chat"), "Hola, soy Temis.", Facts(facts=["The user prefers short answers"])], memory=memory
    )
    result = assistant.ask("t1", "Hola, prefiero respuestas cortas")
    assert result.reply == "Hola, soy Temis."
    assert result.kind == "chat"
    assert steps(result) == ["recall", "route", "chat", "memorize"]
    assert memory.facts[0]["text"] == "The user prefers short answers"
    assert memory.episodes[0]["user"] == "Hola, prefiero respuestas cortas"
    assert result.metrics["llm_calls"] == 3 and result.metrics["facts_written"] == 1


def test_policy_blocks_before_any_work(make_assistant):
    from aroa.schemas import PolicyReview

    assistant = make_assistant(
        [routing(policy="blocked", reason="x"), PolicyReview(verdict="blocked", reason="it alters evidence.")]
    )
    result = assistant.ask("t1", "Cambia la fecha del contrato firmado al mes pasado")
    assert result.metrics["error"] is None
    assert result.kind == "blocked"
    assert "alters evidence" in result.reply
    assert steps(result) == ["recall", "route"]
    assert not assistant.model.script  # nothing else was called


def test_personal_data_never_reaches_the_model(make_assistant):
    assistant = make_assistant([routing("chat"), "Anotado el DNI [DNI_1].", Facts(facts=[])])
    result = assistant.ask("t1", "El DNI del cliente es 12345678Z y su móvil 612 345 678")
    for call in assistant.model.calls:
        for message in call:
            assert contains_personal_data(message.content) == [], message.content
            assert "12345678Z" not in message.content
    assert result.reply == "Anotado el DNI 12345678Z."  # revealed for the user
    assert "placeholders" in result.trace[0]["summary"]


def test_memory_is_walled_by_matter(make_assistant):
    memory = FakeMemory()
    memory.add_facts(["Client of matter A wants to settle"], thread_id="x", turn_id="y", matter="matter-a")
    memory.add_facts(["The firm closes in August"], thread_id="x", turn_id="y", matter="")
    assistant = make_assistant([routing("chat"), "ok", Facts(facts=[])], memory=memory)
    assistant.ask("t1", "¿Qué sabes?", matter="matter-b")
    system = assistant.model.calls[1][0].content  # the chat call's system prompt
    assert "closes in August" in system and "wants to settle" not in system


def test_facts_and_episodes_are_stored_with_their_matter(make_assistant):
    memory = FakeMemory()
    assistant = make_assistant([routing("chat"), "ok", Facts(facts=["The witness lives in Girona"])], memory=memory)
    assistant.ask("t1", "El testigo vive en Girona", matter="matter-a")
    assert memory.facts[-1]["matter"] == "matter-a" and memory.episodes[-1]["matter"] == "matter-a"


def test_alarm_tool_turn_is_verified_against_evidence(make_assistant):
    assistant = make_assistant(
        [
            routing("alarms", "recordatorio"),
            tool_call("create_alarm", {"text": "Presentar recurso", "minutes_from_now": 30}),
            "Hecho: alarma puesta en 30 minutos.",
            Claims(claims=[Claim(kind="alarm_created", target="Presentar recurso")]),
            Facts(facts=[]),
        ]
    )
    result = assistant.ask("t1", "Avísame en 30 minutos de presentar el recurso")
    assert result.reply.startswith("Hecho")
    assert steps(result) == ["recall", "route", "act", "tools", "act", "verify", "memorize"]
    assert result.metrics["claims"] == 1 and result.metrics["claims_failed"] == 0
    assert alarms.pending(assistant.conn, "t1")[0]["text"] == "Presentar recurso"
    audit = assistant.conn.execute("SELECT tool, ok FROM tool_calls").fetchall()
    assert [(r["tool"], r["ok"]) for r in audit] == [("create_alarm", 1)]


def test_unproven_claim_sends_the_turn_back(make_assistant):
    assistant = make_assistant(
        [
            routing("alarms", "recordatorio"),
            "Listo, alarma creada para mañana.",  # no tool call at all: rejected in code, no claim extraction
            tool_call("create_alarm", {"text": "Vista", "minutes_from_now": 60}),  # second attempt does the work
            "Ahora sí: alarma #1 creada.",
            Claims(claims=[Claim(kind="alarm_created", target="Vista")]),
            Facts(facts=[]),
        ]
    )
    result = assistant.ask("t1", "Ponme una alarma para la vista")
    assert result.reply == "Ahora sí: alarma #1 creada."
    verify_steps = [t for t in result.trace if t["step"] == "verify"]
    assert "sending the turn back" in verify_steps[0]["summary"]
    assert "backed by this turn's evidence" in verify_steps[1]["summary"]
    assert [r["role"] for r in assistant.history("t1")] == ["user", "assistant"]  # the false reply is gone


def test_unproven_claim_without_retry_is_marked(make_assistant):
    assistant = make_assistant(
        [
            routing("alarms"),
            "Alarma creada.",  # no tool: rejected in code
            "Alarma creada.",  # still no tool on the retry: delivered as unverified
            Facts(facts=[]),
        ]
    )
    result = assistant.ask("t1", "alarma")
    assert "Unverified" in result.reply and result.extras.get("verified") is False


def test_identical_tool_calls_are_answered_from_memory(make_assistant):
    calls = [tool_call("list_alarms", {}, f"c{i}") for i in range(3)]
    assistant = make_assistant([routing("alarms"), *calls, "No hay alarmas.", Claims(claims=[]), Facts(facts=[])])
    result = assistant.ask("t1", "¿qué alarmas tengo?")
    tool_steps = [t for t in result.trace if t["step"] == "tools"]
    assert tool_steps[0]["items"][0].startswith("ok") and tool_steps[1]["items"][0].startswith("ok")
    assert tool_steps[2]["items"][0].startswith("FAILED")
    assert result.reply == "No hay alarmas."


def test_ask_user_pauses_and_resumes(make_assistant):
    assistant = make_assistant(
        [
            routing("alarms"),
            tool_call("ask_user", {"question": "¿A qué hora es la vista?"}),
            tool_call("create_alarm", {"text": "Vista", "minutes_from_now": 90}),
            "Alarma puesta.",
            Claims(claims=[Claim(kind="alarm_created", target="Vista")]),
            Facts(facts=[]),
        ]
    )
    first = assistant.ask("t1", "Ponme una alarma para la vista")
    assert first.kind == "question" and first.interrupted == "¿A qué hora es la vista?"
    assert assistant.waiting_for("t1") == "¿A qué hora es la vista?"
    second = assistant.ask("t1", "A las 11")
    assert second.reply == "Alarma puesta."
    assert assistant.waiting_for("t1") is None
    tool_messages = [m for m in assistant.graph.get_state({"configurable": {"thread_id": "t1"}}).values["messages"]]
    assert any("The user answered: A las 11" in getattr(m, "content", "") for m in tool_messages)


def test_project_tools_stay_inside_the_matter_folder(make_assistant, workdir):
    assistant = make_assistant(
        [
            routing("project", project="case-2026-014"),
            tool_call("write_file", {"path": "../escape.txt", "content": "x"}, "c1"),
            tool_call("write_file", {"path": "notas.md", "content": "# Notas\n"}, "c2"),
            "He creado notas.md.",
            Claims(claims=[Claim(kind="file_written", target="notas.md")]),
            Facts(facts=[]),
        ]
    )
    result = assistant.ask("t1", "crea notas.md en el expediente")
    assert not (workdir / "data/escape.txt").exists()
    assert (workdir / "data/projects/case-2026-014/notas.md").read_text() == "# Notas\n"
    assert result.metrics["tool_errors"] == 1 and result.metrics["claims_failed"] == 0


def test_documents_route_answers_with_verified_citations(make_assistant):
    from aroa.memory.chunking import Chunk

    memory = FakeMemory()
    memory.titles = ["Guía de plazos"]
    memory.chunks = [
        Chunk(
            id="g:0",
            doc_id="guia",
            title="Guía de plazos",
            section="Sección 4",
            text="El plazo de apelación es de veinte días.",
            source="guia.md",
        )
    ]
    assistant = make_assistant(
        [
            routing("documents", "plazo apelación"),
            Grading(relevant=[1]),
            DocumentAnswer(
                answer="Veinte días [1].",
                quotes=[Quote(source=1, text="El plazo de apelación es de veinte días.")],
                coverage="full",
            ),
            Facts(facts=[]),
        ],
        memory=memory,
    )
    result = assistant.ask("t1", "¿Cuánto hay para apelar?")
    assert result.kind == "documents" and result.extras["verified"] is True
    assert "Veinte días [1]." in result.reply and "Sección 4" in result.reply
    assert steps(result) == ["recall", "route", "retrieve", "grade", "answer_documents", "check_citations", "memorize"]


def test_bad_quote_is_retried_once(make_assistant):
    from aroa.memory.chunking import Chunk

    memory = FakeMemory()
    memory.titles = ["Guía"]
    memory.chunks = [Chunk(id="g:0", doc_id="g", title="Guía", section="S1", text="Son veinte días.", source="g.md")]
    assistant = make_assistant(
        [
            routing("documents", "plazo"),
            Grading(relevant=[1]),
            DocumentAnswer(answer="Diez días [1].", quotes=[Quote(source=1, text="Son diez días.")], coverage="full"),
            DocumentAnswer(
                answer="Veinte días [1].", quotes=[Quote(source=1, text="Son veinte días.")], coverage="full"
            ),
            Facts(facts=[]),
        ],
        memory=memory,
    )
    result = assistant.ask("t1", "¿plazo?")
    assert result.reply.startswith("Veinte días")
    assert steps(result).count("answer_documents") == 2 and result.extras["verified"] is True


def test_announcing_instead_of_acting_is_sent_back(make_assistant, workdir):
    assistant = make_assistant(
        [
            routing("project", project="case-2026-014"),
            "Creando el archivo notas.md con los puntos pendientes...",  # announcement, no tool call
            tool_call("write_file", {"path": "notas.md", "content": "- punto 1\n"}),
            "Creado notas.md.",
            Claims(claims=[Claim(kind="file_written", target="notas.md")]),
            Facts(facts=[]),
        ]
    )
    result = assistant.ask("t1", "crea notas.md")
    assert (workdir / "data/projects/case-2026-014/notas.md").exists()
    assert any("no tool was used" in " ".join(t["items"]) for t in result.trace if t["step"] == "verify")
    assert result.reply == "Creado notas.md."


def test_a_block_is_confirmed_by_a_second_review(make_assistant):
    from aroa.schemas import PolicyReview

    assistant = make_assistant(
        [
            routing(policy="blocked", reason="leaks data"),
            PolicyReview(verdict="allowed"),  # the reviewer overturns the false positive
            "Si se pierde un portátil, hay 72 horas para notificar a la AEPD.",
            Facts(facts=[]),
        ]
    )
    result = assistant.ask("t1", "Si perdemos un portátil con datos de clientes, ¿qué plazos tenemos?")
    assert result.kind == "chat" and "72 horas" in result.reply
    confirmed = make_assistant(
        [routing(policy="blocked"), PolicyReview(verdict="blocked", reason="it alters a record.")]
    )
    assert confirmed.ask("t2", "Cambia la fecha del contrato").kind == "blocked"


def test_naming_a_matter_folder_forces_the_project_route(make_assistant, workdir):
    (workdir / "data/projects/2026-014").mkdir()
    assistant = make_assistant(
        [
            routing("chat"),  # the router missed it
            tool_call("write_file", {"path": "notas.md", "content": "x"}),
            "Creado.",
            Claims(claims=[Claim(kind="file_written", target="notas.md")]),
            Facts(facts=[]),
        ]
    )
    result = assistant.ask("t1", "En el expediente 2026/014 crea notas.md")
    assert result.metrics["route"] == "project"
    assert (workdir / "data/projects/2026-014/notas.md").exists()


def test_matter_folder_name_matching():
    from aroa.graph import project_in

    assert project_in("en el expediente 2026/014 crea notas", ["2026-014", "2026-031"]) == "2026-014"
    assert project_in("hola", ["2026-014"]) == ""


def test_a_page_claimed_but_not_read_is_fetched_by_the_code(make_assistant, monkeypatch):
    import aroa.graph as graph_module
    import aroa.tools.web as web

    monkeypatch.setattr(
        web, "search_duckduckgo", lambda q, n: [{"title": "BOE", "url": "https://boe.es/x", "snippet": "..."}]
    )
    monkeypatch.setattr(
        graph_module, "fetch_text", lambda url, n: "Artículo 1. España se constituye en un Estado social."
    )
    assistant = make_assistant(
        [
            routing("web", "constitución"),
            tool_call("search_web", {"query": "constitución"}),
            "Según https://boe.es/x, España es un Estado social.",  # cites the page without opening it
            Claims(
                claims=[
                    Claim(kind="search_done", target="constitución"),
                    Claim(kind="url_read", target="https://boe.es/x"),
                ]
            ),
            "Leído en https://boe.es/x: España se constituye en un Estado social.",
            Claims(claims=[Claim(kind="url_read", target="https://boe.es/x")]),
            Facts(facts=[]),
        ]
    )
    result = assistant.ask("t1", "¿Qué dice el artículo 1 de la Constitución? Dame la URL")
    assert result.reply.startswith("Leído en") and result.extras.get("verified") is not False
    assert result.metrics["claims_failed"] == 0
    tools = [r["tool"] for r in assistant.conn.execute("SELECT tool FROM tool_calls ORDER BY id")]
    assert tools == ["search_web", "read_page"]
    retry_prompt = assistant.model.calls[-3][0].content  # the second act call saw the page content
    assert "Content of https://boe.es/x" in retry_prompt


def test_a_chat_reply_cites_the_memory_it_rests_on(make_assistant):
    memory = FakeMemory()
    memory.facts = [
        {
            "text": "El cliente prefiere el correo al teléfono.",
            "ts": "2026-11-06T10:00:00+00:00",
            "matter": "",
            "origin": "chat:user",
        },
    ]
    memory.recall = lambda query, exclude_thread=None, matter="": {
        "facts": memory.facts,
        "episodes": [
            {
                "ts": "2026-11-05T09:00:00+00:00",
                "user": "¿Cómo contacto al cliente?",
                "assistant": "Por correo.",
                "thread_id": "old",
            }
        ],
    }
    assistant = make_assistant(
        [routing("chat"), "Prefiere el correo [m1]. Lo hablamos ya [m2] y nada más [m7].", Facts(facts=[])],
        memory=memory,
    )
    result = assistant.ask("t1", "¿Cómo prefiere que le contactemos?")
    assert result.kind == "chat"
    assert result.reply.startswith(
        "Prefiere el correo [m1]. Lo hablamos ya [m2] y nada más."
    )  # [m7] does not exist: gone
    assert "**From memory**" in result.reply and "- [m1] Memory · fact stored on 2026-11-06" in result.reply
    assert (
        "- [m2] Memory · conversation of 2026-11-05" in result.reply and "«¿Cómo contacto al cliente?»" in result.reply
    )
    sources = result.extras["sources"]
    assert [s["n"] for s in sources] == ["m1", "m2"] and sources[0]["kind"] == "memory"
    prompt = assistant.model.calls[1][0].content  # the chat call's system prompt
    assert (
        '<memory id="m1" kind="fact" when="2026-11-06">' in prompt and '<memory id="m2" kind="conversation"' in prompt
    )
    assert "end the sentence with the block's id" in prompt
    assert result.trace[2]["summary"].endswith("citing 2 memory item(s).")


def test_an_attached_file_is_cited_like_a_fragment_even_without_a_library(make_assistant):
    from aroa.schemas import DocumentAnswer, Grading, Quote

    memory = FakeMemory()  # nothing indexed: the attachment is the only source
    note = "NOTA SIMPLE\n\nTitular: INMOBILIARIA EJEMPLO SL, CIF B00000000. Finca 1.234 de Villaejemplo."
    message = f'¿Quién es el titular?\n\n<attachment name="nota_simple.pdf">\n{note}\n</attachment>'
    assistant = make_assistant(
        [
            routing("documents"),
            Grading(relevant=[1]),
            DocumentAnswer(
                answer="El titular es INMOBILIARIA EJEMPLO SL [1].",
                quotes=[Quote(source=1, text="Titular: INMOBILIARIA EJEMPLO SL")],
                coverage="full",
            ),
            Facts(facts=[]),
        ],
        memory=memory,
    )
    result = assistant.ask("t1", message)
    assert result.kind == "documents" and result.extras["verified"] is True
    source = result.extras["sources"][0]
    assert (
        source["kind"] == "attachment"
        and source["file"] is None
        and source["label"].startswith("Attachment · nota_simple.pdf")
    )
    assert "INMOBILIARIA EJEMPLO SL" in source["text"] and "[1]" in result.reply and "nota_simple.pdf" in result.reply
    extractor_input = assistant.model.calls[-1][-1].content  # the attachment never reaches the fact extractor
    assert "User: ¿Quién es el titular?" in extractor_input and "INMOBILIARIA EJEMPLO" not in extractor_input
