# Aroa Assistant — specification

An assistant for a small law firm that runs entirely on the firm's own hardware, built as an explicit
LangGraph state machine instead of a hand-written agent loop. It remembers through retrieval (hybrid
RAG over facts, past conversations and the firm's documents) with a wall between matters, it
pseudonymises personal data before any model sees it, it refuses what the firm's policy forbids, it
can set deadline reminders, read the web and work inside matter folders (including redacting
documents), and every claim it makes about an action is checked in code before the lawyer reads it.
Everything it does is traced, measured, audited and evaluable.

## Goals

1. **Explicit control flow.** Routing, acting, verifying and memorising are graph nodes with typed
   state. The model decides *what* to say and *which* tool to call; the code decides retries, limits,
   when to give up, what is a name and what counts as proven.
2. **Confidentiality by construction.** Identifiers and names are replaced by stable placeholders
   before a message, a document or a tool result reaches a model, a store, a checkpoint or a trace.
   Only the interface reveals them. Memory of one matter is never recalled in another.
3. **Policy gate.** Requests to fabricate or alter evidence, deceive, cross matters, leak data or
   conceal a crime are blocked in the routing step, before any work.
4. **Memory that does not depend on the model remembering to save.** Facts are extracted from every
   turn and stored with deduplication; recall is hybrid (dense + BM25, RRF) over facts, episodes and
   documents.
5. **Verified claims.** After a tool route, the reply's claims are checked against the turn's
   evidence; document answers carry citations with verbatim quotes checked in code.
6. **Observable by default.** OpenTelemetry traces, per-turn metrics in SQLite and Prometheus, an
   audit log, a plain-language path per turn, feedback, offline evals and a daily report.
7. **Resumable.** A SQLite checkpointer keeps every thread; questions to the user are interrupts.

## Non-goals (v1)

Self-modification, multi-user authentication, cloud deployment, PDF redaction (text files only),
providers other than Anthropic, OpenAI-compatible gateways and Ollama.

## The graph

```
START → recall → route ──(blocked)──────────────────────────────────────────→ END
                   │
                   ├─ chat ─────────────────────────────────────────────┐
                   ├─ documents → retrieve → grade → answer_documents → check_citations → finalize ┐
                   └─ alarms | web | project → act ⇄ tools ─────┐       (retry once)               │
                                                (ask_user → interrupt)                              │
                                                              verify ◄──┘                           │
                                                                │ (unproven → act, once)             │
                                                                ▼                                    │
                                                            memorize ◄───────────────────────────────┘
                                                                │
                                                               END
```

| Node | Model | Does |
|---|---|---|
| `recall` | embeddings | Pseudonymises the message (replacing it in the checkpoint), then searches facts and episodes of this matter and general memory. |
| `route` | fast | One structured call: policy verdict, route (`chat`, `documents`, `alarms`, `web`, `project`), standalone query, matter folder. Sees the last turns, the recalled facts and the three document fragments closest to the message. A block is confirmed by a second, narrower call before it stands (false refusals cost more than one extra call). A message that names a matter folder is routed to `project` in code. |
| `chat` | main | Replies from persona + memories + history. One main-model call. |
| `retrieve` / `grade` / `answer_documents` / `check_citations` | — / fast / main / code | Corrective RAG over the firm's documents: hybrid search with the rewritten query and the user's words, keep what helps, answer with `[n]` citations and verbatim quotes, verify both in code, one retry. |
| `act` / `tools` | main | Tool loop bound to the route's tools. Guards: turn cap, identical-call detection, forced conclusion. Tool outputs are pseudonymised. `ask_user` raises an interrupt. |
| `verify` | fast + code | A tool route that never called a tool is rejected in code without a model call. Otherwise extracts the reply's action claims; code checks each against the evidence. Failures remove the reply and send the turn back once; a second failure delivers the reply marked unverified. |
| `memorize` | fast | Extracts durable facts (deduplicated by similarity, tagged with the matter), stores the episode. Runs after the reply is streamed. Alarm turns are not memorised. |

Latency budget: a chat turn costs one main-model call, one fast routing call and one fast facts call
(after the reply); in `full` privacy mode, one more fast call for names. Tool turns add the loop and
one verification call. Recall never calls a model.

## Privacy

`privacy.py`. Modes: `off`, `identifiers` (patterns in code: email, IBAN, card, DNI, NIE, social
security, phone, plate), `full` (also people and organisations proposed by the fast model and accepted
by `plausible_name`, which rejects role words and public bodies). Mapping in SQLite, one placeholder
per original, shared by conversations and documents. `reveal` at the interface only; `redact` for
documents that leave the firm (black boxes, irreversible).

## Untrusted content

Everything the model reads that the firm did not write (document fragments, tool results, web pages,
attachments, memory) goes through `prompts.data_block`: a tag naming its origin, closing tags inside
defused. Retrieved content never enters a system prompt: the router gets `<message>`,
`<document_hints>` and `<facts>` as a human turn, the writer gets `<fragments>`. On routes that read
external content (`documents`, `web`, `project`) the fact extractor sees the user's words only, and
every fact stores its `origin`. `run_command` runs one allowlisted command without a shell
(`ALLOWED_COMMANDS`). See ADR 006.

## Memory

Qdrant (embedded, or a server via `QDRANT_URL`) with fastembed vectors computed locally. Collections
`facts`, `episodes` (recency half-life), `documents` (files in `data/docs`, chunked by headings,
indexed incrementally by file hash, pseudonymised at ingest). Facts and episodes carry a `matter`;
recall filters to the current matter plus general memory on every prefetch. The persona
(`data/persona/*.md`) is always in the prompt, never retrieved.

## Tools

| Route | Tools |
|---|---|
| alarms | `create_alarm`, `list_alarms`, `delete_alarm` — SQLite; the server's scheduler wakes the thread when one is due. `compute_deadline` — procedural terms counted in code (`deadlines.py`: working days, August, the Christmas break, months date-to-date, art. 135.5 LEC grace) against the firm's holiday calendar (`data/calendar/holidays.json`, `COURT_REGION`, `LOCAL_HOLIDAYS`); the reply lists every excluded day and the rules applied |
| web | `search_web` (SearXNG-style JSON endpoint, or DuckDuckGo lite without configuration), `read_page` (HTML and PDF to text, size-capped, private addresses refused) |
| project | `list_files`, `read_file`, `grep`, `write_file`, `edit_file`, `run_command` (allowlisted, no shell), `redact_file` — confined to `data/projects/<matter>`; the folder's `AGENTS.md` is in the prompt |
| all tool routes | `ask_user` (interrupt), `remember` (explicit fact) |

## Whole-document analysis

`analysis.py`: a document of any length is analysed fragment by fragment (map, fast model, structured
findings each with a verbatim quote), then reduced in code: quotes not verbatim in their fragment are
dropped, findings that say the same thing are merged keeping every location, the result is numbered and
an optional overview written by the large model is kept only if every sentence cites an existing
finding. It runs as a job (`jobs` table: queued, running with done/total, done with the result, or
failed) through `POST /api/documents/{source}/analyze` and `GET /api/jobs/{id}`, one at a time, each
model call under the turn lock; the UI shows progress and opens the report with `[n]` citations that
open the fragment. `aroa analizar SOURCE` does the same synchronously. See ADR 009.

## Clippings

`clippings.py`: a passage selected in the reader is saved as `recorte-<doc>-<id>.md` in the documents folder, with a header naming the source document, the matter and an optional note, and the text verbatim. It is indexed, searched, cited and deleted like any document (`POST /api/clippings`).

## The review pass

`review.py`: a draft (pasted text, an attachment or an indexed document) is checked in a job separate
from writing. In code: repeated paragraphs (delete the later one), quoted passages not verbatim in the
indexed documents (the closest original sentence is the proposed fix), legal citations through
`lawcheck` (inexistent article, unknown norm, norm not downloaded, case law unverifiable) and the
article number corrected when the quoted text sits in another article of the same norm. The model
reads the draft in windows and points at legal issues; only pointers that quote the draft verbatim
survive, always as warnings. The result lists issues anchored to passages with a reason and, when
derivable, a change the user accepts or rejects; the UI marks them on the draft and downloads the
corrected text. `POST /api/reviews` queues it, `aroa revisar FILE` runs it synchronously. See ADR 010.

## Editor

`documents.py` + TinyMCE 6 (vendored): the firm's documents are edited in the web UI. Editable documents
are `.html` files in the documents folder; `.md`, `.txt`, `.pdf` and `.docx` open in the editor as a copy
to save under a name. `html_to_text` feeds the index, `docx_to_html`/`html_to_docx` bring Word in and
out (structural subset), and every save keeps the previous content as a version (`.versions/`). Endpoints:
`GET /api/documents/{source}/html`, `POST /api/documents/html`, `PUT /api/documents/{source}`,
`GET …/versions`, `POST …/versions/{v}/restore`, `POST /api/export/docx`. See ADR 011.

## Procedures

`procedures.py`: the firm's small flows as documents (`data/procedures/`, samples copied on first run)
with four sections: template (library document or inline), data (form fields), writing instructions and
review checklist. `POST /api/procedures/{slug}/draft` queues a `draft` job: placeholders filled in code,
`[[…]]` passages written by the model one by one, data checked back, result opened in the editor.
`POST /api/reviews` with `procedure` adds the checklist to the review (met only with a verbatim quote).
Procedures are created and edited in the app (`POST/PUT /api/procedures`). See ADR 012.

## Observability

- **Traces**: OpenInference spans over OTLP/HTTP to any collector; contents hidden unless
  `TRACE_CONTENT=true`, and pseudonymised even then. `turn_id` in the metadata.
- **Metrics**: one row per turn (route, seconds, per-node seconds, model calls, tokens, tool calls and
  errors, retrieved/kept, claims checked/failed, facts written, interrupted, error) and `/metrics`
  in Prometheus text.
- **Path**: the plain-language trace of the turn, stored with it, shown by the UI, the CLI (`-v`) and
  `GET /api/turns/{id}`.
- **Audit**: tool calls (arguments, outcome, result hash, seconds) and memory writes (fact, stored or
  duplicate) per turn.
- **Feedback**: thumbs per reply, tied to the turn.
- **Documents**: `GET/POST/DELETE /api/documents`; an upload is saved in `docs_dir` and indexed before
  the call returns, so the web UI never shows a document the assistant cannot see.
  `GET /api/documents/{source}/text` serves the original for the reader opened from a citation.
  `POST /api/attachments` extracts the text of a file for one turn (appended to the message, capped at
  12,000 characters) without saving or indexing it.
- **Language**: `ASSISTANT_LANGUAGE` (en | es) selects the texts the code writes (`i18n.py`): the fixed
  replies and the path of a turn. Prompts stay in English; the model answers in the user's language.
- **Evals**: `evals/dataset.jsonl` with kinds identity, memory, wall, privacy, policy, chat,
  documents, alarms, project. `aroa eval` runs them in a separate data folder and stores results.
- **Report**: `aroa report` / `GET /api/report`, computed in code.

## Configuration

`.env` (see `.env.example`): models as `<provider>:<model>`, reasoning toggle, privacy mode, embedding
model, data paths, search endpoint, OTLP endpoint. Nothing in the repository refers to a machine.
