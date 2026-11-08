# Plan

Each phase ends with a verifiable "done" and a commit.

## Phase 1 — core
Graph with `recall`, `route`, `chat`, documents route, `memorize`; memory store with
three collections; telemetry (tracing, metrics, audit, path); CLI (`chat`, `ask`,
`ingest`, `eval`, `report`); evals for identity, memory and citations.
**Done when:** `pytest` passes with the fake model, `aroa eval` passes identity,
memory and citation cases with a local Ollama model, and `/metrics` lists the turns.

## Phase 2 — tools
Alarms with scheduler, web search and reading, project tools confined to a folder,
`ask_user` interrupts, `verify` node, loop guards. FastAPI server with SSE streaming,
web UI with the path view and feedback.
**Done when:** evals for alarms, web and project pass, a turn that claims an unwritten
file is sent back and corrected, and an interrupted thread resumes after the answer.

## Phase 3 — deployment
Run against an OpenAI-compatible local gateway on the target machine, traces to its
collector, comparison table against the previous assistant on the same cases.
**Done when:** the table is in the README with real numbers.

## Phase 4 — polish
README with diagram and real transcript, GitHub Actions (ruff + pytest with the fake
model), skills for adding a tool and running evals, ADRs.

---

# Plan v2 — document work

v1 answers questions. v2 makes Aroa work *on* documents: long ones, chosen ones, with every
statement backed by a citation that opens where it came from, reviewed in a separate pass, written
in the firm's style, inside an editor. Same rules as v1: each phase ends with a verifiable "done"
and a commit; nothing is claimed without a check in code.

## Phase 5 — choose the documents, choose the model
- Document selector: the user picks which indexed documents a conversation works on; retrieval,
  hints and the writer are confined to them (filter on every prefetch, like the matter wall).
- `claude-cli:<model>` provider: runs the Claude Code CLI in print mode with a JSON schema for
  structured output and emulated tool calls, so the evals can run on a Claude subscription without
  an API key. For a deployment the `anthropic:` provider with a key remains the way.
- Citations stay clickable and open the source at the quote (v1); every answer from documents
  carries one per statement, and a statement without one is not shown as fact.
**Done when:** a selected-documents eval answers only from them and abstains when they do not
cover the question; the full eval runs on `claude-cli:sonnet` and its table is in the README next
to the local model's.

## Phase 6 — long documents, whole
_Done 2026-11-08: analysis job, merge and verified citations (ADR 009), clippings from the reader; the 320-clause eval passes on the local model._
- Analysis of a document of any length: map over sections or pages (chunks of the indexed text)
  with a per-section extraction, then a reduce that merges findings, deduplicates them and keeps
  every finding tied to the sections it came from. No page limit: the budget is time, not context.
- Repetition control: the reduce step merges findings that say the same thing from different
  sections into one, listing all its locations.
- Clippings: the user can cut a passage out of a document into a "clipping" (text + source +
  location) that lives in the matter and can be cited like a fragment.
**Done when:** a 200-page sample contract (generated, fictitious) yields an analysis where every
finding has at least one citation that resolves to its page, no finding is repeated, and the run
is a separate background job with progress in the UI.

## Phase 7 — the review pass
_Done 2026-11-08: review job (ADR 010): repeats, quotes and legal citations in code with proposed changes, model pointers as warnings; the planted-draft eval flags exactly the planted problems._
- A second process, separate from writing: it takes a draft and checks it. Citations exist and
  are verbatim (code); references to laws, articles and cases resolve (lookup against the firm's
  documents and, when allowed, the web); legal issues flagged with the passage and the reason;
  style and repetition.
- Output: a review report with findings that link to the draft positions, and a corrected draft
  the user accepts or rejects change by change.
**Done when:** a draft with a planted wrong article number, a non-verbatim quote and a repeated
paragraph gets all three flagged, and nothing else.

## Phase 8 — style and templates
- Reference documents marked as *style*: writing conventions, headings, formulas the firm uses.
  Marked as *template*: a structure to fill. Both live in the index with a kind, never cited as
  facts, injected tagged into the writer as data.
**Done when:** the same brief written with and without a style document differs where the style
says it should, and a template produces a draft with every section present.

## Phase 9 — correspondence and matter state
- Emails and letters of a matter imported as documents with a kind (`correspondence`), direction,
  date and status (answered / pending). Aroa can say what is pending and draft the answer from the
  thread; status changes are explicit actions, logged in the audit.
**Done when:** "what is unanswered in matter X?" lists the right items from a sample mailbox and
marking one as answered is recorded with who and when.

## Phase 10 — editor and reader
_Editor done 2026-11-08 (ADR 011): TinyMCE vendored, .html native, Word import/export, versions, open from chat and from the review. Pending: tracked suggestions from the review inside the editor, PDF reader with page navigation._
- A document editor in the web UI (professional: headings, lists, tables, footnotes, tracked
  suggestions from the review pass) that saves into the matter folder and keeps citations as
  links that open the source.
- A PDF reader in the UI for the sources, with page navigation and highlight of the cited span;
  reuse of the MegaPDF viewer if its licence allows embedding, otherwise pdf.js.
**Done when:** a draft written in the editor, reviewed, corrected and exported keeps every citation
resolvable from the exported file.

## Phase 11 — deployment (was phase 3)
As before: run on the target machine against its gateway, compare with the previous assistant.
