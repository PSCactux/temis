# Deploying on the firm's machine

Temis is one Python process plus the model server you already run. Nothing here needs the internet
once the models are on disk.

## 1. Models

Any of:

- **Ollama** on the same machine or on the LAN: `LLM_MODEL=ollama:<model>`; set `OLLAMA_HOST` if it is
  not local.
- **An OpenAI-compatible gateway** (LiteLLM, vLLM, llama.cpp server, a NPU/GPU inference box):
  `OPENAI_BASE_URL=http://<gateway>/v1`, `OPENAI_API_KEY=<whatever it expects>`,
  `LLM_MODEL=openai:<name>`. Use a smaller, faster model for `LLM_FAST_MODEL` if the gateway offers
  one: routing, grading, claims and facts are short structured calls.
- **Anthropic**, if the firm's policy allows a provider outside the machine: `ANTHROPIC_API_KEY`,
  `LLM_MODEL=anthropic:claude-sonnet-5-5`. Personal data still never leaves pseudonymised.

Thinking models (qwen3 family) must run with thinking off for structured output: that is the default
(`LLM_REASONING=false`).

## 2. Embeddings

fastembed downloads the embedding model on first use (`intfloat/multilingual-e5-large`, about
2.2 GB). For an air-gapped machine, run `aroa ingest` once with internet, or copy the fastembed cache.
A lighter model: `EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`.

## 3. Data

Everything lives under `.data/` (SQLite files and the embedded Qdrant store) plus `data/docs`,
`data/projects`, `data/persona`. Back up those folders together; the pseudonym mapping is in
`.data/aroa.sqlite` and without it placeholders cannot be revealed.

Only one process can open the embedded Qdrant store. Run the server, and use the CLI through it, or
point both at a Qdrant server (`QDRANT_URL`).

## 4. Service

```ini
# /etc/systemd/system/aroa.service
[Unit]
Description=Temis
After=network.target

[Service]
User=aroa
WorkingDirectory=/opt/temis
EnvironmentFile=/opt/temis/.env
ExecStart=/opt/temis/.venv/bin/aroa serve --host 127.0.0.1 --port 8000
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

Put a reverse proxy with authentication in front of it (nginx with basic auth, or the firm's SSO):
the application has no users of its own in v1.

## 5. Observability

- Traces: `OTLP_ENDPOINT=http://<collector>/v1/traces`. Phoenix: `pip install arize-phoenix`,
  `phoenix serve`, endpoint `http://localhost:6006/v1/traces`.
- Metrics: scrape `http://127.0.0.1:8000/metrics` with Prometheus; `aroa report --days 7` for a weekly
  summary.
- Health: `GET /api/health`.

## 6. Comparing with what you had

Run `aroa eval` with the production models and keep the JSON. The cases of kinds `memory`, `wall`,
`privacy`, `policy`, `alarms` and `project` describe behaviours the previous assistant may not have
had at all; the `documents` cases measure retrieval and grounding on the same corpus.
