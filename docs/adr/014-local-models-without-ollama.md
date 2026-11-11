# ADR 014 — Local models through any OpenAI-compatible server; settings in the app

**Status:** accepted · 2026-11

## Context

The default local runtime was Ollama. The firm may not want to depend on that company, and on Apple
Silicon there are open alternatives: llama.cpp (MIT, the engine Ollama wraps) and MLX (Apple's own
framework, open source). Both, and LM Studio, expose an OpenAI-compatible HTTP API on the machine.

## Decision

- New provider `local:<model>`: `ChatOpenAI` pointed at `LLM_BASE_URL` (default `http://127.0.0.1:8080/v1`),
  no key, thinking disabled through the chat template unless `LLM_REASONING=true`.
- Structured output method per server, `LLM_STRUCTURED`: `json_schema` (exact, grammar-constrained:
  llama.cpp, LM Studio), `function_calling` (MLX's server has no schema support but supports tools) or
  `json_mode`. A reply that comes back as prose is read for its JSON; otherwise the call is repeated once
  with an explicit instruction; the caller still receives `None` if both fail. Name detection treats a
  missing answer as "no names" and logs it; identifiers are always replaced by pattern.
- A Settings page writes the model, server, output mode, privacy and keys to `.env` and applies them live
  (`Assistant.reconfigure` rebuilds models, name detector and graph under the turn lock). A "Test the
  model" button makes one plain and one structured call and reports both times. Keys are never returned.
- The page carries a guide to install a local model with llama.cpp, MLX, LM Studio or Ollama.

## Measured on an M3 Max, Qwen3-8B 4-bit, same prompt, warm

| Runtime | ~270 tokens |
|---|---|
| Ollama | 6.4 s |
| llama.cpp | 6.4 s |
| MLX | 5.1 s |

## Measured with Temis's evals (same Qwen3-8B 4-bit, same dataset)

| Runtime | Passed | Note |
|---|---|---|
| llama.cpp, `json_schema` | 38/44 | documents 17/20, review 2/2, long contract in 1004 s |
| Ollama | 37/42 | earlier run, before two cases were added |
| MLX, `function_calling` | 25/44 | documents 7/20: answers right, `[n]` markers missing; long contract in 1440 s |

## Consequences

- Ollama stays supported but is no longer required. llama.cpp is the recommended local runtime: open
  source, the engine Ollama wraps, grammar-constrained structured output, same quality. MLX generates
  loose text faster but, without schema support in its server, loses citations in Temis's pipeline.
- `scripts/llama-server.sh` starts the recommended server (downloads the model on first run).
