#!/usr/bin/env bash
# Start a local llama.cpp server for Temis (provider local:, LLM_STRUCTURED=json_schema).
# The first run downloads the model from Hugging Face (~5 GB) into llama.cpp's cache.
#   ./scripts/llama-server.sh            # Qwen3-8B, port 8080
#   MODEL=Qwen/Qwen3-14B-GGUF:Q4_K_M ./scripts/llama-server.sh
set -euo pipefail
MODEL="${MODEL:-Qwen/Qwen3-8B-GGUF:Q4_K_M}"
PORT="${PORT:-8080}"
ALIAS="${ALIAS:-qwen3-8b}"
command -v llama-server >/dev/null || { echo "llama-server not found: brew install llama.cpp" >&2; exit 1; }
exec llama-server -hf "$MODEL" --port "$PORT" -ngl 99 -c 16384 -fa on --jinja --alias "$ALIAS"
