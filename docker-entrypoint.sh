#!/bin/bash
set -e

OLLAMA_HOST="${OLLAMA_HOST:-http://ollama:11434}"
OLLAMA_MODEL="${OLLAMA_MODEL:-qwen3.5}"

echo "=== Zipsa ==="
echo "Ollama host: $OLLAMA_HOST"
echo "Ollama model: $OLLAMA_MODEL"

# Wait for Ollama to be ready
echo "Waiting for Ollama..."
MAX_RETRIES=60
RETRY=0
until curl -sf "$OLLAMA_HOST/api/tags" > /dev/null 2>&1; do
    RETRY=$((RETRY + 1))
    if [ $RETRY -ge $MAX_RETRIES ]; then
        echo "WARNING: Ollama not reachable after ${MAX_RETRIES}s. Starting anyway..."
        break
    fi
    sleep 1
done

if [ $RETRY -lt $MAX_RETRIES ]; then
    echo "Ollama is ready."

    # Pull model if not already available
    if ! curl -sf "$OLLAMA_HOST/api/tags" | grep -q "$OLLAMA_MODEL"; then
        echo "Pulling model $OLLAMA_MODEL (this may take a few minutes on first run)..."
        curl -sf "$OLLAMA_HOST/api/pull" -d "{\"name\": \"$OLLAMA_MODEL\"}" > /dev/null 2>&1 || \
            echo "WARNING: Model pull may have failed. Continuing..."
    else
        echo "Model $OLLAMA_MODEL already available."
    fi
fi

# Copy .env.example to .env if .env doesn't exist
if [ ! -f .env ]; then
    cp .env.example .env 2>/dev/null || true
fi

echo "Starting Zipsa on :8000..."
exec uvicorn main:app --host 0.0.0.0 --port 8000
