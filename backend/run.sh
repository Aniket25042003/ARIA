#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

# Create audio output directory
mkdir -p audio_output

# Load .env if present
if [ -f .env ]; then
    set -a
    source .env
    set +a
fi

echo "Starting ARIA backend on 0.0.0.0:8000 ..."
exec uvicorn app.main:app --host 0.0.0.0 --port 8000
