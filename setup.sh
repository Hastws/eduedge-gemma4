#!/usr/bin/env bash
# setup.sh — One-command environment setup for EduEdge
# Usage: bash setup.sh

set -euo pipefail

echo "==> Installing Python dependencies..."
pip install -r requirements.txt

echo "==> Checking Ollama..."
if ! command -v ollama &>/dev/null; then
  echo "Ollama not found. Install from https://ollama.com/download"
  exit 1
fi

# Copy .env if not present
if [ ! -f .env ]; then
  cp .env.example .env
  echo "==> Created .env from .env.example (edit if needed)"
fi

# Load model choice from .env
MODEL="${GEMMA_MODEL:-gemma4:e4b}"

echo "==> Pulling Gemma 4 model: $MODEL"
echo "    (This is a one-time download. Size: ~9.6 GB for e4b)"
ollama pull "$MODEL"

echo ""
echo "✅  Setup complete!"
echo "    Run the app with: python -m src.app"
echo "    Then open: http://localhost:7860"
