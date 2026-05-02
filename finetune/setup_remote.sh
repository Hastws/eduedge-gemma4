#!/usr/bin/env bash
# setup_remote.sh — Install Unsloth + dependencies on the training server
# Run once: bash finetune/setup_remote.sh
# Requires: sshpass installed locally  (brew install sshpass)

set -euo pipefail

REMOTE="hastws@100.117.100.117"
SSHPASS="xianglaoponi"
SSH="sshpass -p $SSHPASS ssh -o StrictHostKeyChecking=no $REMOTE"
SCP="sshpass -p $SSHPASS scp -o StrictHostKeyChecking=no"

echo "==> Checking remote GPU …"
$SSH 'nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader'

echo "==> Installing Unsloth + training deps on remote …"
$SSH 'pip install --upgrade pip -q && \
  pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git" -q && \
  pip install trl datasets transformers accelerate bitsandbytes -q && \
  pip install xformers --index-url https://download.pytorch.org/whl/cu121 -q 2>/dev/null || true && \
  python3 -c "import unsloth; print(\"unsloth OK\", unsloth.__version__)"'

echo "==> Done. Remote is ready for training."
