#!/usr/bin/env bash
# sync_and_train.sh — Rsync code to remote server and launch training.
# Usage:
#   bash finetune/sync_and_train.sh            # 300 steps, no GGUF
#   bash finetune/sync_and_train.sh --export_gguf  # export after training
#
# Prerequisites:
#   Local : sshpass  (brew install sshpass)
#   Remote: unsloth installed  (run setup_remote.sh first)

set -euo pipefail

REMOTE="hastws@100.117.100.117"
PASS="xianglaoponi"
REMOTE_DIR="/home/hastws/gemma-4-good-hackathon"
SSH="sshpass -p $PASS ssh -o StrictHostKeyChecking=no $REMOTE"
LOCAL_DIR="$(cd "$(dirname "$0")/.." && pwd)"

EXTRA_ARGS="${*:-}"

echo "==> Syncing project to remote …"
sshpass -p "$PASS" rsync -avz --exclude __pycache__ --exclude .git \
  --exclude "runs/" --exclude "*.egg-info" \
  "$LOCAL_DIR/" "$REMOTE:$REMOTE_DIR/"

echo "==> Building dataset on remote (if not cached) …"
$SSH "cd $REMOTE_DIR && \
  [ -f data/eduedge_train.jsonl ] && \
  echo 'Dataset already exists, skipping.' || \
  python3 finetune/dataset.py"

echo "==> Starting training (nohup, logs → runs/train.log) …"
$SSH "cd $REMOTE_DIR && \
  mkdir -p runs && \
  nohup python3 finetune/train.py \
    --steps 300 \
    --export_gguf \
    --output runs/eduedge-v1 \
    $EXTRA_ARGS \
    > runs/train.log 2>&1 &
  echo \"Training PID: \$!\""

echo ""
echo "==> Training launched in background on remote."
echo "    To tail logs: ssh hastws@100.117.100.117 'tail -f $REMOTE_DIR/runs/train.log'"
echo "    To check GPU: ssh hastws@100.117.100.117 'nvidia-smi'"
