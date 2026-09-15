#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET_DIR="$SCRIPT_DIR/liveportrait_src/pretrained_weights"

echo "=========================================================="
echo " Downloading LivePortrait Pretrained Weights from Hugging Face"
echo " Target directory: $TARGET_DIR"
echo "=========================================================="

mkdir -p "$TARGET_DIR"

if command -v huggingface-cli &> /dev/null; then
    HF_CMD="huggingface-cli"
elif [ -f "$SCRIPT_DIR/.venv/bin/huggingface-cli" ]; then
    HF_CMD="$SCRIPT_DIR/.venv/bin/huggingface-cli"
else
    echo "Error: huggingface-cli not found in PATH or .venv."
    echo "Install it via: pip install -U \"huggingface_hub[cli]\""
    exit 1
fi

"$HF_CMD" download KlingTeam/LivePortrait \
    --local-dir "$TARGET_DIR" \
    --exclude "*.git*" "README.md" "docs"

echo "Weights downloaded successfully to $TARGET_DIR"
