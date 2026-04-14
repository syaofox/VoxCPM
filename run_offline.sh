#!/bin/bash
# Run VoxCPM demo with offline models

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODELS_DIR="$SCRIPT_DIR/models"

export HF_HOME="$MODELS_DIR/huggingface"
export TRANSFORMERS_CACHE="$MODELS_DIR/huggingface"
export MODELSCOPE_CACHE="$MODELS_DIR/modelscope"

# Offline mode
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Local model paths
VOXCPM2_PATH="$MODELS_DIR/huggingface/VoxCPM2"
ZIPENHANCER_PATH="$MODELS_DIR/modelscope/speech_zipenhancer_ans_multiloss_16k_base"
SENSEVOICE_PATH="$MODELS_DIR/modelscope/SenseVoiceSmall"

cd "$SCRIPT_DIR"

uv run python app.py \
    --model-id "$VOXCPM2_PATH" \
    --zipenhancer-path "$ZIPENHANCER_PATH" \
    --sensevoice-path "$SENSEVOICE_PATH" \
    --local-files-only \
    --port 8809 "$@"