#!/bin/bash
# Download models for offline VoxCPM demo

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODELS_DIR="$SCRIPT_DIR/models"

export HF_HOME="$MODELS_DIR/huggingface"
export TRANSFORMERS_CACHE="$MODELS_DIR/huggingface"
export MODELSCOPE_CACHE="$MODELS_DIR/modelscope"

export HF_ENDPOINT="https://hf-mirror.com"
#!/bin/bash
# 自动加载 .env 文件中的变量
if [ -f .env ]; then
    export $(cat .env | xargs)
    # 打印加载的环境变量（可选）
    echo "Loaded environment variables from .env:"
    echo "HUGGING_FACE_HUB_TOKEN: $HUGGING_FACE_HUB_TOKEN"

fi

# export HUGGING_FACE_HUB_TOKEN=""

mkdir -p "$HF_HOME" "$MODELSCOPE_CACHE"

echo "Models will be downloaded to: $MODELS_DIR"
echo "HuggingFace cache: $HF_HOME"
echo "ModelScope cache: $MODELSCOPE_CACHE"

cd "$SCRIPT_DIR"

uv run download_models.py