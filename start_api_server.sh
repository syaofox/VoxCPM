#!/bin/bash
# VoxCPM API 服务器启动脚本

# 检查是否安装了 fastapi 和 uvicorn
python3 -c "import fastapi, uvicorn" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "⚠️  检测到缺少依赖，正在安装 fastapi 和 uvicorn..."
    if command -v uv &> /dev/null; then
        uv pip install fastapi uvicorn
    else
        pip install fastapi uvicorn
    fi
fi

# 获取参数
HOST=${1:-0.0.0.0}
PORT=${2:-8000}

echo "🚀 正在启动 VoxCPM API 服务器..."
echo "📍 服务地址: http://${HOST}:${PORT}"
echo "📚 API 文档: http://${HOST}:${PORT}/docs"
echo "💚 健康检查: http://${HOST}:${PORT}/health"
echo ""

# 使用 uv 运行（如果可用）
if command -v uv &> /dev/null; then
    uv run python api_server.py --host "$HOST" --port "$PORT"
else
    python api_server.py --host "$HOST" --port "$PORT"
fi
