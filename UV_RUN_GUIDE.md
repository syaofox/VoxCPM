# 使用 uv run 运行 VoxCPM（CUDA 加速 + VoxCPM 1.5）

## 快速开始

### 1. CLI 命令行使用

```bash
# 基本文本转语音（使用 VoxCPM 1.5，默认模型）
uv run voxcpm --text "Hello, this is VoxCPM 1.5 with CUDA acceleration." --output output.wav

# 指定使用 VoxCPM 1.5（显式指定）
uv run voxcpm --text "Hello world" --output out.wav --hf-model-id openbmb/VoxCPM1.5

# 声音克隆（需要参考音频和文本）
uv run voxcpm --text "目标文本" --prompt-audio reference.wav --prompt-text "参考文本" --output cloned.wav

# 批量处理
uv run voxcpm --input texts.txt --output-dir ./outputs
```

### 2. Web 界面使用

```bash
# 启动 Web 界面（默认使用 VoxCPM 1.5）
uv run python app.py

# 指定端口和主机
uv run python app.py --server-port 7860 --server-name 0.0.0.0
```

### 3. Python API 使用

```python
# 在 Python 脚本中使用
import soundfile as sf
from voxcpm import VoxCPM

# 加载 VoxCPM 1.5 模型（自动使用 CUDA）
model = VoxCPM.from_pretrained("openbmb/VoxCPM1.5")

# 生成语音
wav = model.generate(
    text="VoxCPM 1.5 is running with CUDA acceleration.",
    cfg_value=2.0,
    inference_timesteps=10,
)

# 保存音频
sf.write("output.wav", wav, model.tts_model.sample_rate)
```

## 验证 CUDA 和模型

运行测试脚本验证配置：

```bash
uv run python test_cuda_voxcpm15.py
```

## 环境变量配置

如果需要设置默认模型，可以使用环境变量：

```bash
# 设置默认模型为 VoxCPM 1.5（Web 界面）
export HF_REPO_ID=openbmb/VoxCPM1.5
uv run python app.py
```

## 注意事项

1. **CUDA 加速**：系统会自动检测并使用 CUDA（如果可用）
2. **模型版本**：默认使用 VoxCPM 1.5（`openbmb/VoxCPM1.5`）
3. **采样率**：VoxCPM 1.5 支持 44.1kHz 采样率
4. **首次运行**：首次运行会自动下载模型文件（约 1.5GB）

## 优势

使用 `uv run` 的优势：
- ✅ 无需手动激活虚拟环境
- ✅ 自动使用项目的虚拟环境
- ✅ 命令简洁，适合脚本和自动化
- ✅ 自动管理依赖
