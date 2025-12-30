# VoxCPM API 服务使用指南

本指南介绍如何启动 VoxCPM API 服务，用于 Legado 阅读器的自定义朗读引擎。

## 📋 前置要求

1. **安装依赖**
   ```bash
   # 使用 uv（推荐）
   uv pip install fastapi uvicorn
   
   # 或使用 pip
   pip install fastapi uvicorn
   ```

2. **确保已安装 VoxCPM**
   ```bash
   # 如果使用 uv
   uv pip install -e .
   
   # 或使用 pip
   pip install -e .
   ```

## 🚀 启动 API 服务

### 基本启动

```bash
# 使用 uv（推荐）
uv run python api_server.py

# 或直接使用 Python
python api_server.py
```

### 自定义端口和主机

```bash
# 指定主机和端口
uv run python api_server.py --host 0.0.0.0 --port 8000

# 或
python api_server.py --host 0.0.0.0 --port 8000
```

### 参数说明

- `--host`: 服务器绑定的主机地址（默认: `0.0.0.0`）
- `--port`: 服务器端口（默认: `8000`）

## 📡 API 端点

### 1. 根路径
- **URL**: `GET /`
- **说明**: 返回 API 基本信息

### 2. 健康检查
- **URL**: `GET /health`
- **说明**: 检查服务状态和模型加载情况

### 3. 文本转语音（主要端点，JSON格式）
- **URL**: `POST /tts`
- **Content-Type**: `application/json`
- **请求体**:
  ```json
  {
    "text": "要朗读的文本内容",
    "cfg_value": 2.0,                    // 可选，CFG 值（默认: 2.0）
    "inference_timesteps": 10,           // 可选，推理时间步（默认: 10）
    "normalize": false,                   // 可选，文本正则化（默认: false）
    "denoise": false,                     // 可选，音频降噪（默认: false）
    "prompt_wav_path": "/path/to/ref.wav", // 可选，参考音频文件路径（服务器本地）
    "prompt_text": "参考音频的文本内容"    // 可选，参考音频对应的文本
  }
  ```
- **响应**: WAV 格式的音频文件
- **说明**: 
  - 如果提供 `prompt_wav_path` 和 `prompt_text`，将进行声音克隆
  - 参考音频文件必须是服务器本地可访问的路径

### 4. 文本转语音（支持上传参考音频）
- **URL**: `POST /tts/upload`
- **Content-Type**: `multipart/form-data`
- **请求参数**:
  - `text` (必需): 要朗读的文本内容
  - `prompt_audio` (可选): 参考音频文件（WAV格式）
  - `prompt_text` (可选): 参考音频对应的文本（上传音频时必需）
  - `cfg_value` (可选): CFG 值，默认 2.0
  - `inference_timesteps` (可选): 推理时间步，默认 10
  - `normalize` (可选): 文本正则化，默认 false
  - `denoise` (可选): 音频降噪，默认 false
- **响应**: WAV 格式的音频文件
- **说明**: 适合需要上传参考音频进行声音克隆的场景

### 5. 流式文本转语音（实验性）
- **URL**: `POST /tts/stream`
- **说明**: 支持流式生成，适合长文本

## 🔧 在 Legado 中配置

### 方法一：通过 URL 导入配置

1. 创建 Legado httpTTS 配置文件（JSON 格式）：
   ```json
   {
     "name": "VoxCPM TTS",
     "url": "http://你的服务器IP:8000/tts",
     "method": "POST",
     "headers": {
       "Content-Type": "application/json"
     },
     "body": "{\"text\":\"{content}\",\"cfg_value\":2.0,\"inference_timesteps\":10}",
     "concurrentRate": "0",
     "contentType": "audio/wav"
   }
   ```

2. 将配置文件上传到可访问的 URL（如 GitHub Gist、服务器等）

3. 在 Legado 中导入：
   ```
   legado://import/httpTTS?src=你的配置文件URL
   ```

### 方法二：手动配置

1. 打开 Legado 应用
2. 进入 **设置** → **朗读设置** → **朗读引擎**
3. 点击 **添加引擎** 或 **自定义引擎**
4. 配置如下：
   - **名称**: `VoxCPM TTS`
   - **请求 URL**: `http://你的服务器IP:8000/tts`
   - **请求方法**: `POST`
   - **请求头**:
     ```
     Content-Type: application/json
     ```
   - **请求体**:
     ```json
     {
       "text": "{content}",
       "cfg_value": 2.0,
       "inference_timesteps": 10,
       "normalize": false,
       "denoise": false
     }
     ```
   - **响应格式**: `音频文件`
   - **内容类型**: `audio/wav`

## 📝 使用示例

### 使用 curl 测试

```bash
# 测试健康检查
curl http://localhost:8000/health

# 测试标准 TTS（无参考音频）
curl -X POST http://localhost:8000/tts \
  -H "Content-Type: application/json" \
  -d '{"text":"你好，这是 VoxCPM 语音合成测试。"}' \
  --output output.wav

# 测试声音克隆（使用服务器本地参考音频）
curl -X POST http://localhost:8000/tts \
  -H "Content-Type: application/json" \
  -d '{
    "text":"这是使用参考音频生成的声音。",
    "prompt_wav_path": "/path/to/reference.wav",
    "prompt_text": "参考音频的原始文本内容",
    "cfg_value": 2.0,
    "inference_timesteps": 10
  }' \
  --output cloned.wav

# 测试上传参考音频进行声音克隆
curl -X POST http://localhost:8000/tts/upload \
  -F "text=这是使用上传的参考音频生成的声音。" \
  -F "prompt_audio=@reference.wav" \
  -F "prompt_text=参考音频的原始文本内容" \
  -F "cfg_value=2.0" \
  -F "inference_timesteps=10" \
  --output cloned.wav
```

### 使用 Python 测试

```python
import requests

# 测试标准 TTS（无参考音频）
response = requests.post(
    "http://localhost:8000/tts",
    json={
        "text": "你好，这是 VoxCPM 语音合成测试。",
        "cfg_value": 2.0,
        "inference_timesteps": 10
    }
)

with open("output.wav", "wb") as f:
    f.write(response.content)

# 测试声音克隆（使用服务器本地参考音频）
response = requests.post(
    "http://localhost:8000/tts",
    json={
        "text": "这是使用参考音频生成的声音。",
        "prompt_wav_path": "/path/to/reference.wav",
        "prompt_text": "参考音频的原始文本内容",
        "cfg_value": 2.0,
        "inference_timesteps": 10
    }
)

with open("cloned.wav", "wb") as f:
    f.write(response.content)

# 测试上传参考音频进行声音克隆
with open("reference.wav", "rb") as audio_file:
    response = requests.post(
        "http://localhost:8000/tts/upload",
        files={"prompt_audio": audio_file},
        data={
            "text": "这是使用上传的参考音频生成的声音。",
            "prompt_text": "参考音频的原始文本内容",
            "cfg_value": 2.0,
            "inference_timesteps": 10
        }
    )

with open("cloned_upload.wav", "wb") as f:
    f.write(response.content)
```

## 🎤 参考音频（声音克隆）使用说明

### 什么是参考音频？

参考音频（Prompt Audio）是 VoxCPM 的声音克隆功能，允许你使用一段参考音频来指导语音合成，生成与参考音频相似的声音特征（包括音色、语调、情感等）。

### 两种使用方式

#### 方式一：使用服务器本地参考音频（推荐用于固定音色）

1. **准备参考音频文件**
   - 格式：WAV 格式（推荐）
   - 时长：建议 5-30 秒
   - 质量：清晰、无背景噪音
   - 内容：包含自然、清晰的语音

2. **将参考音频上传到服务器**
   ```bash
   # 例如上传到服务器的 /data/reference_voices/ 目录
   scp reference.wav user@server:/data/reference_voices/
   ```

3. **在 API 请求中使用**
   ```json
   {
     "text": "要合成的文本",
     "prompt_wav_path": "/data/reference_voices/reference.wav",
     "prompt_text": "参考音频的原始文本内容"
   }
   ```

#### 方式二：上传参考音频文件（适合临时使用）

使用 `/tts/upload` 端点，通过 `multipart/form-data` 格式上传参考音频：

```bash
curl -X POST http://localhost:8000/tts/upload \
  -F "text=要合成的文本" \
  -F "prompt_audio=@reference.wav" \
  -F "prompt_text=参考音频的原始文本内容"
```

### 参考音频要求

- **格式**: WAV、MP3 等常见音频格式
- **时长**: 建议 5-30 秒，太短可能效果不佳，太长会增加处理时间
- **质量**: 
  - 清晰、无背景噪音
  - 采样率建议 16kHz 或更高
  - 单声道或立体声均可
- **内容**: 
  - 包含自然、清晰的语音
  - 避免过快的语速
  - 建议使用与目标文本相似的语言

### 参考文本要求

- **准确性**: 必须与参考音频的内容完全一致
- **格式**: 纯文本，无需标点符号（可选）
- **语言**: 建议与目标文本使用相同语言

### 声音克隆示例

```python
import requests

# 使用服务器本地参考音频
response = requests.post(
    "http://localhost:8000/tts",
    json={
        "text": "今天天气真好，适合出去散步。",
        "prompt_wav_path": "/data/reference_voices/voice1.wav",
        "prompt_text": "这是参考音频的原始文本内容",
        "cfg_value": 2.0,
        "inference_timesteps": 10,
        "denoise": False  # 如果参考音频质量好，可以设为 False
    }
)

# 使用上传的参考音频
with open("reference.wav", "rb") as f:
    response = requests.post(
        "http://localhost:8000/tts/upload",
        files={"prompt_audio": f},
        data={
            "text": "今天天气真好，适合出去散步。",
            "prompt_text": "这是参考音频的原始文本内容",
            "cfg_value": 2.0,
            "inference_timesteps": 10
        }
    )
```

### 在 Legado 中使用参考音频

由于 Legado 的 httpTTS 配置限制，建议：

1. **预设参考音频**: 在服务器上准备多个参考音频文件
2. **使用不同的 API 端点**: 为不同的音色创建不同的配置
3. **配置文件示例**:
   ```json
   {
     "name": "VoxCPM - 女声1",
     "url": "http://your-server:8000/tts",
     "method": "POST",
     "headers": {
       "Content-Type": "application/json"
     },
     "body": "{\"text\":\"{content}\",\"prompt_wav_path\":\"/data/voices/female1.wav\",\"prompt_text\":\"参考音频文本\",\"cfg_value\":2.0}",
     "contentType": "audio/wav"
   }
   ```

## ⚙️ 参数说明

### cfg_value（CFG 值）
- **范围**: 1.0 - 3.0
- **默认**: 2.0
- **说明**: 
  - 较低值：允许更多创造性，但可能不够贴合提示
  - 较高值：更好地贴合提示，但可能过于生硬
  - 如果提示语音听起来不自然或过于夸张，可以调低
  - 如果极短文本输入出现稳定性问题，可以调高

### inference_timesteps（推理时间步）
- **范围**: 4 - 30
- **默认**: 10
- **说明**:
  - 较低值：合成速度更快
  - 较高值：合成质量更佳

### normalize（文本正则化）
- **默认**: false
- **说明**:
  - `true`: 使用 WeTextProcessing 组件进行文本正则化
  - `false`: 使用 VoxCPM 内置的文本理解能力（支持音素输入等）

### denoise（音频降噪）
- **默认**: false
- **说明**:
  - `true`: 使用 ZipEnhancer 消除背景噪音，但会将采样率限制在 16kHz
  - `false`: 保留原始音频的全部信息，支持最高 44.1kHz 采样率

## 🔍 API 文档

启动服务后，可以访问以下地址查看交互式 API 文档：

- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

## ⚠️ 注意事项

1. **首次运行**: 首次运行会自动下载模型文件（约 1.5GB），请确保网络连接正常
2. **CUDA 加速**: 如果系统有 NVIDIA GPU，会自动使用 CUDA 加速
3. **内存要求**: 建议至少 4GB 可用内存
4. **并发限制**: 当前实现为单线程处理，避免 CUDA graph 多线程问题
5. **网络访问**: 如果需要在局域网内访问，请使用 `0.0.0.0` 作为主机地址

## 🐛 故障排除

### 问题：模型加载失败
- **解决方案**: 检查网络连接，确保可以访问 HuggingFace
- **备选方案**: 手动下载模型到 `./models/openbmb__VoxCPM1.5` 目录

### 问题：端口被占用
- **解决方案**: 使用 `--port` 参数指定其他端口

### 问题：CUDA 内存不足
- **解决方案**: 降低 `inference_timesteps` 值，或使用 CPU 模式

### 问题：Legado 无法连接
- **解决方案**: 
  1. 检查防火墙设置
  2. 确保使用正确的 IP 地址和端口
  3. 如果使用手机，确保手机和服务器在同一网络

## 📚 相关资源

- [VoxCPM 项目主页](https://github.com/OpenBMB/VoxCPM)
- [Legado 阅读器](https://github.com/gedoor/legado)
- [FastAPI 文档](https://fastapi.tiangolo.com/)
