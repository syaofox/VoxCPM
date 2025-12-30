# 模型缓存配置说明

## 概述

项目已配置为将模型缓存保存在项目目录下的 `models_cache/` 文件夹中，而不是默认的用户缓存目录（`~/.cache/`）。

## 缓存目录结构

```
VoxCPM/
├── models_cache/              # 项目模型缓存目录（自动创建）
│   ├── huggingface/          # HuggingFace 模型缓存
│   │   └── hub/              # HuggingFace Hub 缓存
│   │       └── models--openbmb--VoxCPM1.5/
│   └── modelscope/           # ModelScope 模型缓存
│       └── hub/              # ModelScope Hub 缓存
│           └── models/
│               └── iic/
│                   ├── SenseVoiceSmall/
│                   └── speech_zipenhancer_ans_multiloss_16k_base/
```

## 配置说明

### 1. HuggingFace 模型缓存

- **环境变量**：`HF_HUB_CACHE`
- **默认路径**：`项目目录/models_cache/huggingface/hub/`
- **模型**：VoxCPM 1.5（`openbmb/VoxCPM1.5`）

### 2. ModelScope 模型缓存

- **环境变量**：`MODELSCOPE_CACHE`
- **默认路径**：`项目目录/models_cache/modelscope/hub/`
- **模型**：
  - ZipEnhancer（降噪模型）
  - SenseVoiceSmall（ASR 模型）

## 使用方法

### 自动使用项目缓存（推荐）

无需任何配置，代码会自动使用项目目录下的缓存：

```bash
# CLI
uv run voxcpm --text "Hello" --output out.wav

# Web 界面
uv run python app.py

# Python API
from voxcpm import VoxCPM
model = VoxCPM.from_pretrained("openbmb/VoxCPM1.5")
```

### 自定义缓存目录

如果需要使用其他缓存目录，可以通过环境变量设置：

```bash
# 设置 HuggingFace 缓存
export HF_HUB_CACHE=/path/to/your/cache/huggingface/hub

# 设置 ModelScope 缓存
export MODELSCOPE_CACHE=/path/to/your/cache/modelscope/hub

# 运行程序
uv run python app.py
```

或者在代码中指定：

```python
from voxcpm import VoxCPM

model = VoxCPM.from_pretrained(
    "openbmb/VoxCPM1.5",
    cache_dir="/path/to/your/cache/huggingface/hub"
)
```

## 迁移现有缓存

如果之前已经下载了模型到 `~/.cache/`，可以迁移到项目目录：

```bash
# 创建缓存目录
mkdir -p models_cache/huggingface/hub
mkdir -p models_cache/modelscope/hub

# 迁移 HuggingFace 模型（可选）
cp -r ~/.cache/huggingface/hub/models--openbmb--VoxCPM1.5 \
      models_cache/huggingface/hub/ 2>/dev/null || echo "模型不存在，将自动下载"

# 迁移 ModelScope 模型（可选）
cp -r ~/.cache/modelscope/hub/models/iic \
      models_cache/modelscope/hub/models/ 2>/dev/null || echo "模型不存在，将自动下载"
```

## 注意事项

1. **首次运行**：首次运行时会自动下载模型到 `models_cache/` 目录（约 1.9GB）
2. **Git 忽略**：`models_cache/` 目录已添加到 `.gitignore`，不会被提交到 Git
3. **磁盘空间**：确保项目目录有足够的磁盘空间（至少 2GB）
4. **权限**：确保对项目目录有写入权限

## 查看缓存大小

```bash
# 查看项目缓存目录大小
du -sh models_cache/

# 查看各个模型大小
du -sh models_cache/huggingface/hub/*
du -sh models_cache/modelscope/hub/models/iic/*
```

## 清理缓存

如果需要清理缓存：

```bash
# 删除整个缓存目录
rm -rf models_cache/

# 或只删除特定模型
rm -rf models_cache/huggingface/hub/models--openbmb--VoxCPM1.5
```
