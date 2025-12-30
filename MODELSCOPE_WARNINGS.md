# ModelScope 警告信息说明

## 警告信息

在加载 ZipEnhancer（降噪模型）时，可能会看到以下警告信息：

```
2025-12-30 15:08:02,470 - modelscope - WARNING - No preprocessor field found in cfg.
2025-12-30 15:08:02,470 - modelscope - WARNING - No val key and type key found in preprocessor domain of configuration.json file.
2025-12-30 15:08:02,470 - modelscope - WARNING - Cannot find available config to build preprocessor at mode inference, current config: {'model_dir': '/mnt/github/VoxCPM/models_cache/modelscope/hub/models/iic/speech_zipenhancer_ans_multiloss_16k_base'}. trying to build by task and model information.
2025-12-30 15:08:02,470 - modelscope - INFO - No preprocessor key ('speech_zipenhancer_ans_multiloss_16k_base', 'acoustic-noise-suppression') found in PREPROCESSOR_MAP, skip building preprocessor. If the pipeline runs normally, please ignore this log.
```

## 警告含义

这些警告表示：

1. **No preprocessor field found in cfg**：模型配置文件中没有找到 preprocessor（预处理器）字段
2. **Cannot find available config**：无法找到可用的预处理器配置
3. **skip building preprocessor**：跳过预处理器构建

## 是否影响功能？

**✅ 不影响功能！**

- 这些警告是 ModelScope 库在尝试自动构建预处理器时产生的
- ZipEnhancer 模型本身可以正常工作，不需要额外的预处理器
- 最后一行明确说明：**"If the pipeline runs normally, please ignore this log."**
- 降噪功能完全正常，可以安全忽略这些警告

## 如何抑制警告（可选）

如果这些警告信息影响阅读，可以通过以下方式抑制：

### 方法 1：设置日志级别（推荐）

在代码中设置 ModelScope 的日志级别：

```python
import logging
logging.getLogger('modelscope').setLevel(logging.ERROR)  # 只显示 ERROR 及以上级别
```

### 方法 2：修改 zipenhancer.py

在 `src/voxcpm/zipenhancer.py` 文件开头添加：

```python
import logging
logging.getLogger('modelscope').setLevel(logging.ERROR)
```

### 方法 3：环境变量

设置环境变量：

```bash
export MODELSCOPE_LOG_LEVEL=ERROR
```

### 方法 4：在 app.py 中设置

在 `app.py` 文件开头添加：

```python
import logging
logging.getLogger('modelscope').setLevel(logging.ERROR)
```

## 验证功能正常

可以通过以下方式验证降噪功能是否正常：

```python
from voxcpm import VoxCPM

model = VoxCPM.from_pretrained("openbmb/VoxCPM1.5")

# 使用降噪功能
wav = model.generate(
    text="测试文本",
    prompt_wav_path="reference.wav",
    prompt_text="参考文本",
    denoise=True,  # 启用降噪
)
```

如果降噪功能正常工作，说明这些警告可以安全忽略。

## 总结

- ✅ **功能正常**：这些警告不影响 ZipEnhancer 的功能
- ✅ **可以忽略**：按照日志提示，可以安全忽略这些警告
- ⚙️ **可选抑制**：如果需要，可以通过设置日志级别来抑制这些警告
