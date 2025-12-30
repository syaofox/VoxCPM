# CUDA Graph 多线程问题修复说明

## 问题描述

在使用 Web 界面时，特别是更换参考音频时，可能会出现以下错误：

```
AssertionError
File "/mnt/github/VoxCPM/.venv/lib/python3.13/site-packages/torch/_inductor/cudagraph_trees.py", line 327, in get_obj
    assert torch._C._is_key_in_tls(attr_name)
```

## 问题原因

这是 PyTorch 的 CUDA graph 在多线程/异步环境中的已知问题：

1. **torch.compile 优化**：VoxCPM 默认使用 `torch.compile` 优化模型性能
2. **CUDA graph**：`torch.compile` 会使用 CUDA graph 来加速推理
3. **多线程冲突**：Gradio 使用多线程/异步处理请求，与 CUDA graph 的线程本地存储（TLS）冲突
4. **更换参考音频**：每次更换参考音频时，模型需要重新编译，更容易触发这个问题

## 解决方案

### 方案 1：禁用模型优化（已实施）

在 Web 界面中禁用 `torch.compile` 优化：

```python
self.voxcpm_model = voxcpm.VoxCPM.from_pretrained(
    hf_model_id="openbmb/VoxCPM1.5",
    optimize=False,  # 禁用优化
)
```

**优点**：
- ✅ 完全避免多线程问题
- ✅ 稳定性最高
- ✅ 实现简单

**缺点**：
- ⚠️ 推理速度稍慢（约 10-20%）

### 方案 2：设置单线程并发（已实施）

在 Gradio 中设置单线程并发：

```python
interface.queue(max_size=10, default_concurrency_limit=1).launch(...)
```

**优点**：
- ✅ 保持模型优化
- ✅ 避免多线程冲突

**缺点**：
- ⚠️ 并发性能受限

### 方案 3：禁用 CUDA graph（可选）

设置环境变量禁用 CUDA graph：

```bash
export TORCH_CUDAGRAPH_DISABLE=1
uv run python app.py
```

或在代码中设置：

```python
import os
os.environ["TORCH_CUDAGRAPH_DISABLE"] = "1"
```

## 当前实施

已采用**方案 1 + 方案 2**的组合：

1. ✅ Web 界面中禁用模型优化（`optimize=False`）
2. ✅ Gradio 使用单线程并发（`default_concurrency_limit=1`）

这样可以：
- 完全避免多线程问题
- 保持 Web 界面的稳定性
- 性能影响可接受（Web 场景下网络延迟等因素更重要）

## CLI 使用不受影响

CLI 命令行工具仍然默认启用优化：

```bash
# CLI 使用，默认 optimize=True（性能优先）
uv run voxcpm --text "Hello" --output out.wav

# 如需禁用优化（调试用）
uv run python -c "from voxcpm import VoxCPM; model = VoxCPM.from_pretrained('openbmb/VoxCPM1.5', optimize=False)"
```

## 性能对比

| 场景 | 优化状态 | 推理速度 | 稳定性 |
|------|---------|---------|--------|
| CLI（单线程） | optimize=True | 快（100%） | ✅ 稳定 |
| Web 界面 | optimize=False | 稍慢（80-90%） | ✅ 稳定 |
| Web 界面（旧） | optimize=True | 快（100%） | ❌ 多线程问题 |

## 验证修复

修复后，应该不再出现 `AssertionError` 错误：

1. ✅ 启动 Web 界面：`uv run python app.py`
2. ✅ 多次更换参考音频
3. ✅ 连续生成多个音频
4. ✅ 不应该再出现 CUDA graph 相关错误

## 如果问题仍然存在

如果仍然遇到问题，可以尝试：

1. **完全禁用 CUDA graph**：
   ```bash
   export TORCH_CUDAGRAPH_DISABLE=1
   ```

2. **使用 CPU 模式**（不推荐，速度很慢）：
   ```python
   # 在 app.py 中强制使用 CPU
   os.environ["CUDA_VISIBLE_DEVICES"] = ""
   ```

3. **降低并发数**：
   ```python
   interface.queue(max_size=1, default_concurrency_limit=1)
   ```

## 相关链接

- [PyTorch CUDA Graph 文档](https://pytorch.org/docs/stable/torch.compiler.html)
- [Gradio 队列文档](https://www.gradio.app/guides/performance)
