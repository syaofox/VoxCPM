"""
VoxCPM API Server for Legado Reader
为 Legado 阅读器提供自定义朗读引擎的 HTTP API 服务
"""
import os
import io
import json
import logging
import tempfile
from pathlib import Path
from typing import Optional
import numpy as np
import torch
import soundfile as sf
from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Form
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel
import uvicorn

# 抑制 ModelScope 的警告信息
logging.getLogger('modelscope').setLevel(logging.ERROR)

# 设置模型缓存目录到项目目录
PROJECT_ROOT = Path(__file__).parent.absolute()
MODELS_CACHE_DIR = PROJECT_ROOT / "models_cache"

# 设置 HuggingFace 缓存目录
os.environ["HF_HOME"] = str(MODELS_CACHE_DIR / "huggingface")
os.environ["HF_HUB_CACHE"] = str(MODELS_CACHE_DIR / "huggingface" / "hub")

# 设置 ModelScope 缓存目录
os.environ["MODELSCOPE_CACHE"] = str(MODELS_CACHE_DIR / "modelscope" / "hub")

os.environ["TOKENIZERS_PARALLELISM"] = "false"
if os.environ.get("HF_REPO_ID", "").strip() == "":
    os.environ["HF_REPO_ID"] = "openbmb/VoxCPM1.5"

import voxcpm

# 创建 FastAPI 应用
app = FastAPI(title="VoxCPM TTS API", description="VoxCPM Text-to-Speech API for Legado Reader")

# 全局模型实例
voxcpm_model: Optional[voxcpm.VoxCPM] = None


class TTSRequest(BaseModel):
    """TTS 请求模型（JSON格式）"""
    text: str
    voice: Optional[str] = None  # 预留，用于未来支持多音色
    speed: Optional[float] = 1.0  # 预留，用于未来支持语速控制
    pitch: Optional[float] = 1.0  # 预留，用于未来支持音调控制
    cfg_value: Optional[float] = 2.0  # CFG 值
    inference_timesteps: Optional[int] = 10  # 推理时间步
    normalize: Optional[bool] = False  # 文本正则化
    denoise: Optional[bool] = False  # 音频降噪
    prompt_wav_path: Optional[str] = None  # 参考音频文件路径（服务器本地路径）
    prompt_text: Optional[str] = None  # 参考音频对应的文本


def _resolve_model_dir() -> str:
    """
    解析模型目录:
    1) 如果本地检查点目录存在，使用本地目录
    2) 如果设置了 HF_REPO_ID 环境变量，下载到 models/{repo}
    3) 否则回退到 'models'
    """
    default_local_model_dir = "./models/openbmb__VoxCPM1.5"
    if os.path.isdir(default_local_model_dir):
        return default_local_model_dir
    
    repo_id = os.environ.get("HF_REPO_ID", "").strip()
    if len(repo_id) > 0:
        target_dir = os.path.join("models", repo_id.replace("/", "__"))
        if not os.path.isdir(target_dir):
            try:
                from huggingface_hub import snapshot_download
                os.makedirs(target_dir, exist_ok=True)
                print(f"正在从 HuggingFace 下载模型 '{repo_id}' 到 '{target_dir}' ...")
                snapshot_download(repo_id=repo_id, local_dir=target_dir, local_dir_use_symlinks=False)
            except Exception as e:
                print(f"警告: HuggingFace 下载失败: {e}. 回退到缓存目录。")
                return None  # 返回 None 表示需要使用 from_pretrained
        return target_dir
    return None


def get_or_load_model() -> voxcpm.VoxCPM:
    """获取或加载 VoxCPM 模型"""
    global voxcpm_model
    if voxcpm_model is not None:
        return voxcpm_model
    
    print("正在加载 VoxCPM 模型...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 运行设备: {device}")
    
    # 解析模型目录
    model_dir = _resolve_model_dir()
    
    # 如果本地目录不存在，使用 from_pretrained 下载到项目缓存目录
    if model_dir is None or not os.path.isdir(model_dir):
        print("正在下载模型到项目缓存目录...")
        repo_id = os.environ.get("HF_REPO_ID", "openbmb/VoxCPM1.5")
        cache_dir = str(MODELS_CACHE_DIR / "huggingface" / "hub")
        voxcpm_model = voxcpm.VoxCPM.from_pretrained(
            hf_model_id=repo_id,
            cache_dir=cache_dir,
            optimize=False,  # 禁用优化以避免多线程问题
        )
    else:
        print(f"使用模型目录: {model_dir}")
        voxcpm_model = voxcpm.VoxCPM(
            voxcpm_model_path=model_dir,
            optimize=False,  # 禁用优化以避免多线程问题
        )
    
    print("模型加载成功！")
    return voxcpm_model


@app.on_event("startup")
async def startup_event():
    """应用启动时预加载模型"""
    print("正在初始化 VoxCPM API 服务...")
    get_or_load_model()
    print("API 服务已就绪！")


@app.get("/")
async def root():
    """根路径，返回 API 信息"""
    return {
        "name": "VoxCPM TTS API",
        "version": "1.0.0",
        "description": "VoxCPM Text-to-Speech API for Legado Reader",
        "endpoints": {
            "/tts": "POST - 文本转语音（JSON格式，支持参考音频路径）。添加 ?stream=true 启用流式输出",
            "/tts/upload": "POST - 文本转语音（multipart/form-data，支持上传参考音频）",
            "/tts/stream": "POST - 流式文本转语音（PCM格式，适合长文本）",
            "/health": "GET - 健康检查"
        }
    }


@app.get("/health")
async def health_check():
    """健康检查端点"""
    model = get_or_load_model()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": device,
        "sample_rate": model.tts_model.sample_rate if model else None
    }


@app.get("/tts")
async def text_to_speech_get():
    """
    GET 端点用于测试连接
    Legado 可能会先发送 GET 请求测试连接
    """
    return {
        "status": "ok",
        "message": "VoxCPM TTS API is running. Please use POST method for TTS requests.",
        "endpoint": "/tts",
        "method": "POST",
        "content_type": "application/json"
    }


@app.post("/tts")
async def text_to_speech(request: Request):
    """
    文本转语音端点（JSON格式）
    支持 Legado 阅读器的 httpTTS API 格式
    支持声音克隆：通过 prompt_wav_path 和 prompt_text 参数
    支持流式输出：通过查询参数 ?stream=true
    """
    try:
        # 检查是否启用流式输出
        stream_mode = request.query_params.get("stream", "false").lower() == "true"
        
        # 记录请求信息用于调试
        print(f"\n{'='*60}")
        print(f"收到 POST 请求: {request.url}")
        print(f"请求方法: {request.method}")
        print(f"流式模式: {stream_mode}")
        print(f"请求头: {dict(request.headers)}")
        
        # 读取请求体
        try:
            body_bytes = await request.body()
            body_str = body_bytes.decode('utf-8')
            print(f"原始请求体: {body_str[:500]}")
            
            # 解析 JSON
            body_json = json.loads(body_str)
            print(f"解析的 JSON: {body_json}")
            
            # 创建请求对象
            tts_request = TTSRequest(**body_json)
        except json.JSONDecodeError as e:
            print(f"JSON 解析错误: {e}")
            print(f"请求体内容: {body_str if 'body_str' in locals() else '无法读取'}")
            raise HTTPException(status_code=400, detail=f"JSON 格式错误: {str(e)}")
        except Exception as e:
            print(f"解析请求体失败: {e}")
            import traceback
            traceback.print_exc()
            raise HTTPException(status_code=400, detail=f"请求体格式错误: {str(e)}")
        
        # 获取模型
        model = get_or_load_model()
        
        # 验证文本
        text = (tts_request.text or "").strip()
        
        # 兼容性处理：如果收到的是占位符字面量，尝试从请求中获取
        if text in ["content", "{content}", "{{speakText}}", "{{content}}"]:
            print(f"⚠️  检测到占位符字面量: {text}")
            # 尝试从请求头或其他地方获取实际文本
            # 注意：这可能需要 Legado 的特殊支持
            raise HTTPException(
                status_code=400, 
                detail=f"占位符未被替换，收到: {text}。请检查 Legado 配置中的占位符格式。对于 JSON body，应使用 {{speakText}} 或 {{content}}"
            )
        
        if len(text) == 0:
            raise HTTPException(status_code=400, detail="文本内容不能为空")
        
        # 验证参考音频参数
        prompt_wav_path = tts_request.prompt_wav_path
        prompt_text = tts_request.prompt_text
        
        if (prompt_wav_path is not None) != (prompt_text is not None):
            raise HTTPException(
                status_code=400, 
                detail="参考音频和参考文本必须同时提供或同时为空"
            )
        
        # 如果提供了参考音频路径，验证文件是否存在
        if prompt_wav_path:
            if not os.path.exists(prompt_wav_path):
                raise HTTPException(
                    status_code=400,
                    detail=f"参考音频文件不存在: {prompt_wav_path}"
                )
            print(f"使用参考音频进行声音克隆: {prompt_wav_path}")
            print(f"参考文本: '{prompt_text[:60]}...'")
        else:
            print(f"标准语音合成，文本: '{text[:60]}...'")
        
        # 获取采样率
        sample_rate = model.tts_model.sample_rate
        
        # 流式输出模式
        if stream_mode:
            def generate_audio_stream():
                try:
                    for chunk in model.generate_streaming(
                        text=text,
                        prompt_text=prompt_text,
                        prompt_wav_path=prompt_wav_path,
                        cfg_value=tts_request.cfg_value or 2.0,
                        inference_timesteps=tts_request.inference_timesteps or 10,
                        normalize=tts_request.normalize or False,
                        denoise=tts_request.denoise or False,
                    ):
                        # 将音频块转换为 16-bit PCM 格式
                        chunk_int16 = (chunk * 32767).astype(np.int16)
                        chunk_bytes = chunk_int16.tobytes()
                        yield chunk_bytes
                except Exception as e:
                    logging.error(f"流式生成过程中出错: {str(e)}", exc_info=True)
                    raise
            
            return StreamingResponse(
                generate_audio_stream(),
                media_type="audio/pcm",
                headers={
                    "Content-Type": f"audio/pcm; rate={sample_rate}; channels=1; encoding=pcm_s16le",
                    "X-Sample-Rate": str(sample_rate),
                    "X-Channels": "1",
                    "X-Encoding": "pcm_s16le",
                }
            )
        
        # 非流式输出模式（默认）
        # 生成语音
        wav = model.generate(
            text=text,
            prompt_text=prompt_text,
            prompt_wav_path=prompt_wav_path,
            cfg_value=tts_request.cfg_value or 2.0,
            inference_timesteps=tts_request.inference_timesteps or 10,
            normalize=tts_request.normalize or False,
            denoise=tts_request.denoise or False,
        )
        
        # 将音频数据转换为 WAV 格式的字节流
        buffer = io.BytesIO()
        sf.write(buffer, wav, sample_rate, format='WAV')
        buffer.seek(0)
        
        # 返回音频文件
        return Response(
            content=buffer.read(),
            media_type="audio/wav",
            headers={
                "Content-Disposition": "attachment; filename=tts_output.wav"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"生成语音时出错: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"生成语音失败: {str(e)}")


@app.post("/tts/upload")
async def text_to_speech_with_upload(
    text: str = Form(...),
    prompt_audio: Optional[UploadFile] = File(None),
    prompt_text: Optional[str] = Form(None),
    cfg_value: float = Form(2.0),
    inference_timesteps: int = Form(10),
    normalize: bool = Form(False),
    denoise: bool = Form(False),
):
    """
    文本转语音端点（支持上传参考音频文件）
    使用 multipart/form-data 格式
    适合需要上传参考音频进行声音克隆的场景
    """
    try:
        # 获取模型
        model = get_or_load_model()
        
        # 验证文本
        text = (text or "").strip()
        if len(text) == 0:
            raise HTTPException(status_code=400, detail="文本内容不能为空")
        
        # 处理参考音频
        prompt_wav_path = None
        temp_file = None
        
        if prompt_audio is not None:
            # 验证参考文本
            if not prompt_text:
                raise HTTPException(
                    status_code=400,
                    detail="上传参考音频时必须提供对应的参考文本 (prompt_text)"
                )
            
            # 保存上传的音频到临时文件
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
            try:
                content = await prompt_audio.read()
                temp_file.write(content)
                temp_file.close()
                prompt_wav_path = temp_file.name
                
                print(f"使用上传的参考音频进行声音克隆: {prompt_wav_path}")
                print(f"参考文本: '{prompt_text[:60]}...'")
            except Exception as e:
                if temp_file:
                    os.unlink(temp_file.name)
                raise HTTPException(status_code=400, detail=f"读取音频文件失败: {str(e)}")
        else:
            print(f"标准语音合成，文本: '{text[:60]}...'")
        
        try:
            # 生成语音
            wav = model.generate(
                text=text,
                prompt_text=prompt_text if prompt_audio else None,
                prompt_wav_path=prompt_wav_path,
                cfg_value=cfg_value,
                inference_timesteps=inference_timesteps,
                normalize=normalize,
                denoise=denoise,
            )
            
            # 获取采样率
            sample_rate = model.tts_model.sample_rate
            
            # 将音频数据转换为 WAV 格式的字节流
            buffer = io.BytesIO()
            sf.write(buffer, wav, sample_rate, format='WAV')
            buffer.seek(0)
            
            return Response(
                content=buffer.read(),
                media_type="audio/wav",
                headers={
                    "Content-Disposition": "attachment; filename=tts_output.wav"
                }
            )
        finally:
            # 清理临时文件
            if temp_file and os.path.exists(temp_file.name):
                os.unlink(temp_file.name)
        
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"生成语音时出错: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"生成语音失败: {str(e)}")


@app.post("/tts/stream")
async def text_to_speech_stream(request: TTSRequest):
    """
    流式文本转语音端点
    支持流式生成，适合长文本
    返回格式：PCM 音频流（16-bit, 单声道）
    """
    try:
        # 获取模型
        model = get_or_load_model()
        
        # 验证文本
        text = (request.text or "").strip()
        if len(text) == 0:
            raise HTTPException(status_code=400, detail="文本内容不能为空")
        
        # 验证参考音频参数
        prompt_wav_path = request.prompt_wav_path
        prompt_text = request.prompt_text
        
        if (prompt_wav_path is not None) != (prompt_text is not None):
            raise HTTPException(
                status_code=400, 
                detail="参考音频和参考文本必须同时提供或同时为空"
            )
        
        if prompt_wav_path and not os.path.exists(prompt_wav_path):
            raise HTTPException(
                status_code=400,
                detail=f"参考音频文件不存在: {prompt_wav_path}"
            )
        
        print(f"正在流式生成语音，文本: '{text[:60]}...'")
        
        # 获取采样率
        sample_rate = model.tts_model.sample_rate
        
        # 流式生成语音
        def generate_audio_stream():
            try:
                # 首先发送采样率信息（作为 JSON 元数据，可选）
                # 或者直接开始发送音频数据
                
                # 使用流式生成
                for chunk in model.generate_streaming(
                    text=text,
                    prompt_text=prompt_text,
                    prompt_wav_path=prompt_wav_path,
                    cfg_value=request.cfg_value or 2.0,
                    inference_timesteps=request.inference_timesteps or 10,
                    normalize=request.normalize or False,
                    denoise=request.denoise or False,
                ):
                    # 将音频块转换为 16-bit PCM 格式
                    # chunk 是 float32 格式，范围通常在 [-1, 1]
                    chunk_int16 = (chunk * 32767).astype(np.int16)
                    # 转换为字节流
                    chunk_bytes = chunk_int16.tobytes()
                    yield chunk_bytes
                    
            except Exception as e:
                logging.error(f"流式生成过程中出错: {str(e)}", exc_info=True)
                # 注意：一旦开始流式传输，无法发送 HTTP 错误响应
                # 可以考虑发送错误标记或记录日志
                raise
        
        return StreamingResponse(
            generate_audio_stream(),
            media_type="audio/pcm",
            headers={
                "Content-Type": f"audio/pcm; rate={sample_rate}; channels=1; encoding=pcm_s16le",
                "X-Sample-Rate": str(sample_rate),
                "X-Channels": "1",
                "X-Encoding": "pcm_s16le",
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"流式生成语音时出错: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"流式生成语音失败: {str(e)}")


def run_server(host: str = "0.0.0.0", port: int = 8000):
    """运行 API 服务器"""
    print(f"正在启动 VoxCPM API 服务器...")
    print(f"服务地址: http://{host}:{port}")
    print(f"API 文档: http://{host}:{port}/docs")
    print(f"健康检查: http://{host}:{port}/health")
    print(f"TTS 端点: http://{host}:{port}/tts")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="VoxCPM TTS API Server for Legado Reader")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="服务器主机地址")
    parser.add_argument("--port", type=int, default=8000, help="服务器端口")
    args = parser.parse_args()
    
    run_server(host=args.host, port=args.port)
