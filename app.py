import os
import numpy as np
import torch
import gradio as gr  
import spaces
from typing import Optional, Tuple, List, Dict
from funasr import AutoModel
from pathlib import Path
import logging
import hashlib
import tempfile

# 抑制 ModelScope 的警告信息（不影响功能）
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


def split_text_into_sentences(text: str, max_length: int = 200) -> List[str]:
    """
    将长文本分割成句子，用于避免显存溢出
    
    分割策略：
    1. 优先按换行符分割
    2. 对于每一行，如果超过 max_length 字符，再按照句号、叹号、问号等标点符号分割
    
    Args:
        text: 输入文本
        max_length: 单个句子的最大长度（字符数），超过此长度会在标点处分割
    
    Returns:
        句子列表
    """
    if not text.strip():
        return []
    
    # 句末标点符号（用于句子分割）
    sentence_endings = ['。', '？', '！', '.', '?', '!']
    
    # 第一步：按换行符分割
    lines = text.split('\n')
    sentences = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # 如果这一行不超过 max_length，直接添加
        if len(line) <= max_length:
            sentences.append(line)
            continue
        
        # 如果这一行超过 max_length，按句末标点符号分割
        current_sentence = ""
        i = 0
        
        while i < len(line):
            char = line[i]
            current_sentence += char
            
            # 如果遇到句末标点符号，结束当前句子
            if char in sentence_endings:
                # 检查下一个字符是否是引号
                if i + 1 < len(line) and line[i + 1] in ['"', '"', '"', '"', ''', ''']:
                    current_sentence += line[i + 1]
                    i += 2
                else:
                    i += 1
                
                if current_sentence.strip():
                    sentences.append(current_sentence.strip())
                current_sentence = ""
            # 如果当前句子太长，尝试在逗号或分号处分割
            elif len(current_sentence) >= max_length:
                # 尝试在逗号、分号或空格处分割
                last_comma_zh = current_sentence.rfind('，')
                last_comma_en = current_sentence.rfind(',')
                last_semicolon_zh = current_sentence.rfind('；')
                last_semicolon_en = current_sentence.rfind(';')
                last_space = current_sentence.rfind(' ')
                
                split_pos = max(last_comma_zh, last_comma_en, last_semicolon_zh, 
                               last_semicolon_en, last_space)
                
                if split_pos > max_length * 0.5:  # 如果找到合适的分割点
                    sentences.append(current_sentence[:split_pos + 1].strip())
                    current_sentence = current_sentence[split_pos + 1:]
                else:
                    # 没有合适的分割点，直接按长度分割
                    sentences.append(current_sentence.strip())
                    current_sentence = ""
                i += 1
            else:
                i += 1
        
        # 添加剩余的文本
        if current_sentence.strip():
            sentences.append(current_sentence.strip())
    
    # 过滤空句子
    sentences = [s for s in sentences if s.strip()]
    
    # 如果没有找到任何句子（比如没有换行也没有标点），按长度分割
    if not sentences:
        for i in range(0, len(text), max_length):
            chunk = text[i:i + max_length].strip()
            if chunk:
                sentences.append(chunk)
    
    return sentences if sentences else [text]


class VoxCPMDemo:
    def __init__(self) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🚀 Running on device: {self.device}")

        # ASR model for prompt text recognition
        self.asr_model_id = "iic/SenseVoiceSmall"
        self.asr_model: Optional[AutoModel] = AutoModel(
            model=self.asr_model_id,
            disable_update=True,
            log_level='DEBUG',
            device="cuda:0" if self.device == "cuda" else "cpu",
        )

        # TTS model (lazy init)
        self.voxcpm_model: Optional[voxcpm.VoxCPM] = None
        self.default_local_model_dir = "./models/VoxCPM1.5"
        
        # Prompt cache 缓存：key 为 (prompt_wav_path, prompt_text) 的哈希，value 为 prompt_cache
        self.prompt_cache_dict: Dict[str, dict] = {}

    # ---------- Model helpers ----------
    def _resolve_model_dir(self) -> str:
        """
        Resolve model directory:
        1) Use local checkpoint directory if exists
        2) If HF_REPO_ID env is set, download into models/{repo}
        3) Fallback to 'models'
        """
        if os.path.isdir(self.default_local_model_dir):
            return self.default_local_model_dir

        repo_id = os.environ.get("HF_REPO_ID", "").strip()
        if len(repo_id) > 0:
            target_dir = os.path.join("models", repo_id.replace("/", "__"))
            if not os.path.isdir(target_dir):
                try:
                    from huggingface_hub import snapshot_download  # type: ignore
                    os.makedirs(target_dir, exist_ok=True)
                    print(f"Downloading model from HF repo '{repo_id}' to '{target_dir}' ...")
                    snapshot_download(repo_id=repo_id, local_dir=target_dir, local_dir_use_symlinks=False)
                except Exception as e:
                    print(f"Warning: HF download failed: {e}. Falling back to 'data'.")
                    return "models"
            return target_dir
        return "models"

    def get_or_load_voxcpm(self) -> voxcpm.VoxCPM:
        if self.voxcpm_model is not None:
            return self.voxcpm_model
        print("Model not loaded, initializing...")
        model_dir = self._resolve_model_dir()
        print(f"Using model dir: {model_dir}")
        # 如果本地目录不存在，使用 from_pretrained 下载到项目缓存目录
        if not os.path.isdir(model_dir):
            print(f"Downloading model to project cache directory...")
            cache_dir = str(MODELS_CACHE_DIR / "huggingface" / "hub")
            self.voxcpm_model = voxcpm.VoxCPM.from_pretrained(
                hf_model_id=os.environ.get("HF_REPO_ID", "openbmb/VoxCPM1.5"),
                cache_dir=cache_dir,
                optimize=False,  # 在 Web 界面中禁用优化以避免多线程问题
            )
        else:
            self.voxcpm_model = voxcpm.VoxCPM(
                voxcpm_model_path=model_dir,
                optimize=False,  # 在 Web 界面中禁用优化以避免多线程问题
            )
        print("Model loaded successfully.")
        return self.voxcpm_model

    # ---------- Functional endpoints ----------
    def prompt_wav_recognition(self, prompt_wav: Optional[str]) -> str:
        if prompt_wav is None:
            return ""
        res = self.asr_model.generate(input=prompt_wav, language="auto", use_itn=True)
        text = res[0]["text"].split('|>')[-1]
        return text
    
    def _get_prompt_cache_key(self, prompt_wav_path: Optional[str], prompt_text: Optional[str]) -> Optional[str]:
        """生成 prompt cache 的键"""
        if prompt_wav_path is None or prompt_text is None:
            return None
        # 使用文件路径和文本内容的哈希作为键
        key_str = f"{prompt_wav_path}:{prompt_text}"
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _get_or_build_prompt_cache(
        self, 
        prompt_wav_path: Optional[str], 
        prompt_text: Optional[str],
        denoise: bool = False
    ) -> Optional[dict]:
        """获取或构建 prompt cache"""
        if prompt_wav_path is None or prompt_text is None:
            return None
        
        cache_key = self._get_prompt_cache_key(prompt_wav_path, prompt_text)
        if cache_key is None:
            return None
        
        # 如果缓存中存在，直接返回
        if cache_key in self.prompt_cache_dict:
            print(f"使用缓存的 prompt cache (key: {cache_key[:8]}...)")
            return self.prompt_cache_dict[cache_key]
        
        # 构建新的 prompt cache
        print(f"构建新的 prompt cache (key: {cache_key[:8]}...)")
        current_model = self.get_or_load_voxcpm()
        
        # 如果需要降噪，先处理音频
        actual_prompt_wav_path = prompt_wav_path
        temp_prompt_wav_path = None
        
        try:
            if denoise and current_model.denoiser is not None:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                    temp_prompt_wav_path = tmp_file.name
                current_model.denoiser.enhance(prompt_wav_path, output_path=temp_prompt_wav_path)
                actual_prompt_wav_path = temp_prompt_wav_path
            
            # 构建 prompt cache
            prompt_cache = current_model.tts_model.build_prompt_cache(
                prompt_wav_path=actual_prompt_wav_path,
                prompt_text=prompt_text
            )
            
            # 缓存起来
            self.prompt_cache_dict[cache_key] = prompt_cache
            print(f"Prompt cache 已缓存 (key: {cache_key[:8]}...)")
            
            return prompt_cache
            
        finally:
            # 清理临时文件
            if temp_prompt_wav_path and os.path.exists(temp_prompt_wav_path):
                try:
                    os.unlink(temp_prompt_wav_path)
                except OSError:
                    pass

    def generate_tts_audio(
        self,
        text_input: str,
        prompt_wav_path_input: Optional[str] = None,
        prompt_text_input: Optional[str] = None,
        cfg_value_input: float = 2.0,
        inference_timesteps_input: int = 10,
        do_normalize: bool = True,
        denoise: bool = True,
    ) -> Tuple[int, np.ndarray]:
        """
        Generate speech from text using VoxCPM; optional reference audio for voice style guidance.
        Returns (sample_rate, waveform_numpy)
        
        对于长文本，会自动分割成多个片段分别生成，然后拼接，以避免显存溢出。
        """
        current_model = self.get_or_load_voxcpm()

        text = (text_input or "").strip()
        if len(text) == 0:
            raise ValueError("Please input text to synthesize.")

        prompt_wav_path = prompt_wav_path_input if prompt_wav_path_input else None
        prompt_text = prompt_text_input if prompt_text_input else None

        # 设置长文本阈值（字符数），超过此长度则分割处理
        # 根据经验，单个片段建议不超过 200-300 字符，这里设置为 200
        MAX_TEXT_LENGTH = 200
        
        # 如果有参考音频，先构建或获取 prompt cache
        prompt_cache = None
        if prompt_wav_path and prompt_text:
            prompt_cache = self._get_or_build_prompt_cache(
                prompt_wav_path=prompt_wav_path,
                prompt_text=prompt_text,
                denoise=denoise
            )
        
        # 如果文本较短，直接生成
        if len(text) <= MAX_TEXT_LENGTH:
            print(f"Generating audio for text: '{text[:60]}...'")
            
            # 如果有 prompt cache，直接使用它
            if prompt_cache is not None:
                # 文本正则化
                normalized_text = text
                if do_normalize:
                    if current_model.text_normalizer is None:
                        from voxcpm.utils.text_normalize import TextNormalizer
                        current_model.text_normalizer = TextNormalizer()
                    normalized_text = current_model.text_normalizer.normalize(text)
                
                # 使用缓存的 prompt cache 生成
                generate_result = current_model.tts_model._generate_with_prompt_cache(
                    target_text=normalized_text,
                    prompt_cache=prompt_cache,
                    min_len=2,
                    max_len=4096,
                    inference_timesteps=int(inference_timesteps_input),
                    cfg_value=float(cfg_value_input),
                    retry_badcase=True,
                    retry_badcase_max_times=3,
                    retry_badcase_ratio_threshold=6.0,
                    streaming=False,
                )
                # generate_result 返回 (wav, text_tokens, audio_features) 元组
                wav, _, _ = next(generate_result)
                wav = wav.squeeze(0).cpu().numpy()
            else:
                # 没有 prompt cache，使用常规生成方法
                wav = current_model.generate(
                    text=text,
                    prompt_text=prompt_text,
                    prompt_wav_path=prompt_wav_path,
                    cfg_value=float(cfg_value_input),
                    inference_timesteps=int(inference_timesteps_input),
                    normalize=do_normalize,
                    denoise=denoise,
                )
            return (current_model.tts_model.sample_rate, wav)
        
        # 长文本处理：分割成多个片段
        print(f"检测到长文本（{len(text)} 字符），将分割处理以避免显存溢出...")
        sentences = split_text_into_sentences(text, max_length=MAX_TEXT_LENGTH)
        print(f"文本已分割为 {len(sentences)} 个片段")
        
        # 文本正则化（如果有）
        normalized_sentences = sentences
        if do_normalize:
            if current_model.text_normalizer is None:
                from voxcpm.utils.text_normalize import TextNormalizer
                current_model.text_normalizer = TextNormalizer()
            normalized_sentences = [current_model.text_normalizer.normalize(s) for s in sentences]
        
        # 分段生成音频
        audio_chunks = []
        sample_rate = current_model.tts_model.sample_rate
        
        for i, sentence in enumerate(normalized_sentences):
            print(f"正在生成第 {i+1}/{len(normalized_sentences)} 个片段: '{sentence[:50]}...'")
            
            try:
                # 所有片段都使用同一个缓存的 prompt cache（如果有）
                if prompt_cache is not None:
                    # 使用缓存的 prompt cache 生成
                    generate_result = current_model.tts_model._generate_with_prompt_cache(
                        target_text=sentence,
                        prompt_cache=prompt_cache,
                        min_len=2,
                        max_len=4096,
                        inference_timesteps=int(inference_timesteps_input),
                        cfg_value=float(cfg_value_input),
                        retry_badcase=True,
                        retry_badcase_max_times=3,
                        retry_badcase_ratio_threshold=6.0,
                        streaming=False,
                    )
                    # generate_result 返回 (wav, text_tokens, audio_features) 元组
                    chunk_wav, _, _ = next(generate_result)
                    chunk_wav = chunk_wav.squeeze(0).cpu().numpy()
                else:
                    # 没有 prompt cache，使用常规生成方法
                    # 对于第一个片段，使用原始 prompt（如果有）
                    # 对于后续片段，不使用 prompt，以保持音色一致性
                    chunk_prompt_wav = prompt_wav_path if i == 0 else None
                    chunk_prompt_text = prompt_text if i == 0 else None
                    
                    chunk_wav = current_model.generate(
                        text=sentence,
                        prompt_text=chunk_prompt_text,
                        prompt_wav_path=chunk_prompt_wav,
                        cfg_value=float(cfg_value_input),
                        inference_timesteps=int(inference_timesteps_input),
                        normalize=False,  # 已经正则化过了
                        denoise=denoise if i == 0 else False,  # 只在第一个片段降噪
                    )
                
                audio_chunks.append(chunk_wav)
                
                # 清理显存（每个片段生成后清理，避免累积）
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                print(f"生成第 {i+1} 个片段时出错: {e}")
                import traceback
                traceback.print_exc()
                # 如果第一个片段失败，直接抛出异常
                if i == 0:
                    raise RuntimeError(f"第一个片段生成失败，无法继续: {e}")
                # 如果后续片段失败，尝试继续处理其他片段
                print(f"警告：跳过第 {i+1} 个片段，继续处理后续片段...")
                continue
        
        if not audio_chunks:
            raise RuntimeError("所有片段生成失败")
        
        # 拼接所有音频片段
        print(f"正在拼接 {len(audio_chunks)} 个音频片段...")
        final_wav = np.concatenate(audio_chunks)
        
        # 最终清理显存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print("长文本音频生成完成！")
        return (sample_rate, final_wav)


# ---------- UI Builders ----------

def create_demo_interface(demo: VoxCPMDemo):
    """Build the Gradio UI for VoxCPM demo."""
    # static assets (logo path)
    gr.set_static_paths(paths=[Path.cwd().absolute()/"assets"])

    with gr.Blocks(
        theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="gray",
            neutral_hue="slate",
            font=[gr.themes.GoogleFont("Inter"), "Arial", "sans-serif"]
        ),
        css="""
        .logo-container {
            text-align: center;
            margin: 0.5rem 0 1rem 0;
        }
        .logo-container img {
            height: 80px;
            width: auto;
            max-width: 200px;
            display: inline-block;
        }
        /* Bold accordion labels */
        #acc_quick details > summary,
        #acc_tips details > summary {
            font-weight: 600 !important;
            font-size: 1.1em !important;
        }
        /* Bold labels for specific checkboxes */
        #chk_denoise label,
        #chk_denoise span,
        #chk_normalize label,
        #chk_normalize span {
            font-weight: 600;
        }
        """
    ) as interface:
        # Header logo
        gr.HTML('<div class="logo-container"><img src="/gradio_api/file=assets/voxcpm_logo.png" alt="VoxCPM Logo"></div>')

        # Quick Start
        with gr.Accordion("📋 Quick Start Guide ｜快速入门", open=False, elem_id="acc_quick"):
            gr.Markdown("""
            ### How to Use ｜使用说明
            1. **(Optional) Provide a Voice Prompt** - Upload or record an audio clip to provide the desired voice characteristics for synthesis.  
               **（可选）提供参考声音** - 上传或录制一段音频，为声音合成提供音色、语调和情感等个性化特征
            2. **(Optional) Enter prompt text** - If you provided a voice prompt, enter the corresponding transcript here (auto-recognition available).  
               **（可选项）输入参考文本** - 如果提供了参考语音，请输入其对应的文本内容（支持自动识别）。
            3. **Enter target text** - Type the text you want the model to speak.  
               **输入目标文本** - 输入您希望模型朗读的文字内容。
            4. **Generate Speech** - Click the "Generate" button to create your audio.  
               **生成语音** - 点击"生成"按钮，即可为您创造出音频。
            """)

        # Pro Tips
        with gr.Accordion("💡 Pro Tips ｜使用建议", open=False, elem_id="acc_tips"):
            gr.Markdown("""
            ### Prompt Speech Enhancement｜参考语音降噪
            - **Enable** to remove background noise for a clean voice, with an external ZipEnhancer component. However, this will limit the audio sampling rate to 16kHz, restricting the cloning quality ceiling.  
              **启用**：通过 ZipEnhancer 组件消除背景噪音，但会将音频采样率限制在16kHz，限制克隆上限。
            - **Disable** to preserve the original audio's all information, including background atmosphere, and support audio cloning up to 44.1kHz sampling rate.  
              **禁用**：保留原始音频的全部信息，包括背景环境声，最高支持44.1kHz的音频复刻。

            ### Text Normalization｜文本正则化
            - **Enable** to process general text with an external WeTextProcessing component.  
              **启用**：使用 WeTextProcessing 组件，可支持常见文本的正则化处理。
            - **Disable** to use VoxCPM's native text understanding ability. For example, it supports phonemes input (For Chinese, phonemes are converted using pinyin, {ni3}{hao3}; For English, phonemes are converted using CMUDict, {HH AH0 L OW1}), try it!  
              **禁用**：将使用 VoxCPM 内置的文本理解能力。如，支持音素输入（如中文转拼音：{ni3}{hao3}；英文转CMUDict：{HH AH0 L OW1}）和公式符号合成，尝试一下！

            ### CFG Value｜CFG 值
            - **Lower CFG** if the voice prompt sounds strained or expressive, or instability occurs with long text input.  
              **调低**：如果提示语音听起来不自然或过于夸张，或者长文本输入出现稳定性问题。
            - **Higher CFG** for better adherence to the prompt speech style or input text, or instability occurs with too short text input.
              **调高**：为更好地贴合提示音频的风格或输入文本， 或者极短文本输入出现稳定性问题。

            ### Inference Timesteps｜推理时间步
            - **Lower** for faster synthesis speed.  
              **调低**：合成速度更快。
            - **Higher** for better synthesis quality.  
              **调高**：合成质量更佳。
            """)

        # Main controls
        with gr.Row():
            with gr.Column():
                prompt_wav = gr.Audio(
                    sources=["upload", 'microphone'],
                    type="filepath",
                    label="Prompt Speech (Optional, or let VoxCPM improvise)",
                    value="./examples/example.wav",
                )
                DoDenoisePromptAudio = gr.Checkbox(
                    value=False,
                    label="Prompt Speech Enhancement",
                    elem_id="chk_denoise",
                    info="We use ZipEnhancer model to denoise the prompt audio."
                )
                with gr.Row():
                    prompt_text = gr.Textbox(
                        value="Just by listening a few minutes a day, you'll be able to eliminate negative thoughts by conditioning your mind to be more positive.",
                        label="Prompt Text",
                        placeholder="Please enter the prompt text. Automatic recognition is supported, and you can correct the results yourself..."
                    )
                run_btn = gr.Button("Generate Speech", variant="primary")

            with gr.Column():
                cfg_value = gr.Slider(
                    minimum=1.0,
                    maximum=3.0,
                    value=2.0,
                    step=0.1,
                    label="CFG Value (Guidance Scale)",
                    info="Higher values increase adherence to prompt, lower values allow more creativity"
                )
                inference_timesteps = gr.Slider(
                    minimum=4,
                    maximum=30,
                    value=10,
                    step=1,
                    label="Inference Timesteps",
                    info="Number of inference timesteps for generation (higher values may improve quality but slower)"
                )
                with gr.Row():
                    text = gr.Textbox(
                        value="VoxCPM is an innovative end-to-end TTS model from ModelBest, designed to generate highly realistic speech.",
                        label="Target Text",
                    )
                with gr.Row():
                    DoNormalizeText = gr.Checkbox(
                        value=False,
                        label="Text Normalization",
                        elem_id="chk_normalize",
                        info="We use wetext library to normalize the input text."
                    )
                audio_output = gr.Audio(label="Output Audio")

        # Wiring
        run_btn.click(
            fn=demo.generate_tts_audio,
            inputs=[text, prompt_wav, prompt_text, cfg_value, inference_timesteps, DoNormalizeText, DoDenoisePromptAudio],
            outputs=[audio_output],
            show_progress=True,
            api_name="generate",
        )
        prompt_wav.change(fn=demo.prompt_wav_recognition, inputs=[prompt_wav], outputs=[prompt_text])

    return interface


def run_demo(server_name: str = "localhost", server_port: int = 7860, show_error: bool = True):
    demo = VoxCPMDemo()
    interface = create_demo_interface(demo)
    # Recommended to enable queue on Spaces for better throughput
    # default_concurrency_limit=1 确保单线程执行，避免 torch.compile CUDA graph 多线程问题
    interface.queue(max_size=10, default_concurrency_limit=1).launch(server_name=server_name, server_port=server_port, show_error=show_error)


if __name__ == "__main__":
    run_demo()