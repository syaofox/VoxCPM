#!/usr/bin/env python3
"""
自动识别参考音频的文本内容
用于获取 prompt_text 参数
"""
import sys
import torch
from funasr import AutoModel

def recognize_audio(audio_path):
    """识别音频文件的文本内容"""
    print(f"正在识别音频: {audio_path}")
    
    # 检查文件是否存在
    import os
    if not os.path.exists(audio_path):
        print(f"错误: 文件不存在: {audio_path}")
        return None
    
    # 加载 ASR 模型
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    
    try:
        asr_model = AutoModel(
            model="iic/SenseVoiceSmall",
            disable_update=True,
            device=device,
        )
        
        # 识别音频
        print("正在识别中...")
        result = asr_model.generate(input=audio_path, language="auto", use_itn=True)
        text = result[0]["text"].split('|>')[-1]
        
        return text.strip()
    except Exception as e:
        print(f"识别失败: {e}")
        return None

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python get_audio_text.py <音频文件路径>")
        print("示例: python get_audio_text.py data/京京.wav")
        sys.exit(1)
    
    audio_path = sys.argv[1]
    text = recognize_audio(audio_path)
    
    if text:
        print("\n" + "="*50)
        print("识别到的文本内容:")
        print("="*50)
        print(text)
        print("="*50)
        print("\n请将上述文本复制到 Legado 配置的 prompt_text 字段中")
    else:
        print("\n识别失败，请手动填写参考文本")
