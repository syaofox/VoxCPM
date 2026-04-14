#!/usr/bin/env python3
"""
Model download script for VoxCPM demo.
Downloads VoxCPM2, ZipEnhancer, and SenseVoiceSmall models to local directories.
"""

import os
import sys
from pathlib import Path

def download_hf_model(repo_id: str, dst_path: str):
    """Download all files from HuggingFace model repo."""
    os.makedirs(dst_path, exist_ok=True)
    
    from huggingface_hub import list_repo_files, hf_hub_download
    files = list_repo_files(repo_id)
    
    for f in files:
        print(f"  Downloading {f}...")
        hf_hub_download(
            repo_id=repo_id,
            filename=f,
            local_dir=dst_path,
        )
    
    print(f"  -> Downloaded to {dst_path}")

def download_ms_model(model_id: str, dst_path: str):
    """Download all files from ModelScope model repo."""
    from modelscope import snapshot_download
    
    snapshot_download(
        model_id=model_id,
        local_dir=dst_path,
    )
    
    print(f"  -> Downloaded to {dst_path}")

def main():
    models_dir = Path("./models")
    models_dir.mkdir(exist_ok=True)
    (models_dir / "huggingface").mkdir(exist_ok=True)
    (models_dir / "modelscope").mkdir(exist_ok=True)
    
    print(f"Models will be downloaded to: {models_dir.absolute()}")
    
    try:
        print("\n[1/3] Downloading VoxCPM2 model...")
        download_hf_model(
            "openbmb/VoxCPM2",
            str(models_dir / "huggingface" / "VoxCPM2"),
        )
        
        print("\n[2/3] Downloading ZipEnhancer (denoiser) model...")
        download_ms_model(
            "iic/speech_zipenhancer_ans_multiloss_16k_base",
            str(models_dir / "modelscope" / "speech_zipenhancer_ans_multiloss_16k_base"),
        )
        
        print("\n[3/3] Downloading SenseVoiceSmall ASR model...")
        download_ms_model(
            "iic/SenseVoiceSmall",
            str(models_dir / "modelscope" / "SenseVoiceSmall"),
        )
        
        voxel_path = str(models_dir / "huggingface" / "VoxCPM2")
        zipenhancer_path = str(models_dir / "modelscope" / "speech_zipenhancer_ans_multiloss_16k_base")
        sensevoice_path = str(models_dir / "modelscope" / "SenseVoiceSmall")

        print(f"\nAll models downloaded successfully to {models_dir.absolute()}")
        print("\nLocal model paths:")
        print(f"  VoxCPM2:      {voxel_path}")
        print(f"  ZipEnhancer:  {zipenhancer_path}")
        print(f"  SenseVoice:   {sensevoice_path}")
        print("\nTo run the demo offline:")
        print(f"  ./run_offline.sh")
        
    except Exception as e:
        print(f"Error downloading models: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()