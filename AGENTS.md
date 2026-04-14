# VoxCPM Agent Guide

## Key Commands
- Install: `pip install voxcpm`
- CLI voice design: `voxcpm design --text "Hello" --output out.wav`
- CLI voice cloning: `voxcpm clone --text "Hello" --reference-audio ref.wav --output out.wav`
- CLI batch processing: `voxcpm batch --input texts.txt --output-dir ./outs`
- Web demo: `python app.py --port 8808`
- LoRA fine-tuning: `python scripts/train_voxcpm_finetune.py --config_path conf/voxcpm_v2/voxcpm_finetune_lora.yaml`
- Full fine-tuning: `python scripts/train_voxcpm_finetune.py --config_path conf/voxcpm_v2/voxcpm_finetune_all.yaml`
- Training WebUI: `python lora_ft_webui.py`

## Model Architecture
- VoxCPM2: 2B parameters, 48kHz output, 30 languages, voice design & controllable cloning
- Auto-detects model type from config.json architecture field ("voxcpm" or "voxcpm2")
- Pipeline: LocEnc → TSLM → RALM → LocDiT (AudioVAE V2 latent space)

## Common Gotchas
- Voice Design format: put description in parentheses at start of text: `"(A young woman)Hello"`
- Ultimate cloning requires both `reference_wav_path` and `prompt_wav_path` + `prompt_text`
- VoxCPM2 required for reference audio cloning (VoxCPM1.5 doesn't support it)
- Denoiser only applied when prompt/reference audio provided
- LoRA requires explicit config: `--lora-r 32 --lora-alpha 16` (default values)

## Environment
- Python ≥ 3.10 (<3.13), PyTorch ≥ 2.5.0, CUDA ≥ 12.0
- Device auto-selection: CUDA → MPS → CPU (override with `--device cuda:0`)
- Optimizations: torch.compile enabled by default (disable with `--no-optimize` for debugging)