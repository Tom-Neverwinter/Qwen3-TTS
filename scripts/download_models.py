import os
import subprocess

models = [
    "Qwen/Qwen3-TTS-Tokenizer-12Hz",
    "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
    "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
]

base_dir = "models"
os.makedirs(base_dir, exist_ok=True)

for model in models:
    local_dir = os.path.join(base_dir, model.split('/')[-1])
    print(f"Downloading {model} to {local_dir}...")
    subprocess.run([
        "huggingface-cli", "download", model,
        "--local-dir", local_dir
    ], check=True)
