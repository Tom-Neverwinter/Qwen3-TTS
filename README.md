# Qwen3-TTS Project

This project integrates several Qwen3-TTS models into a single, user-friendly interface.

## Folder Structure

- `models/`: Contains the downloaded Hugging Face models.
  - `Qwen3-TTS-12Hz-1.7B-CustomVoice`: Model for preset high-quality speakers with instructions.
  - `Qwen3-TTS-12Hz-1.7B-VoiceDesign`: Model for generating voices based on natural language descriptions.
  - `Qwen3-TTS-12Hz-1.7B-Base`: Base model for Zero-Shot Voice Cloning.
  - `Qwen3-TTS-12Hz-0.6B-Base`: Lightweight version of the base model.
  - `Qwen3-TTS-Tokenizer-12Hz`: Tokenizer for audio processing.
- `outputs/`: Directory where generated `.wav` files are saved.
- `scripts/`: Utility scripts.
  - `download_models.py`: Script used to download the models from Hugging Face.
- `app.py`: The main Gradio web application.

## How to Run

### ⚡ Quick Start (Recommended)
Double-click the **`RUN_QWEN3.bat`** file in the project folder.

**First-time users**: The batch file will automatically detect if dependencies are missing and offer to install them. The installer will:
- Detect your GPU and CUDA version
- Install Python 3.12 (if needed)
- Install PyTorch with appropriate CUDA support
- Install Flash Attention (if you have a compatible GPU)
- Install Qwen-TTS and all dependencies

### Manual Installation
If you prefer to install manually, run:
```powershell
powershell -ExecutionPolicy Bypass -File install.ps1
```

Then start the application:
```bash
.\RUN_QWEN3.bat
```

The web interface will open automatically at `http://127.0.0.1:7861`.

---

## ⚡ Performance Optimization

**Flash Attention 2** is automatically installed during setup if you have a compatible NVIDIA GPU (RTX 30/40 series or newer with compute capability ≥ 8.0).

This provides:
- **Faster generation** (up to 2x speedup)
- **Reduced VRAM usage** (up to 40% less memory)
- **Better quality** for long-form audio

The application will automatically use Flash Attention when available. Check the console output on startup to confirm.

## Features

- **Custom Voice**: Use pre-defined speakers and control their emotions/style with natural language.
- **Voice Design**: Create entirely new voices by describing them (e.g., "A deep, raspy voice for a villain").
- **Voice Clone & Library**: High-fidelity cloning of any voice from a short audio sample. Save your own voice or actors' voices to the library for easy access.

## 👥 Voice Library & Cloning
To use your own voice or a voice actor:
1.  Go to the **Voice Clone & Library** tab.
2.  **Upload or Record** a 5-15 second clear audio sample.
3.  Enter a name (e.g., `Hero`) and click **💾 Save to /voices**.
4.  This voice is now available in the dropdown (after clicking **Refresh**) and also via the API!

## 🔌 SillyTavern Integration
1.  In `app.py`, go to the **🔌 OpenAI API** tab and click **Start API Server**.
2.  In **SillyTavern**, go to **TTS Settings** -> **OpenAI**.
3.  Set **API URL** to `http://localhost:5000/v1`.
4.  In the **Voice** field, you can use:
    - Preset names like `Vivian`, `Serena`, `Uncle_Fu`, `Dylan`, `Eric`, `Ryan`, `Aiden`, `Ono_Anna`, or `Sohee`.
    - Any name you saved in your **Voice Library** (e.g., `Hero`).
