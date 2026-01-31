import torch
import soundfile as sf
import gradio as gr
from qwen_tts import Qwen3TTSModel
import os
import glob
import subprocess
import time
import requests
import torchaudio
import shutil
import argparse
import sys
import json
import atexit
import signal
import asyncio
import platform

# Fix for Windows asyncio ConnectionResetError [WinError 10054]
if platform.system() == "Windows":
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    except Exception:
        pass

CONFIG_FILE = "config.json"

def load_config():
    defaults = {
        "use_ssl": False,
        "host": "0.0.0.0",
        "port": 7861,
        "api_port": 5000,
        "share": False,
        "theme": "soft",
        "load_on_startup": True
    }
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r") as f:
                return {**defaults, **json.load(f)}
        except:
            return defaults
    return defaults

def update_config_item(key, value):
    config = load_config()
    config[key] = value
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=4)
    return f"Settings saved. Restart suggested for {key}."

# Create required directories
os.makedirs("outputs", exist_ok=True)
os.makedirs("voices", exist_ok=True)

# Shared state for API process
api_process = None

def cleanup_api_process():
    """Cleanup function to ensure API server subprocess is terminated"""
    global api_process
    if api_process is not None:
        try:
            print("🧹 Cleaning up API server subprocess...")
            api_process.terminate()
            api_process.wait(timeout=5)
        except:
            try:
                api_process.kill()
            except:
                pass
        api_process = None

# Register cleanup handlers
atexit.register(cleanup_api_process)

def signal_handler(signum, frame):
    """Handle termination signals"""
    print(f"\n🛑 Received signal {signum}, shutting down...")
    cleanup_api_process()
    sys.exit(0)

# Register signal handlers (Windows supports SIGINT and SIGTERM)
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# Cache for loaded models and voice prompts
CURRENT_MODEL = {
    "key": None,
    "instance": None
}
CACHED_PROMPTS = {}  # Format: { "filename": VoiceClonePromptItem }

MODELS_CONFIG = {
    "CustomVoice-1.7B": "models/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    "CustomVoice-0.6B": "models/Qwen3-TTS-12Hz-0.6B-CustomVoice",
    "VoiceDesign-1.7B": "models/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
    "Base-1.7B": "models/Qwen3-TTS-12Hz-1.7B-Base",
    "Base-0.6B": "models/Qwen3-TTS-12Hz-0.6B-Base"
}

def load_model(model_key):
    global CURRENT_MODEL
    if CURRENT_MODEL["key"] == model_key and CURRENT_MODEL["instance"] is not None:
        return CURRENT_MODEL["instance"]
    
    print(f"Loading {model_key} model...")
    CURRENT_MODEL["instance"] = None
    torch.cuda.empty_cache()
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    # Select attention implementation
    attn_impl = "sdpa" # Default
    if torch.cuda.is_available():
        try:
            if torch.cuda.get_device_capability()[0] >= 8:
                # Still check if flash_attn is actually installed to avoid crash
                import importlib.util
                if importlib.util.find_spec("flash_attn"):
                    attn_impl = "flash_attention_2"
        except:
             pass

    path = MODELS_CONFIG[model_key]
    model = Qwen3TTSModel.from_pretrained(
        path,
        device_map=device,
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        attn_implementation=attn_impl,
    )
    CURRENT_MODEL["instance"] = model
    CURRENT_MODEL["key"] = model_key

    # Auto-cache prompts if it's a Base model
    if "Base" in model_key:
        cache_all_prompts(model)
        
    return model

def cache_all_prompts(model):
    global CACHED_PROMPTS
    print("Pre-calculating voice prompts for library...")
    files = glob.glob("voices/*.wav") + glob.glob("voices/*.mp3") + glob.glob("voices/*.flac")
    for f in files:
        name = os.path.basename(f)
        if name not in CACHED_PROMPTS:
            try:
                # We always cache in x_vector_only mode for library items for maximum flexibility
                prompt_items = model.create_voice_clone_prompt(ref_audio=f, x_vector_only_mode=True)
                CACHED_PROMPTS[name] = prompt_items[0]
            except Exception as e:
                print(f"Failed to cache {name}: {e}")
    print(f"Cached {len(CACHED_PROMPTS)} voices.")

def toggle_api(enable):
    global api_process
    if enable:
        if api_process is None or api_process.poll() is not None:
            # Start the API server in a separate process
            api_process = subprocess.Popen([sys.executable, "api_server.py"])
            time.sleep(2) # Wait for startup
            return "✅ API Server Started (Port 5000). Use 'http://localhost:5000/v1' in SillyTavern."
        else:
            return "✅ API Server is already running."
    else:
        if api_process is not None:
            api_process.terminate()
            api_process = None
            return "❌ API Server Stopped."
        return "❌ API Server is not running."

def get_voice_files():
    files = glob.glob("voices/*.wav") + glob.glob("voices/*.mp3") + glob.glob("voices/*.flac")
    return [os.path.basename(f) for f in files]

def refresh_voices():
    voices = get_voice_files()
    return gr.Dropdown(choices=voices)

def generate_custom_voice(text, speaker, language, instruct, model_key):
    model = load_model(model_key)
    wavs, sr = model.generate_custom_voice(
        text=text,
        language=language if language != "Auto" else None,
        speaker=speaker,
        instruct=instruct if instruct else None,
    )
    output_path = f"outputs/custom_{speaker}.wav"
    sf.write(output_path, wavs[0], sr)
    return output_path

def generate_voice_design(text, language, instruct, model_key):
    model = load_model(model_key)
    wavs, sr = model.generate_voice_design(
        text=text,
        language=language if language != "Auto" else None,
        instruct=instruct,
    )
    output_path = f"outputs/designed_voice.wav"
    sf.write(output_path, wavs[0], sr)
    return output_path

def generate_voice_clone(text, language, ref_audio, ref_text, model_key, local_voice):
    model = load_model(model_key)
    
    prompt_item = None
    if local_voice and local_voice != "None" and local_voice in CACHED_PROMPTS:
        prompt_item = CACHED_PROMPTS[local_voice]
    
    # If not cached or using an ad-hoc ref_audio
    if not prompt_item and not ref_audio:
        # Check if local_voice was provided but not cached
        if local_voice and local_voice != "None":
            ref_audio = os.path.join("voices", local_voice)
        else:
            return None

    # Use prompt_item for ultra-fast generation if we have it
    if prompt_item:
        wavs, sr = model.generate_voice_clone(
            text=text,
            language=language if language != "Auto" else None,
            voice_clone_prompt=[prompt_item], # Model expects a list
        )
    else:
        # Standard flow (extracts features on the fly)
        wavs, sr = model.generate_voice_clone(
            text=text,
            language=language if language != "Auto" else None,
            ref_audio=ref_audio,
            ref_text=ref_text if ref_text else None,
            x_vector_only_mode=True if not ref_text else False
        )
        
    output_path = f"outputs/cloned_voice.wav"
    sf.write(output_path, wavs[0], sr)
    return output_path

def save_voice_sample(audio_path, name):
    if not audio_path or not name:
        return "❌ Please provide both audio and a name."
    
    # Sanitize name
    name = "".join([c for c in name if c.isalnum() or c in (' ', '.', '_')]).strip()
    if not name.endswith(".wav"):
        name += ".wav"
    
    target_path = os.path.join("voices", name)
    try:
        shutil.copy(audio_path, target_path)
        return f"✅ Saved as {name} in /voices folder. Click 'Refresh' to see it in the list!"
    except Exception as e:
        return f"❌ Error saving: {str(e)}"

def generate_chatterbox(script_text, model_key):
    """Parses script and generates multi-speaker audio"""
    import re
    import numpy as np
    from api_server import chatterbox_parse
    
    parts = chatterbox_parse(script_text)
    if not parts:
        return None
    
    combined_wav = []
    sample_rate = 24000 # Default
    
    for part in parts:
        spk = part["speaker"]
        text = part["text"]
        
        # Determine if it's a built-in or a library voice
        voices = get_voice_files()
        
        try:
            if spk in SUPPORTED_SPEAKERS:
                # Built-in
                wav, sr = load_model(model_key).generate_custom_voice(text=text, speaker=spk)
            elif spk in voices or f"{spk}.wav" in voices or f"{spk}.mp3" in voices:
                # Library (using Base-1.7B for quality in studio)
                lib_name = spk if spk in voices else (f"{spk}.wav" if f"{spk}.wav" in voices else f"{spk}.mp3")
                wav, sr = generate_voice_clone_raw(text, "Auto", None, None, "Base-1.7B", lib_name)
            else:
                # Fallback to Narrator (Uncle_Fu)
                wav, sr = load_model(model_key).generate_custom_voice(text=text, speaker="Uncle_Fu")
            
            combined_wav.append(wav[0])
            sample_rate = sr
        except Exception as e:
            print(f"Chatterbox Error for {spk}: {e}")
            continue
            
    if not combined_wav:
        return None
        
    final_wav = np.concatenate(combined_wav)
    output_path = "outputs/chatterbox_production.wav"
    sf.write(output_path, final_wav, sample_rate)
    return output_path

def generate_voice_clone_raw(text, language, ref_audio, ref_text, model_key, local_voice):
    """Helper for internal usage without Gradio typing"""
    model = load_model(model_key)
    prompt_item = None
    if local_voice and local_voice in CACHED_PROMPTS:
        prompt_item = CACHED_PROMPTS[local_voice]
    
    if prompt_item:
        wavs, sr = model.generate_voice_clone(text=text, language=None, voice_clone_prompt=[prompt_item])
    else:
        ref = os.path.join("voices", local_voice) if local_voice else ref_audio
        wavs, sr = model.generate_voice_clone(text=text, language=None, ref_audio=ref, x_vector_only_mode=True)
    return wavs[0], sr

# Initial metadata
SUPPORTED_SPEAKERS = ["Vivian", "Serena", "Uncle_Fu", "Dylan", "Eric", "Ryan", "Aiden", "Ono_Anna", "Sohee"]
SUPPORTED_LANGS = ["Auto", "Chinese", "English", "Japanese", "Korean", "German", "French", "Russian", "Portuguese", "Spanish", "Italian"]

# Handle Startup Config
configs = load_config()

if configs.get("load_on_startup", True):
    try:
        print("🚀 Initializing startup model...")
        temp = load_model("CustomVoice-1.7B")
        # Merge with existing voices to ensure all known presets and model-detected presets are available
        detected_speakers = temp.get_supported_speakers()
        for s in detected_speakers:
            if s not in SUPPORTED_SPEAKERS:
                SUPPORTED_SPEAKERS.append(s)
        print(f"✅ Loaded {len(SUPPORTED_SPEAKERS)} speakers.")
    except Exception as e:
        print(f"⚠️ Startup model failed to load (possibly VRAM): {e}")

with gr.Blocks(title="Qwen3-TTS Ultra Studio") as demo:
    gr.Markdown("# 🎙️ Qwen3-TTS Ultra Studio")
    gr.Markdown("---")
    
    with gr.Tabs():
        # TAB 1: PRODUCTION (THE LAB)
        with gr.Tab("🎭 Speech Lab"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 1. Configure the Voice")
                    voice_source = gr.Radio(
                        choices=["Built-in Presets", "My Actor Library", "Instant Design"], 
                        label="Voice Source", 
                        value="Built-in Presets"
                    )
                    
                    with gr.Group() as grp_builtin:
                        lab_model_preset = gr.Dropdown(choices=["CustomVoice-1.7B", "CustomVoice-0.6B"], label="Model", value="CustomVoice-1.7B")
                        lab_speaker_preset = gr.Dropdown(choices=SUPPORTED_SPEAKERS, label="Speaker", value=SUPPORTED_SPEAKERS[0])
                    
                    with gr.Group(visible=False) as grp_actor:
                        lab_model_clone = gr.Dropdown(choices=["Base-0.6B", "Base-1.7B"], label="Model (Cloning Engine)", value="Base-0.6B")
                        with gr.Row():
                            actor_list = get_voice_files()
                            lab_actor_select = gr.Dropdown(choices=actor_list if actor_list else ["None Found"], label="Select Actor", value=actor_list[0] if actor_list else "None Found")
                            lab_refresh = gr.Button("🔄", scale=0)
                    
                    with gr.Group(visible=False) as grp_design:
                        lab_model_design = gr.Dropdown(choices=["VoiceDesign-1.7B"], label="Model", value="VoiceDesign-1.7B")
                        lab_design_prompt = gr.Textbox(label="Voice Description", placeholder="e.g. A warm, elderly man with a British accent")

                    gr.Markdown("### 2. Enter Speech")
                    lab_text = gr.Textbox(label="Target Text", placeholder="What should the actor say?", lines=4)
                    lab_lang = gr.Dropdown(choices=SUPPORTED_LANGS, label="Language", value="Auto")
                    lab_instruct = gr.Textbox(label="Style/Emotion Instruction (Optional)", placeholder="e.g. Speak with intense fear")
                    
                    lab_btn = gr.Button("🚀 Generate Production Audio", variant="primary", size="lg")
                
                with gr.Column(scale=1):
                    gr.Markdown("### 3. Output")
                    lab_out = gr.Audio(label="Final Studio Output")
                    gr.Markdown("""
                    ### 💡 Production Tips:
                    - **Built-in Voices** are highly stable and follow emotional instructions best.
                    - **My Actors** uses zero-shot cloning from your MP3 samples.
                    - **Instant Design** creates a unique voice every time based on your description.
                    
                    ### 🎭 Premade Voice Reference:
                    | Speaker | Description | Native Lang |
                    | :--- | :--- | :--- |
                    | **Vivian** | Bright, sl. edgy young female | Chinese |
                    | **Serena** | Warm, gentle young female | Chinese |
                    | **Uncle_Fu** | Seasoned male, low mellow timbre | Chinese |
                    | **Dylan** | Youthful Beijing male, natural | Chinese |
                    | **Eric** | Lively Chengdu male, sl. husky | Chinese |
                    | **Ryan** | Dynamic male, strong rhythm | English |
                    | **Aiden** | Sunny American male, clear | English |
                    | **Ono_Anna**| Playful Japanese female, light | Japanese |
                    | **Sohee** | Warm Korean female, rich emotion| Korean |
                    """)

            # Visibility Logic
            def update_visibility(source):
                return {
                    grp_builtin: gr.update(visible=(source == "Built-in Presets")),
                    grp_actor: gr.update(visible=(source == "My Actor Library")),
                    grp_design: gr.update(visible=(source == "Instant Design"))
                }
            voice_source.change(update_visibility, inputs=[voice_source], outputs=[grp_builtin, grp_actor, grp_design])
            
            def production_generate(source, text, lang, instruct, model_preset, speaker_preset, model_clone, actor_file, model_design, design_prompt):
                if source == "Built-in Presets":
                    return generate_custom_voice(text, speaker_preset, lang, instruct, model_preset)
                elif source == "My Actor Library":
                    return generate_voice_clone(text, lang, None, None, model_clone, actor_file)
                elif source == "Instant Design":
                    # Combine prompt and instruct for design
                    full_instruct = f"{design_prompt}. {instruct}" if instruct else design_prompt
                    return generate_voice_design(text, lang, full_instruct, model_design)

            lab_btn.click(
                production_generate, 
                inputs=[voice_source, lab_text, lab_lang, lab_instruct, lab_model_preset, lab_speaker_preset, lab_model_clone, lab_actor_select, lab_model_design, lab_design_prompt],
                outputs=lab_out
            )
            lab_refresh.click(refresh_voices, outputs=lab_actor_select)

        # TAB 2: ACTOR FACTORY (MAKING VOICES)
        with gr.Tab("🏗️ Actor Factory"):
            gr.Markdown("### 🏭 Make Your Own Voices")
            gr.Markdown("Use this tab to enroll your voice actors or yourself into the library.")
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### Step 1: Provide Reference Material")
                    factory_upload = gr.Audio(label="Upload MP3/WAV or Record Voice", type="filepath", sources=["upload", "microphone"])
                    factory_filename = gr.Textbox(label="Current File", interactive=False, placeholder="No file selected")
                    gr.Markdown("*Note: For MIDI files, please convert to audio first or upload for reference storage.*")
                
                with gr.Column():
                    gr.Markdown("#### Step 2: Train & Preview the Voice")
                    factory_model = gr.Dropdown(choices=["Base-0.6B", "Base-1.7B"], label="Training Model", value="Base-0.6B")
                    factory_test_text = gr.Textbox(label="Preview Sentence", value="Hello there! I am your new voice actor. I'm ready to record.", lines=2)
                    factory_test_btn = gr.Button("🎯 Train & Preview Voice")
                    factory_test_out = gr.Audio(label="Voice Preview")
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### Step 3 (Final): Enroll Voice Actor into Library")
                    factory_name = gr.Textbox(label="Actor Name", placeholder="e.g. DarkLord_James")
                    factory_save_btn = gr.Button("💾 Enroll Voice Actor into Library", variant="primary", size="lg")
                    factory_status = gr.Markdown("")

            # Update filename display when audio is uploaded
            def update_filename(audio_path):
                if audio_path:
                    return os.path.basename(audio_path)
                return "No file selected"
            
            factory_upload.change(update_filename, inputs=[factory_upload], outputs=[factory_filename])
            factory_save_btn.click(save_voice_sample, inputs=[factory_upload, factory_name], outputs=factory_status)
            factory_test_btn.click(
                lambda text, audio, model: generate_voice_clone(text, "Auto", audio, None, model, "None"),
                inputs=[factory_test_text, factory_upload, factory_model],
                outputs=factory_test_out
            )

        # TAB 2.5: CHATTERBOX (STUDIO)
        with gr.Tab("🎭 Studio (Chatterbox)"):
            gr.Markdown("### 🎞️ Multi-Speaker Script Production")
            gr.Markdown("Write a script with `Speaker: Text` format to generate a multi-role dialogue.")
            
            with gr.Row():
                with gr.Column(scale=2):
                    chatter_script = gr.Textbox(
                        label="Script", 
                        placeholder="Narrator: It was a dark and stormy night.\nVivian: (Scared) I hear something in the attic!\nRyan: Don't worry, I'll go check.",
                        lines=15
                    )
                    chatter_model = gr.Dropdown(choices=["CustomVoice-1.7B", "CustomVoice-0.6B"], label="Primary Engine", value="CustomVoice-1.7B")
                    chatter_btn = gr.Button("🎬 Generate Final Production", variant="primary", size="lg")
                
                with gr.Column(scale=1):
                    chatter_out = gr.Audio(label="Final Combined Audio")
                    gr.Markdown("""
                    #### 📋 How to use:
                    1. Use **Speaker Names** from the lists below.
                    2. Use `(Emotion)` to add style instructions.
                    3. Each line must start with `Name:`.
                    
                    #### 🗣️ Available Speakers:
                    **Built-in:**  
                    `Vivian`, `Serena`, `Uncle_Fu`, `Dylan`, `Eric`, `Ryan`, `Aiden`, `Ono_Anna`, `Sohee`
                    
                    **Your Library Actors:**
                    """ + ", ".join([f"`{v}`" for v in get_voice_files()]))

            chatter_btn.click(generate_chatterbox, inputs=[chatter_script, chatter_model], outputs=chatter_out)

        # TAB 3: API CONTROL
        with gr.Tab("🔌 Connections"):
            gr.Markdown("### OpenAI API / SillyTavern / External Tools")
            with gr.Row():
                api_btn_start = gr.Button("▶️ Start API Server", variant="primary")
                api_btn_stop = gr.Button("🛑 Stop API Server", variant="secondary")
            api_status = gr.Textbox(label="Status", value="Inactive", interactive=False)
            
            gr.Markdown("""
            #### How to use your Custom Actors in SillyTavern:
            1. Enroll your actor in the **Actor Factory** (e.g. named `MyHero`).
            2. Start the API Server here.
            3. In SillyTavern TTS settings, set the **Voice** name to exactly `MyHero`.
            """)
            
            api_btn_start.click(lambda: toggle_api(True), outputs=api_status)
            api_btn_stop.click(lambda: toggle_api(False), outputs=api_status)

        # TAB 4: SETTINGS
        with gr.Tab("⚙️ Settings"):
            gr.Markdown("### 🛠️ Global Configuration")
            with gr.Row():
                with gr.Column():
                    cfg_ssl = gr.Checkbox(label="Enable SSL (HTTPS)", value=configs["use_ssl"], info="Requires cert.pem and key.pem in .cert folder. Enable this if you need microphone access from other devices.")
                    cfg_startup = gr.Checkbox(label="Load Model on Startup", value=configs["load_on_startup"], info="Automatically load the 1.7B model when the app starts.")
                    cfg_port = gr.Number(label="Web UI Port", value=configs["port"], precision=0)
                    cfg_api_port = gr.Number(label="API Server Port", value=configs["api_port"], precision=0)
                    cfg_save = gr.Button("💾 Save Configuration", variant="primary")
                    cfg_msg = gr.Markdown("")
                
                with gr.Column():
                    gr.Markdown("""
                    #### ⚠️ Important Note:
                    Most settings (SSL, Port) require a **full application restart** to take effect.
                    
                    If you enable SSL, ensure you have:
                    - `.cert/cert.pem`
                    - `.cert/key.pem`
                    
                    The app will automatically fail back to HTTP if certificates are missing.
                    """)
            
            def save_all_settings(ssl, startup, port, api_port):
                c = load_config()
                c["use_ssl"] = ssl
                c["load_on_startup"] = startup
                c["port"] = int(port)
                c["api_port"] = int(api_port)
                with open(CONFIG_FILE, "w") as f:
                    json.dump(c, f, indent=4)
                return "✅ Settings saved to `config.json`. Please restart the application for changes to take effect."

            cfg_save.click(save_all_settings, inputs=[cfg_ssl, cfg_startup, cfg_port, cfg_api_port], outputs=cfg_msg)

if __name__ == "__main__":
    configs = load_config()
    
    parser = argparse.ArgumentParser(description="Qwen3-TTS Ultra Studio")
    parser.add_argument("--host", type=str, default=configs["host"], help="Host IP")
    parser.add_argument("--port", type=int, default=configs["port"], help="Port number")
    parser.add_argument("--ssl", action="store_true", help="Enable HTTPS with self-signed cert")
    parser.add_argument("--share", action="store_true", help="Create a public link")
    args = parser.parse_args()

    # CLI args override config where explicit
    use_ssl = args.ssl or configs["use_ssl"]

    launch_kwargs = {
        "server_name": args.host,
        "server_port": args.port,
        "share": args.share or configs["share"],
        "inbrowser": True
    }

    if use_ssl:
        cert_path = os.path.join(".cert", "cert.pem")
        key_path = os.path.join(".cert", "key.pem")
        if os.path.exists(cert_path) and os.path.exists(key_path):
            launch_kwargs["ssl_certfile"] = cert_path
            launch_kwargs["ssl_keyfile"] = key_path
            launch_kwargs["ssl_verify"] = False
            print(f"🔐 SSL Enabled. Using certificates from .cert/")
        else:
            print("⚠️ SSL requested but certificates not found in .cert/. Running without SSL.")

    demo.launch(**launch_kwargs)
