import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel
import os
import io
import re
import base64
from fastapi import FastAPI, Request, HTTPException, Query
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from typing import List, Optional, Union, Generator
import uvicorn
import threading
import time
import json
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
        "api_port": 5000,
        "load_on_startup": True
    }
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r") as f:
                return {**defaults, **json.load(f)}
        except:
            return defaults
    return defaults

app = FastAPI(title="Qwen3-TTS Ultra API")

# Model Management
CURRENT_MODEL = {
    "key": None,
    "instance": None
}
CACHED_PROMPTS = {}

MODELS_CONFIG = {
    "CustomVoice-1.7B": "models/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    "CustomVoice-0.6B": "models/Qwen3-TTS-12Hz-0.6B-CustomVoice",
    "VoiceDesign-1.7B": "models/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
    "Base-1.7B": "models/Qwen3-TTS-12Hz-1.7B-Base",
    "Base-0.6B": "models/Qwen3-TTS-12Hz-0.6B-Base"
}

def get_model(model_key):
    global CURRENT_MODEL
    if CURRENT_MODEL["key"] == model_key and CURRENT_MODEL["instance"] is not None:
        return CURRENT_MODEL["instance"]
    
    print(f"API: Loading {model_key}...")
    CURRENT_MODEL["instance"] = None
    torch.cuda.empty_cache()
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    # Select attention implementation
    attn_impl = "sdpa"
    if torch.cuda.is_available():
        try:
            if torch.cuda.get_device_capability()[0] >= 8:
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

    # Pre-cache prompts for the library if it's a Base model
    if "Base" in model_key:
        import glob
        print("API: Pre-calculating voice library...")
        files = glob.glob("voices/*.wav") + glob.glob("voices/*.mp3") + glob.glob("voices/*.flac")
        for f in files:
            name = os.path.basename(f)
            if name not in CACHED_PROMPTS:
                try:
                    pts = model.create_voice_clone_prompt(ref_audio=f, x_vector_only_mode=True)
                    CACHED_PROMPTS[name] = pts[0]
                except:
                    pass
    return model

# Utils
def split_sentences(text: str) -> List[str]:
    """Split text into sentences for better streaming"""
    # Simple regex split for English and common punctuation
    sentences = re.split(r'([.!?。！？\n])', text)
    result = []
    current = ""
    for i in range(0, len(sentences)-1, 2):
        s = sentences[i].strip()
        p = sentences[i+1]
        if s:
            result.append(s + p)
    # Check for remaining bit
    if len(sentences) % 2 == 1:
        last = sentences[-1].strip()
        if last:
            result.append(last)
    
    # If no sentences found, return whole text
    return result if result else [text]

def parse_emotion(text: str):
    """
    Extracts emotion/instruction from various formats:
    (Emotion: Happy) Hello! -> instruct="Happy", text="Hello!"
    [Crying] Please don't go. -> instruct="Crying", text="Please don't go."
    *Whispering* Come closer. -> instruct="Whispering", text="Come closer."
    """
    patterns = [
        r'^\((?:Emotion|Instruct|Style):\s*(.*?)\)\s*(.*)',
        r'^\[(.*?)\s*\]\s*(.*)',
        r'^\*(.*?)\*\s*(.*)'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip(), match.group(2).strip()
    
    return None, text

# OpenAI Speech Request Schema
class SpeechRequest(BaseModel):
    model: str = "tts-1"
    input: str
    voice: str = "alloy"
    response_format: str = "mp3"
    speed: float = 1.0
    # Custom fields
    instruct: Optional[str] = None
    ref_text: Optional[str] = None
    stream: bool = False

@app.get("/speakers")
@app.get("/voices")
async def list_voices(request: Request):
    """Returns a list of all available voices (Presets + Local Actors)"""
    presets = ["Vivian", "Serena", "Uncle_Fu", "Dylan", "Eric", "Ryan", "Aiden", "Ono_Anna", "Sohee", "Narrator"]
    
    local_voices = []
    if os.path.exists("voices"):
        files = os.listdir("voices")
        # Keep original filenames for cloning
        local_voices = [f for f in files if f.lower().endswith(('.wav', '.mp3', '.flac'))]
    
    all_voices = presets + local_voices
    
    # If the user specifically hit /speakers, return a flat list (XTTS/ST standard)
    if "speakers" in str(request.url):
        return all_voices
        
    return {
        "presets": presets,
        "library": local_voices,
        "all": all_voices
    }

def chatterbox_parse(text: str) -> List[dict]:
    """
    Parses a script like:
    Narrator: It was a dark night.
    Vivian: (Scared) Who is there?
    """
    lines = text.strip().split('\n')
    script = []
    # Match "Speaker Name: Text"
    pattern = r'^([\w\s._-]+):\s*(.*)'
    
    for line in lines:
        line = line.strip()
        if not line: continue
        
        match = re.match(pattern, line)
        if match:
            spk = match.group(1).strip()
            content = match.group(2).strip()
            script.append({"speaker": spk, "text": content})
        else:
            # If no speaker detected, treat as previous speaker or default
            if script:
                script[-1]["text"] += " " + line
            else:
                script.append({"speaker": "Narrator", "text": line})
    return script

@app.post("/v1/audio/speech")
async def text_to_speech(request: SpeechRequest):
    try:
        raw_text = request.input
        
        # Check if this is a Multi-Speaker Script (Chatterbox Mode)
        # We detect this if there are multiple lines with Speaker: formatting
        if "\n" in raw_text and ":" in raw_text and ("Narrator:" in raw_text or any(v in raw_text for v in ["Vivian:", "Serena:", "Ryan:"])):
            script = chatterbox_parse(raw_text)
            
            async def generate_script_stream():
                for part in script:
                    spk = part["speaker"]
                    txt = part["text"]
                    
                    # Create a sub-request for each part
                    sub_req = SpeechRequest(
                        input=txt,
                        voice=spk,
                        model=request.model,
                        stream=False
                    )
                    
                    # Reuse the logic by calling a helper (refactored below)
                    async for chunk in generate_voice_chunk(sub_req):
                        yield chunk
            
            return StreamingResponse(generate_script_stream(), media_type="audio/wav")

        # Standard Single Speaker Path (Refactored into generate_voice_chunk)
        return StreamingResponse(generate_voice_chunk(request), media_type="audio/wav")

    except Exception as e:
        print(f"API Error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

async def generate_voice_chunk(request: SpeechRequest) -> Generator[bytes, None, None]:
    raw_text = request.input
    speaker = request.voice
    
    # 1. Parse emotion
    extracted_instruct, clean_text = parse_emotion(raw_text)
    final_instruct = request.instruct or extracted_instruct
    is_hd = (request.model == "tts-1-hd" or "-1.7B" in request.model)
    
    # 2. Check for library voice
    voice_path = None
    speaker_filename = None
    if os.path.exists("voices"):
        possible_names = [speaker, f"{speaker}.wav", f"{speaker}.mp3", f"{speaker}.flac"]
        for name in possible_names:
            p = os.path.join("voices", name)
            if os.path.exists(p):
                voice_path = p
                speaker_filename = name
                break

    text_chunks = split_sentences(clean_text) if request.stream else [clean_text]
    
    for chunk in text_chunks:
        if not chunk.strip(): continue
        
        if voice_path:
            m_key = "Base-1.7B" if is_hd else "Base-0.6B"
            model = get_model(m_key)
            if speaker_filename in CACHED_PROMPTS and not request.ref_text:
                wavs, sr = model.generate_voice_clone(text=chunk, voice_clone_prompt=[CACHED_PROMPTS[speaker_filename]])
            else:
                wavs, sr = model.generate_voice_clone(text=chunk, ref_audio=voice_path, ref_text=request.ref_text, x_vector_only_mode=not bool(request.ref_text))
        else:
            m_key = "CustomVoice-1.7B" if is_hd else "CustomVoice-0.6B"
            model = get_model(m_key)
            qwen_speakers = model.get_supported_speakers()
            target_spk = speaker if speaker in qwen_speakers else qwen_speakers[0]
            wavs, sr = model.generate_custom_voice(text=chunk, speaker=target_spk, instruct=final_instruct)

        buffer = io.BytesIO()
        sf.write(buffer, wavs[0], sr, format='WAV')
        yield buffer.getvalue()

@app.post("/tts")
@app.get("/tts")
async def xtts_compatibility(
    request: Request,
    text: Optional[str] = Query(None),
    speaker_wav: Optional[str] = Query(None),
    language: Optional[str] = Query("en")
):
    """
    XTTSv2 Compatibility Endpoint for SillyTavern.
    Supports both GET and POST.
    """
    # 1. Extract params from GET or POST
    if request.method == "POST":
        try:
            body = await request.json()
            text = body.get("text", text)
            speaker_wav = body.get("speaker_wav", speaker_wav)
            language = body.get("language", language)
        except:
            pass

    if not text:
        raise HTTPException(status_code=400, detail="Missing 'text' parameter")

    # 2. Map language code
    lang_map = {
        "en": "English", "zh": "Chinese", "jp": "Japanese", "ko": "Korean",
        "de": "German", "fr": "French", "es": "Spanish", "ru": "Russian"
    }
    target_lang = lang_map.get(language.split("-")[0].lower(), "Auto")

    # 3. Create a SpeechRequest and forward to our main logic
    # We use tts-1-hd (1.7B) by default for XTTS compatibility as users expect quality
    req = SpeechRequest(
        input=text,
        voice=speaker_wav or "Vivian",
        model="tts-1-hd",
        stream=True # Faster response for ST
    )
    
    return await text_to_speech(req)

def run_api():
    configs = load_config()
    port = configs.get("api_port", 5000)
    print(f"🚀 Qwen3-TTS API Server starting on port {port}...")
    print(f"🔗 OpenAI Endpoint: http://localhost:{port}/v1")
    print(f"🔗 XTTS Endpoint:   http://localhost:{port}")
    uvicorn.run(app, host="0.0.0.0", port=int(port))

if __name__ == "__main__":
    run_api()
