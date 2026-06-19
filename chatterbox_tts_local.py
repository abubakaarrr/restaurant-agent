"""
Chatterbox TTS - Local standalone server (no Modal, no R2).

Voices are read from a local directory (./voices/ by default).
Run with:
    uvicorn chatterbox_tts_local:web_app --host 0.0.0.0 --port 8080

Or via Docker Compose (recommended):
    docker compose up tts

Voice files should be placed in:
    voices/system/<voice-id>.wav    (system voices)
    voices/custom/<voice-id>.wav    (custom cloned voices)
"""

import io
import os
from pathlib import Path

import torch
import torchaudio as ta
from chatterbox.tts_turbo import ChatterboxTurboTTS
from fastapi import Depends, FastAPI, HTTPException, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

VOICES_DIR = Path(os.environ.get("VOICES_DIR", "./voices"))
API_KEY = os.environ.get("CHATTERBOX_API_KEY", "local-dev-key")

# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

api_key_scheme = APIKeyHeader(
    name="x-api-key",
    scheme_name="ApiKeyAuth",
    auto_error=False,
)


def verify_api_key(x_api_key: str | None = Security(api_key_scheme)):
    if not API_KEY or x_api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")
    return x_api_key


# ---------------------------------------------------------------------------
# Model loading (once at startup)
# ---------------------------------------------------------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[chatterbox] Loading model on {device} ...")

if device == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True

model = ChatterboxTurboTTS.from_pretrained(device=device)

if device == "cuda":
    try:
        model = model.half()
        print("[chatterbox] Model loaded in float16 (4GB VRAM optimisation)")
    except Exception:
        print("[chatterbox] float16 not supported, using float32")

print(f"[chatterbox] Model ready. Voices dir: {VOICES_DIR.resolve()}")

# ---------------------------------------------------------------------------
# Request schema
# ---------------------------------------------------------------------------


class TTSRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=5000)
    voice_key: str = Field(..., min_length=1, max_length=300)
    temperature: float = Field(default=0.8, ge=0.0, le=2.0)
    top_p: float = Field(default=0.95, ge=0.0, le=1.0)
    top_k: int = Field(default=1000, ge=1, le=10000)
    repetition_penalty: float = Field(default=1.2, ge=1.0, le=2.0)
    norm_loudness: bool = Field(default=True)


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

web_app = FastAPI(
    title="Chatterbox TTS API (Local)",
    description="Text-to-speech with voice cloning — local dev server",
    docs_url="/docs",
)

web_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@web_app.get("/health")
def health():
    return {"status": "ok", "device": device}


@web_app.post(
    "/generate",
    dependencies=[Depends(verify_api_key)],
    responses={200: {"content": {"audio/wav": {}}}},
)
def generate_speech(request: TTSRequest):
    voice_path = VOICES_DIR / request.voice_key

    if not voice_path.exists():
        raise HTTPException(
            status_code=400,
            detail=f"Voice file not found: '{request.voice_key}'. "
                   f"Place WAV files in {VOICES_DIR.resolve()}",
        )

    try:
        with torch.inference_mode():
            wav = model.generate(
                request.prompt,
                audio_prompt_path=str(voice_path),
                temperature=request.temperature,
                top_p=request.top_p,
                top_k=request.top_k,
                repetition_penalty=request.repetition_penalty,
                norm_loudness=request.norm_loudness,
            )

        buffer = io.BytesIO()
        ta.save(buffer, wav, model.sr, format="wav")
        buffer.seek(0)

        return StreamingResponse(buffer, media_type="audio/wav")

    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        raise HTTPException(
            status_code=500,
            detail="GPU out of memory. Try a shorter text or restart the container.",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {e}")
