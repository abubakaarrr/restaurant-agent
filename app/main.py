"""FastAPI entry point — chat API + Vapi Custom LLM + Voice Studio + browser demo UI."""

from __future__ import annotations

import json
import logging
import io
import traceback
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
import httpx
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile, WebSocket
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from app.agent.runner import clear_session, get_session_history, run_agent, stream_agent_tokens
from app.config import settings
from app.db_pool import get_pool, close_pool
from app.retell_handler import handle_retell_connection

VOICES_DIR = Path("voices")
ALLOWED_AUDIO_TYPES = {
    "audio/wav", "audio/wave", "audio/x-wav",
    "audio/mpeg", "audio/mp3",
    "audio/ogg", "audio/flac",
    "audio/webm",
}
MAX_UPLOAD_SIZE = 20 * 1024 * 1024  # 20 MB


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm the connection pool at startup so the first call doesn't pay the
    # connection-setup cost mid-conversation.
    await get_pool()
    yield
    await close_pool()


app = FastAPI(
    title="Restaurant AI Receptionist",
    description="La Casa Restaurant — AI receptionist powered by LangGraph + GPT-4o",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")

VOICES_DIR = Path("voices")

class ChatRequest(BaseModel):
    message: str
    session_id: str = ""
    caller_phone: str = ""


class ChatResponse(BaseModel):
    reply: str
    session_id: str
    turn_count: int


# ── Health ────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "restaurant": settings.restaurant_name}


# ── Text chat endpoint (browser demo + testing) ───────────────

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """Text chat endpoint — used by the browser demo and simulate_call.py."""
    session_id = req.session_id or str(uuid.uuid4())
    try:
        reply = await run_agent(session_id, req.message, req.caller_phone)
    except Exception as e:
        logger.error("Agent error:\n%s", traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

    return ChatResponse(
        reply=reply,
        session_id=session_id,
        turn_count=len(get_session_history(session_id)) // 2,
    )


@app.delete("/session/{session_id}")
async def clear_session(session_id: str):
    clear_session(session_id)
    return {"cleared": session_id}


# ── Vapi Custom LLM endpoint ──────────────────────────────────

def _sse_chunk(chunk_id: str, delta: dict, finish_reason: str | None = None) -> str:
    payload = {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }
    return f"data: {json.dumps(payload)}\n\n"


@app.post("/vapi/llm/chat/completions")
async def vapi_llm(request: Request):
    """
    OpenAI-compatible streaming endpoint for Vapi's Custom LLM feature.
    Streams real LLM tokens as they are generated (after any tool calls).
    """
    body = await request.json()

    messages: list[dict] = body.get("messages", [])
    call_info: dict = body.get("call", {})
    call_id: str = call_info.get("id", str(uuid.uuid4()))
    caller_number: str = call_info.get("customer", {}).get("number", "")

    user_messages = [m for m in messages if m.get("role") == "user"]
    latest_user_msg = user_messages[-1]["content"] if user_messages else ""

    async def sse_stream():
        chunk_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        yield _sse_chunk(chunk_id, {"role": "assistant", "content": ""})

        if not user_messages:
            greeting = (
                f"Hello! Thank you for calling {settings.restaurant_name}. "
                "How can I help you today?"
            )
            yield _sse_chunk(chunk_id, {"content": greeting})
        else:
            try:
                async for token in stream_agent_tokens(call_id, latest_user_msg, caller_number):
                    if token:
                        yield _sse_chunk(chunk_id, {"content": token})
            except Exception:
                logger.error("Vapi agent stream error:\n%s", traceback.format_exc())
                yield _sse_chunk(
                    chunk_id,
                    {"content": "I'm sorry, I had a technical issue. Could you please repeat that?"},
                )

        yield _sse_chunk(chunk_id, {}, finish_reason="stop")
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        sse_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/vapi/webhook")
async def vapi_webhook(request: Request):
    """Vapi server webhook — receives call lifecycle events."""
    body = await request.json()
    msg = body.get("message", {})
    event_type = msg.get("type", "unknown")

    if event_type == "end-of-call-report":
        call_id = msg.get("call", {}).get("id", "")
        clear_session(call_id)

    return {"status": "ok"}


# ── Retell Custom LLM WebSocket ───────────────────────────────

@app.websocket("/retell-ws/{call_id}")
async def retell_ws(websocket: WebSocket, call_id: str):
    """Retell connects here for each call. Retell does STT/turn-taking/TTS;
    we run the LangGraph agent and stream text replies back."""
    await websocket.accept()
    await handle_retell_connection(websocket, call_id)


@app.post("/api/retell/web-call")
async def retell_web_call():
    """Mint a short-lived Retell web-call access token for the browser SDK.

    The API key stays server-side; the browser only ever sees the access token,
    which Retell invalidates after 30s if a call isn't started.
    """
    if not settings.retell_api_key or not settings.retell_agent_id:
        raise HTTPException(
            status_code=400,
            detail="RETELL_API_KEY and RETELL_AGENT_ID must be set in .env",
        )

    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            "https://api.retellai.com/v2/create-web-call",
            headers={"Authorization": f"Bearer {settings.retell_api_key}"},
            json={"agent_id": settings.retell_agent_id},
        )

    if resp.status_code not in (200, 201):
        logger.error("Retell create-web-call failed: %s %s", resp.status_code, resp.text)
        raise HTTPException(status_code=502, detail=f"Retell error: {resp.text}")

    data = resp.json()
    return {
        "access_token": data.get("access_token", ""),
        "call_id": data.get("call_id", ""),
    }


# ── Voice Studio API ──────────────────────────────────────────

@app.get("/api/voices")
async def list_voices():
    """List all available voices (system + custom)."""
    voices = []
    for category in ("system", "custom"):
        folder = VOICES_DIR / category
        if not folder.exists():
            continue
        for f in folder.iterdir():
            if f.suffix.lower() in (".wav", ".mp3", ".ogg", ".flac", ".webm"):
                voices.append({
                    "id": f.stem,
                    "name": f.stem.replace("-", " ").replace("_", " ").title(),
                    "category": category,
                    "file": f.name,
                    "voice_key": f"{category}/{f.name}",
                    "size_kb": round(f.stat().st_size / 1024, 1),
                })
    return {"voices": voices}


@app.post("/api/voices/upload")
async def upload_voice(
    file: UploadFile = File(...),
    name: str = Form(...),
):
    """Upload an audio sample to create a custom cloned voice."""
    if file.content_type and file.content_type not in ALLOWED_AUDIO_TYPES:
        raise HTTPException(400, f"Unsupported audio type: {file.content_type}")

    content = await file.read()
    if len(content) > MAX_UPLOAD_SIZE:
        raise HTTPException(400, "File too large (max 20 MB)")
    if len(content) < 1000:
        raise HTTPException(400, "File too small — need at least 10 seconds of audio")

    safe_name = "".join(c if c.isalnum() or c in "-_" else "-" for c in name.strip())
    if not safe_name:
        safe_name = uuid.uuid4().hex[:8]

    ext = Path(file.filename).suffix.lower() if file.filename else ".wav"
    if ext not in (".wav", ".mp3", ".ogg", ".flac", ".webm"):
        ext = ".wav"

    voice_file = VOICES_DIR / "custom" / f"{safe_name}{ext}"
    voice_file.parent.mkdir(parents=True, exist_ok=True)
    with open(voice_file, "wb") as f:
        f.write(content)

    return {
        "status": "ok",
        "voice": {
            "id": safe_name,
            "name": name.strip(),
            "category": "custom",
            "file": voice_file.name,
            "voice_key": f"custom/{voice_file.name}",
            "size_kb": round(len(content) / 1024, 1),
        },
    }


@app.delete("/api/voices/{voice_id}")
async def delete_voice(voice_id: str):
    """Delete a custom voice."""
    folder = VOICES_DIR / "custom"
    for f in folder.iterdir():
        if f.stem == voice_id:
            f.unlink()
            return {"status": "deleted", "id": voice_id}
    raise HTTPException(404, "Voice not found")


@app.get("/api/voices/{category}/{filename}")
async def get_voice_audio(category: str, filename: str):
    """Stream a voice reference audio file for preview."""
    if category not in ("system", "custom"):
        raise HTTPException(400, "Invalid category")
    path = VOICES_DIR / category / filename
    if not path.exists():
        raise HTTPException(404, "Voice file not found")
    return FileResponse(path, media_type="audio/wav")


@app.post("/api/tts/generate")
async def tts_generate(request: Request):
    """Proxy TTS generation to the Chatterbox service and return audio."""
    body = await request.json()
    text = body.get("text", "").strip()
    voice_key = body.get("voice_key", "").strip()
    temperature = body.get("temperature", 0.8)
    top_p = body.get("top_p", 0.95)
    top_k = body.get("top_k", 1000)
    repetition_penalty = body.get("repetition_penalty", 1.2)

    if not text:
        raise HTTPException(400, "Text is required")
    if not voice_key:
        raise HTTPException(400, "voice_key is required")
    if len(text) > 5000:
        raise HTTPException(400, "Text too long (max 5000 chars)")

    tts_url = f"{settings.chatterbox_api_url}/generate"
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(
                tts_url,
                json={
                    "prompt": text,
                    "voice_key": voice_key,
                    "temperature": temperature,
                    "top_p": top_p,
                    "top_k": top_k,
                    "repetition_penalty": repetition_penalty,
                    "norm_loudness": True,
                },
                headers={"x-api-key": settings.chatterbox_api_key},
            )
        if resp.status_code != 200:
            detail = resp.text[:500]
            raise HTTPException(resp.status_code, f"TTS service error: {detail}")

        return StreamingResponse(
            iter([resp.content]),
            media_type="audio/wav",
            headers={"Content-Disposition": "inline; filename=generated.wav"},
        )
    except httpx.ConnectError:
        raise HTTPException(503, "TTS service not reachable. Is the tts container running on port 8080?")
    except httpx.TimeoutException:
        raise HTTPException(504, "TTS generation timed out. Try shorter text.")


@app.get("/api/tts/health")
async def tts_health():
    """Check if the Chatterbox TTS service is running."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{settings.chatterbox_api_url}/health")
        return resp.json()
    except Exception as e:
        return {"status": "unreachable", "error": str(e)}


# ── Vapi Custom Voice endpoint ────────────────────────────────

def _wav_to_pcm(wav_bytes: bytes, target_rate: int) -> bytes:
    """Convert a 16-bit WAV blob to raw mono PCM (s16le) at target_rate.

    Vapi's custom-voice provider expects raw little-endian 16-bit PCM, mono,
    at the sample rate it requested. Chatterbox returns a WAV at its own rate,
    so we down/up-sample and force mono here using only the stdlib.
    """
    import audioop
    import wave

    with wave.open(io.BytesIO(wav_bytes), "rb") as w:
        n_channels = w.getnchannels()
        sampwidth = w.getsampwidth()
        src_rate = w.getframerate()
        frames = w.readframes(w.getnframes())

    # Force 16-bit samples
    if sampwidth != 2:
        frames = audioop.lin2lin(frames, sampwidth, 2)
        sampwidth = 2

    # Force mono
    if n_channels == 2:
        frames = audioop.tomono(frames, sampwidth, 0.5, 0.5)

    # Resample to the rate Vapi asked for
    if src_rate != target_rate:
        frames, _ = audioop.ratecv(frames, sampwidth, 1, src_rate, target_rate, None)

    return frames


@app.post("/vapi/voice")
async def vapi_custom_voice(request: Request):
    """Vapi Custom Voice provider endpoint.

    Vapi POSTs a `voice-request` with the text to synthesize and a target
    sample rate. We generate speech via Chatterbox in the configured cloned
    voice and stream back raw PCM (s16le, mono) at the requested rate.
    """
    body = await request.json()
    message = body.get("message", body)
    text = (message.get("text") or "").strip()
    sample_rate = int(message.get("sampleRate") or message.get("sample_rate") or 24000)

    if not text:
        raise HTTPException(400, "No text in voice-request")

    voice_key = settings.vapi_voice_key.strip()
    if not voice_key:
        raise HTTPException(
            500,
            "VAPI_VOICE_KEY is not set. Add e.g. VAPI_VOICE_KEY=custom/your-voice.wav to .env",
        )

    tts_url = f"{settings.chatterbox_api_url}/generate"
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(
                tts_url,
                json={
                    "prompt": text,
                    "voice_key": voice_key,
                    "norm_loudness": True,
                },
                headers={"x-api-key": settings.chatterbox_api_key},
            )
        if resp.status_code != 200:
            raise HTTPException(resp.status_code, f"TTS service error: {resp.text[:300]}")

        pcm = _wav_to_pcm(resp.content, sample_rate)
        return StreamingResponse(
            iter([pcm]),
            media_type="application/octet-stream",
        )
    except httpx.ConnectError:
        raise HTTPException(503, "TTS service not reachable on the Chatterbox port.")
    except httpx.TimeoutException:
        raise HTTPException(504, "TTS generation timed out.")


# ── Browser demo UI ───────────────────────────────────────────

@app.get("/")
async def browser_demo(request: Request):
    """Browser demo UI — served from app/templates/index.html."""
    vapi_pub   = settings.vapi_public_key or ""
    vapi_asst  = settings.vapi_assistant_id or ""
    vapi_ready = bool(vapi_pub and vapi_pub != "your_vapi_public_key_here")
    retell_ready = bool(settings.retell_api_key and settings.retell_agent_id)
    return templates.TemplateResponse("index.html", {
        "request":      request,
        "vapi_pub":     vapi_pub,
        "vapi_asst":    vapi_asst,
        "restaurant":   settings.restaurant_name,
        "vapi_ready":   vapi_ready,
        "retell_ready": retell_ready,
    })
