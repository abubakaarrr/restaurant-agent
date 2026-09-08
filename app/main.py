"""FastAPI entry point for managed voice tools, webhooks, and operator UI."""

from __future__ import annotations

import hmac
import json
import logging
import io
import secrets
import traceback
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from app.logging_config import configure_logging

configure_logging()
logger = logging.getLogger(__name__)
import httpx
from fastapi import FastAPI, File, Form, HTTPException, Request, Security, UploadFile, WebSocket
from fastapi.responses import FileResponse, RedirectResponse, Response, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from starlette.middleware.sessions import SessionMiddleware

from app.agent.runner import clear_session as _clear_session, get_session_history, run_agent, stream_agent_tokens
from app.call_analytics import ingest_retell_webhook, purge_expired_call_data
from app.config import settings
from app.db_pool import get_pool, close_pool
from app.rate_limit import chat_limiter, login_limiter, web_call_limiter
from app.restaurant_settings import (
    load_restaurant_settings as _load_settings,
    save_restaurant_settings as _save_settings,
    validate_restaurant_settings_update,
)
from app.retell_handler import handle_retell_connection
from app.retell_ws_auth import remember_retell_call, retell_ws_authorized
from app.security import constant_time_equal, verify_retell_webhook_signature
from app.services.restaurant import RestaurantServiceError, restaurant_service
from app.tool_api import router as voice_tool_router

# ── Auth helpers ──────────────────────────────────────────────

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def require_dashboard_access(
    request: Request,
    api_key: str | None = Security(_api_key_header),
) -> None:
    """Allow a server API key or a logged-in dashboard session with CSRF."""
    if settings.dashboard_api_key and constant_time_equal(api_key, settings.dashboard_api_key):
        return
    if not _is_logged_in(request):
        if not settings.dashboard_api_key and not settings.is_production:
            return
        raise HTTPException(status_code=401, detail="Dashboard authentication required")
    if request.method not in {"GET", "HEAD", "OPTIONS"}:
        provided = request.headers.get("X-CSRF-Token")
        expected = str(request.session.get("csrf_token") or "")
        if not constant_time_equal(provided, expected):
            raise HTTPException(status_code=403, detail="Invalid or missing CSRF token")


def _require_vapi_access(request: Request) -> None:
    if not settings.enable_legacy_vapi:
        raise HTTPException(status_code=404, detail="Legacy Vapi adapter is disabled")
    if not settings.vapi_server_secret:
        raise HTTPException(status_code=503, detail="Vapi server secret is not configured")
    provided = request.headers.get("x-vapi-secret", "")
    if not provided:
        authorization = request.headers.get("authorization", "")
        if authorization.lower().startswith("bearer "):
            provided = authorization[7:]
    if not constant_time_equal(provided, settings.vapi_server_secret):
        raise HTTPException(status_code=401, detail="Invalid Vapi server secret")


def _is_logged_in(request: Request) -> bool:
    return bool(request.session.get("authenticated"))


def _check_login(username: str, password: str) -> bool:
    """Constant-time-ish compare for the single shared dashboard login."""
    u_ok = hmac.compare_digest(username, settings.login_username) if len(username) == len(settings.login_username) else False
    p_ok = hmac.compare_digest(password, settings.login_password) if len(password) == len(settings.login_password) else False
    return u_ok and p_ok


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
    settings.validate_runtime_security()
    # Warm the connection pool at startup so the first call doesn't pay the
    # connection-setup cost mid-conversation.
    await get_pool()
    try:
        await purge_expired_call_data()
    except Exception:
        # Existing deployments must apply the pilot migration first. Keep
        # startup available for the health endpoint while making the gap clear.
        logger.warning("Call-data retention cleanup skipped; apply DB migrations", exc_info=True)
    yield
    await close_pool()


app = FastAPI(
    title="Restaurant AI Receptionist",
    description="Managed voice tools, call operations, and restaurant dashboard",
    version="0.2.0",
    lifespan=lifespan,
    docs_url=None if settings.is_production else "/docs",
    redoc_url=None if settings.is_production else "/redoc",
    openapi_url=None if settings.is_production else "/openapi.json",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=[
        "Authorization",
        "Content-Type",
        "Idempotency-Key",
        "X-API-Key",
        "X-CSRF-Token",
        "X-Voice-Tool-Secret",
    ],
    allow_credentials=True,
)

# Session cookie for the single shared dashboard login.
# Must be added after CORS so the session is available on every request.
_session_secret = (
    settings.session_secret
    or settings.dashboard_api_key
    or "dev-session-secret-change-me"
)
app.add_middleware(
    SessionMiddleware,
    secret_key=_session_secret,
    session_cookie="restaurant_session",
    max_age=60 * 60 * 24 * 7,  # 7 days
    same_site="lax",
    https_only=settings.app_env == "production",
)

app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")
app.include_router(voice_tool_router)

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
    pool = await get_pool()
    async with pool.acquire() as conn:
        database_ok = await conn.fetchval("SELECT 1")
        pilot_schema = await conn.fetchval(
            "SELECT to_regclass('public.voice_action_idempotency') IS NOT NULL"
        )
    if database_ok != 1 or not pilot_schema:
        raise HTTPException(status_code=503, detail="Database migration required")
    runtime = _load_settings()
    return {
        "status": "ok",
        "restaurant": runtime["restaurant_name"],
        "agent_name": runtime.get("ai_agent_name") or settings.ai_agent_name,
        "voice_live_writes_enabled": settings.voice_live_writes_enabled,
        "managed_retell_ready": bool(
            settings.retell_agent_id and settings.voice_tool_secret
        ),
    }


# ── Text chat endpoint (browser demo + testing) ───────────────

@app.post(
    "/chat",
    response_model=ChatResponse,
    dependencies=[Security(require_dashboard_access)],
)
async def chat(req: ChatRequest, request: Request):
    """Text chat endpoint — used by the browser demo and simulate_call.py."""
    await chat_limiter.check(request, scope="chat")
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


@app.delete(
    "/session/{session_id}",
    dependencies=[Security(require_dashboard_access)],
)
async def delete_session(session_id: str):
    _clear_session(session_id)
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
    _require_vapi_access(request)
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
            runtime = _load_settings()
            restaurant_name = runtime["restaurant_name"]
            agent_name = runtime.get("ai_agent_name") or settings.ai_agent_name
            greeting = (
                f"Hi, you've reached {restaurant_name}. This is {agent_name}. "
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
    """Vapi server webhook — receives call lifecycle events.

    Rollback only. Configure VAPI_SERVER_SECRET as Vapi's server-URL secret.
    """
    _require_vapi_access(request)

    body = await request.json()
    msg = body.get("message", {})
    event_type = msg.get("type", "unknown")

    if event_type == "end-of-call-report":
        call_id = msg.get("call", {}).get("id", "")
        _clear_session(call_id)

    return {"status": "ok"}


# ── Retell Custom LLM WebSocket ───────────────────────────────

@app.websocket("/retell-ws/{call_id}")
async def retell_ws(websocket: WebSocket, call_id: str):
    """Retell connects here for each call. Retell does STT/turn-taking/TTS;
    we run the LangGraph agent and stream text replies back."""
    # Accept first. Closing an unaccepted socket is returned as HTTP 403, which
    # Retell retries without a useful close reason.
    await websocket.accept()
    if not settings.enable_legacy_retell_custom_llm:
        logger.warning("Retell WS rejected for %s: custom-LLM adapter disabled", call_id)
        await websocket.close(code=1008, reason="Legacy custom-LLM adapter is disabled")
        return
    provided = websocket.query_params.get("token", "")
    if not retell_ws_authorized(call_id, provided):
        logger.warning(
            "Retell WS rejected for %s: add ?token=RETELL_WS_TOKEN to the custom LLM URL "
            "in the Retell dashboard, or start the call from this app so the call id is minted",
            call_id,
        )
        await websocket.close(code=1008, reason="Invalid WebSocket token")
        return
    await handle_retell_connection(websocket, call_id)


@app.post(
    "/api/retell/web-call",
    dependencies=[Security(require_dashboard_access)],
)
async def retell_web_call(request: Request):
    """Mint a short-lived Retell web-call access token for the browser SDK.

    The API key stays server-side; the browser only ever sees the access token,
    which Retell invalidates after 30s if a call isn't started.
    """
    await web_call_limiter.check(request, scope="retell-web-call")
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
    call_id = str(data.get("call_id") or "")
    remember_retell_call(call_id)
    return {
        "access_token": data.get("access_token", ""),
        "call_id": call_id,
    }


@app.post("/api/retell/webhook")
async def retell_webhook(request: Request):
    """Receive signed, replay-protected Retell call lifecycle events."""
    raw_body = await request.body()
    signature = request.headers.get("X-Retell-Signature")
    if not verify_retell_webhook_signature(
        raw_body,
        settings.retell_api_key,
        signature,
    ):
        raise HTTPException(status_code=401, detail="Invalid Retell webhook signature")
    try:
        payload = json.loads(raw_body)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail="Invalid JSON body") from exc
    await ingest_retell_webhook(payload, raw_body)
    event_type = payload.get("event") or payload.get("event_type")
    call = payload.get("call") if isinstance(payload.get("call"), dict) else {}
    call_id = call.get("call_id") or call.get("id")
    if event_type in {"call_ended", "call_analyzed"} and call_id:
        _clear_session(str(call_id))
    return Response(status_code=204)


# ── Voice Studio API ──────────────────────────────────────────

@app.get("/api/voices", dependencies=[Security(require_dashboard_access)])
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


@app.post("/api/voices/upload", dependencies=[Security(require_dashboard_access)])
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


@app.delete("/api/voices/{voice_id}", dependencies=[Security(require_dashboard_access)])
async def delete_voice(voice_id: str):
    """Delete a custom voice."""
    folder = VOICES_DIR / "custom"
    for f in folder.iterdir():
        if f.stem == voice_id:
            f.unlink()
            return {"status": "deleted", "id": voice_id}
    raise HTTPException(404, "Voice not found")


@app.get("/api/voices/{category}/{filename}", dependencies=[Security(require_dashboard_access)])
async def get_voice_audio(category: str, filename: str):
    """Stream a voice reference audio file for preview."""
    if category not in ("system", "custom"):
        raise HTTPException(400, "Invalid category")
    path = VOICES_DIR / category / filename
    if not path.exists():
        raise HTTPException(404, "Voice file not found")
    return FileResponse(path, media_type="audio/wav")


@app.post("/api/tts/generate", dependencies=[Security(require_dashboard_access)])
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


@app.get("/api/tts/health", dependencies=[Security(require_dashboard_access)])
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
    _require_vapi_access(request)
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


# ── History API ───────────────────────────────────────────────

@app.get("/api/history", dependencies=[Security(require_dashboard_access)])
async def get_history():
    """Return all reservations (with table + any pre-order) and standalone
    pickup orders, for the History dashboard tab."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        booking_rows = await conn.fetch(
            """
            SELECT b.id, b.customer_name, b.customer_phone, b.party_size,
                   b.booked_at, b.status, b.cancellation_reason, b.notes, b.created_at,
                   t.table_number, t.location
            FROM bookings b
            LEFT JOIN tables t ON t.id = b.table_id
            ORDER BY b.created_at DESC
            """
        )
        order_rows = await conn.fetch(
            """
            SELECT id, session_id, booking_id, customer_name, customer_phone,
                   status, total_amount, created_at
            FROM orders
            ORDER BY created_at DESC
            """
        )
        item_rows = await conn.fetch(
            """
            SELECT order_id, item_name, quantity, unit_price, subtotal, notes
            FROM order_items
            ORDER BY id
            """
        )

    # Group order items by their order id
    items_by_order: dict[int, list[dict]] = {}
    for it in item_rows:
        items_by_order.setdefault(it["order_id"], []).append({
            "item_name": it["item_name"],
            "quantity": it["quantity"],
            "unit_price": float(it["unit_price"]),
            "subtotal": float(it["subtotal"]),
            "notes": it["notes"] or "",
        })

    def serialize_order(o) -> dict:
        return {
            "order_id": o["id"],
            "customer_name": o["customer_name"] or "",
            "customer_phone": o["customer_phone"] or "",
            "status": o["status"],
            "total_amount": float(o["total_amount"]),
            "created_at": o["created_at"].isoformat() if o["created_at"] else None,
            "items": items_by_order.get(o["id"], []),
        }

    # Index pre-orders by the booking they belong to
    orders_by_booking: dict[int, dict] = {}
    pickup_orders: list[dict] = []
    for o in order_rows:
        if o["booking_id"]:
            orders_by_booking[o["booking_id"]] = serialize_order(o)
        else:
            pickup_orders.append(serialize_order(o))

    reservations = []
    for b in booking_rows:
        reservations.append({
            "booking_id": b["id"],
            "customer_name": b["customer_name"],
            "customer_phone": b["customer_phone"] or "",
            "party_size": b["party_size"],
            "booked_at": b["booked_at"].isoformat() if b["booked_at"] else None,
            "status": b["status"],
            "cancellation_reason": b["cancellation_reason"] or "",
            "notes": b["notes"] or "",
            "created_at": b["created_at"].isoformat() if b["created_at"] else None,
            "table_number": b["table_number"],
            "location": b["location"] or "",
            "preorder": orders_by_booking.get(b["id"]),
        })

    return {"reservations": reservations, "pickup_orders": pickup_orders}


# ── Settings API ──────────────────────────────────────────


@app.get("/api/settings", dependencies=[Security(require_dashboard_access)])
async def get_settings_api():
    return _load_settings()


@app.put("/api/settings", dependencies=[Security(require_dashboard_access)])
async def update_settings(request: Request):
    body = await request.json()
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="Settings body must be an object")
    try:
        validated = validate_restaurant_settings_update(body)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    current = _load_settings()
    current.update(validated)
    _save_settings(current)
    return {"status": "ok", "settings": current}


# ── Menu API ──────────────────────────────────────────────

@app.get("/api/menu", dependencies=[Security(require_dashboard_access)])
async def get_menu():
    """Return all menu items grouped by category."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT id, name, category, price, description, dietary, available,
                   COALESCE(price_estimated, FALSE) AS price_estimated
            FROM menu_items
            ORDER BY
                CASE category
                    WHEN 'starter' THEN 1
                    WHEN 'main'    THEN 2
                    WHEN 'dessert' THEN 3
                    WHEN 'drink'   THEN 4
                    WHEN 'special' THEN 5
                    ELSE 6
                END,
                name
            """
        )
    categories: dict[str, list[dict]] = {}
    for r in rows:
        item = {
            "id": r["id"],
            "name": r["name"],
            "price": float(r["price"]),
            "description": r["description"] or "",
            "dietary": list(r["dietary"]) if r["dietary"] else [],
            "available": r["available"],
            "price_estimated": bool(r["price_estimated"]),
        }
        categories.setdefault(r["category"], []).append(item)
    return {"categories": categories}


@app.post("/api/menu", dependencies=[Security(require_dashboard_access)])
async def create_menu_item(request: Request):
    """Add a new menu item."""
    body = await request.json()
    name = (body.get("name") or "").strip()
    category = (body.get("category") or "main").strip()
    price = float(body.get("price", 0))
    description = (body.get("description") or "").strip()
    dietary = body.get("dietary", [])

    if not name:
        raise HTTPException(400, "Item name is required")
    if price <= 0:
        raise HTTPException(400, "Price must be greater than 0")
    if category not in ("starter", "main", "dessert", "drink", "special"):
        raise HTTPException(400, "Invalid category")

    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            INSERT INTO menu_items (name, category, price, description, dietary, available)
            VALUES ($1, $2, $3, $4, $5, TRUE)
            RETURNING id
            """,
            name, category, price, description, dietary,
        )
    return {
        "status": "ok",
        "item": {
            "id": row["id"],
            "name": name,
            "category": category,
            "price": price,
            "description": description,
            "dietary": dietary,
            "available": True,
        },
    }


@app.patch("/api/menu/{item_id}", dependencies=[Security(require_dashboard_access)])
async def update_menu_item(item_id: int, request: Request):
    """Toggle menu item availability or update fields."""
    body = await request.json()
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow("SELECT id FROM menu_items WHERE id = $1", item_id)
        if not row:
            raise HTTPException(404, "Menu item not found")

        if "available" in body:
            await conn.execute(
                "UPDATE menu_items SET available = $1 WHERE id = $2",
                bool(body["available"]), item_id,
            )

    return {"status": "ok", "id": item_id}


# ── Knowledge loop ────────────────────────────────────────


@app.get("/api/knowledge/gaps", dependencies=[Security(require_dashboard_access)])
async def list_knowledge_gaps():
    try:
        return await restaurant_service.list_knowledge_gaps()
    except RestaurantServiceError as error:
        raise HTTPException(status_code=error.status, detail=error.message) from error


@app.post(
    "/api/knowledge/gaps/{gap_id}/resolve",
    dependencies=[Security(require_dashboard_access)],
)
async def resolve_knowledge_gap(gap_id: int, request: Request):
    body = await request.json()
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="Body must be an object")
    answer = str(body.get("answer") or "")
    username = str(request.session.get("username") or "admin")
    try:
        return await restaurant_service.resolve_knowledge_gap(
            gap_id,
            answer=answer,
            resolved_by=username,
        )
    except RestaurantServiceError as error:
        raise HTTPException(status_code=error.status, detail=error.message) from error


# ── Stats API ─────────────────────────────────────────────

@app.get("/api/stats", dependencies=[Security(require_dashboard_access)])
async def get_stats():
    """Aggregated dashboard stats."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        res_count = await conn.fetchval("SELECT COUNT(*) FROM bookings")
        order_count = await conn.fetchval("SELECT COUNT(*) FROM orders")
        revenue = await conn.fetchval(
            "SELECT COALESCE(SUM(total_amount), 0) FROM orders WHERE status = 'confirmed'"
        )
    return {
        "total_reservations": res_count,
        "total_orders": order_count,
        "total_revenue": float(revenue),
    }


@app.get("/api/voice/metrics", dependencies=[Security(require_dashboard_access)])
async def get_voice_metrics(days: int = 7):
    """Return pilot call outcomes and locally measured protocol timings."""
    days = min(max(days, 1), 90)
    pool = await get_pool()
    async with pool.acquire() as conn:
        summary = await conn.fetchrow(
            """
            SELECT
                COUNT(DISTINCT session_id) AS calls,
                COUNT(*) FILTER (WHERE ended_at IS NOT NULL) AS ended_calls
            FROM call_sessions
            WHERE started_at >= NOW() - ($1::int * interval '1 day')
            """,
            days,
        )
        timing = await conn.fetchrow(
            """
            SELECT
                percentile_cont(0.5) WITHIN GROUP (ORDER BY duration_ms)
                    FILTER (WHERE event_type = 'first_response_chunk') AS first_chunk_p50_ms,
                percentile_cont(0.95) WITHIN GROUP (ORDER BY duration_ms)
                    FILTER (WHERE event_type = 'first_response_chunk') AS first_chunk_p95_ms,
                percentile_cont(0.95) WITHIN GROUP (ORDER BY duration_ms)
                    FILTER (WHERE event_type = 'generation_cancelled') AS cancel_p95_ms,
                COUNT(*) FILTER (WHERE event_type LIKE '%error%') AS errors
            FROM call_events
            WHERE created_at >= NOW() - ($1::int * interval '1 day')
            """,
            days,
        )
        recent = await conn.fetch(
            """
            SELECT call_id, event_type, duration_ms, created_at
            FROM call_events
            WHERE created_at >= NOW() - ($1::int * interval '1 day')
            ORDER BY created_at DESC
            LIMIT 100
            """,
            days,
        )
    return {
        "window_days": days,
        "calls": int(summary["calls"] or 0),
        "ended_calls": int(summary["ended_calls"] or 0),
        "first_chunk_p50_ms": (
            float(timing["first_chunk_p50_ms"])
            if timing["first_chunk_p50_ms"] is not None
            else None
        ),
        "first_chunk_p95_ms": (
            float(timing["first_chunk_p95_ms"])
            if timing["first_chunk_p95_ms"] is not None
            else None
        ),
        "generation_cancel_p95_ms": (
            float(timing["cancel_p95_ms"])
            if timing["cancel_p95_ms"] is not None
            else None
        ),
        "errors": int(timing["errors"] or 0),
        "recent_events": [
            {
                "call_id": row["call_id"],
                "event_type": row["event_type"],
                "duration_ms": row["duration_ms"],
                "created_at": row["created_at"].isoformat(),
            }
            for row in recent
        ],
    }


# ── Browser demo UI ───────────────────────────────────────────

@app.get("/login")
async def login_page(request: Request):
    """Single shared login page for the restaurant dashboard."""
    if _is_logged_in(request):
        return RedirectResponse("/", status_code=302)
    return templates.TemplateResponse(
        request,
        "login.html",
        {"restaurant": _load_settings()["restaurant_name"], "error": None},
    )


@app.post("/login")
async def login_submit(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
):
    await login_limiter.check(request, scope="dashboard-login")
    if _check_login(username.strip(), password):
        request.session["authenticated"] = True
        request.session["username"] = username.strip()
        request.session["csrf_token"] = secrets.token_urlsafe(32)
        return RedirectResponse("/", status_code=302)

    return templates.TemplateResponse(
        request,
        "login.html",
        {
            "restaurant": _load_settings()["restaurant_name"],
            "error": "Invalid username or password",
        },
        status_code=401,
    )


@app.post("/logout")
async def logout(request: Request):
    request.session.clear()
    return RedirectResponse("/login", status_code=302)


@app.get("/")
async def browser_demo(request: Request):
    """Browser demo UI — served from app/templates/index.html."""
    if not _is_logged_in(request):
        return RedirectResponse("/login", status_code=302)

    runtime = _load_settings()
    restaurant_name = runtime["restaurant_name"]
    agent_name = runtime.get("ai_agent_name") or settings.ai_agent_name
    vapi_pub = settings.vapi_public_key or ""
    vapi_asst = settings.vapi_assistant_id or ""
    vapi_ready = bool(
        settings.enable_legacy_vapi
        and vapi_pub
        and vapi_pub != "your_vapi_public_key_here"
    )
    retell_ready = bool(settings.retell_api_key and settings.retell_agent_id)
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "vapi_pub": vapi_pub,
            "vapi_asst": vapi_asst,
            "restaurant": restaurant_name,
            "agent_name": agent_name,
            "vapi_ready": vapi_ready,
            "retell_ready": retell_ready,
            "csrf_token": request.session.get("csrf_token", ""),
            "username": request.session.get("username", ""),
        },
    )


@app.get("/widget-demo")
async def widget_demo(request: Request):
    """Preview the same public Retell widget clients embed on their websites."""
    if not settings.widget_enabled:
        raise HTTPException(status_code=404, detail="Website widget is disabled")
    return templates.TemplateResponse(
        request,
        "widget_demo.html",
        {
            "restaurant": _load_settings()["restaurant_name"],
            "widget_mode": settings.widget_mode,
            "retell_public_key": settings.retell_public_key,
            "retell_voice_agent_id": settings.retell_agent_id,
            "retell_chat_agent_id": settings.retell_chat_agent_id,
            "retell_callback_phone_number": settings.retell_phone_number,
            "widget_title": settings.widget_title,
            "widget_logo_url": settings.widget_logo_url,
            "widget_color": settings.widget_color,
            "widget_fab_text": settings.widget_fab_text,
            "callback_countries": settings.callback_countries,
            "callback_terms_url": settings.callback_terms_url,
            "recaptcha_site_key": settings.recaptcha_site_key,
        },
    )
