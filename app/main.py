"""FastAPI entry point — chat API + Vapi Custom LLM + browser demo UI."""

from __future__ import annotations

import asyncio
import json
import logging
import traceback
import uuid

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from app.agent.graph import restaurant_agent
from app.config import settings

app = FastAPI(
    title="Restaurant AI Receptionist",
    description="La Casa Restaurant — AI receptionist powered by LangGraph + GPT-4o",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")

# ── In-memory session store ───────────────────────────────────
_sessions: dict[str, list[dict]] = {}


# ── Shared agent runner ───────────────────────────────────────

async def _run_agent(session_id: str, user_message: str, caller_phone: str = "") -> str:
    """Run one agent turn and return the text reply."""
    history = _sessions.get(session_id, [])
    history.append({"role": "user", "content": user_message})

    result = await restaurant_agent.ainvoke(
        {
            "messages": history,
            "session_id": session_id,
            "caller_phone": caller_phone,
            "turn_count": len(history) // 2,
        },
        config={"configurable": {"restaurant_name": settings.restaurant_name}},
    )

    result_messages = result.get("messages", [])
    reply = ""
    for msg in reversed(result_messages):
        if hasattr(msg, "type") and msg.type == "ai":
            content = msg.content
            if content and isinstance(content, str) and content.strip():
                reply = content.strip()
                break
            elif isinstance(content, list):
                text_parts = [p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"]
                if text_parts:
                    reply = " ".join(text_parts).strip()
                    break

    if not reply:
        reply = "I'm sorry, could you repeat that?"

    history.append({"role": "assistant", "content": reply})
    _sessions[session_id] = history[-20:]
    return reply


# ── Models ────────────────────────────────────────────────────

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
        reply = await _run_agent(session_id, req.message, req.caller_phone)
    except Exception as e:
        logger.error("Agent error:\n%s", traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

    return ChatResponse(
        reply=reply,
        session_id=session_id,
        turn_count=len(_sessions.get(session_id, [])) // 2,
    )


@app.delete("/session/{session_id}")
async def clear_session(session_id: str):
    _sessions.pop(session_id, None)
    return {"cleared": session_id}


# ── Vapi Custom LLM endpoint ──────────────────────────────────

@app.post("/vapi/llm/chat/completions")
async def vapi_llm(request: Request):
    """
    OpenAI-compatible streaming endpoint for Vapi's Custom LLM feature.
    Vapi sends conversation messages → we run the LangGraph agent → stream reply back.
    """
    body = await request.json()

    messages: list[dict] = body.get("messages", [])
    call_info: dict = body.get("call", {})
    call_id: str = call_info.get("id", str(uuid.uuid4()))
    caller_number: str = call_info.get("customer", {}).get("number", "")

    user_messages = [m for m in messages if m.get("role") == "user"]
    if not user_messages:
        reply = f"Hello! Thank you for calling {settings.restaurant_name}. How can I help you today?"
    else:
        latest_user_msg = user_messages[-1]["content"]
        try:
            reply = await _run_agent(call_id, latest_user_msg, caller_number)
        except Exception:
            reply = "I'm sorry, I had a technical issue. Could you please repeat that?"

    async def sse_stream(text: str):
        """Stream reply word-by-word in OpenAI SSE format."""
        chunk_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        words = text.split(" ")

        first = {
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}],
        }
        yield f"data: {json.dumps(first)}\n\n"

        for i, word in enumerate(words):
            content = word if i == 0 else f" {word}"
            chunk = {
                "id": chunk_id,
                "object": "chat.completion.chunk",
                "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": None}],
            }
            yield f"data: {json.dumps(chunk)}\n\n"
            await asyncio.sleep(0.015)

        final = {
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
        yield f"data: {json.dumps(final)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        sse_stream(reply),
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
        _sessions.pop(call_id, None)

    return {"status": "ok"}


# ── Browser demo UI ───────────────────────────────────────────

@app.get("/")
async def browser_demo(request: Request):
    """Browser demo UI — served from app/templates/index.html."""
    vapi_pub   = settings.vapi_public_key or ""
    vapi_asst  = settings.vapi_assistant_id or ""
    vapi_ready = bool(vapi_pub and vapi_pub != "your_vapi_public_key_here")
    return templates.TemplateResponse("index.html", {
        "request":    request,
        "vapi_pub":   vapi_pub,
        "vapi_asst":  vapi_asst,
        "restaurant": settings.restaurant_name,
        "vapi_ready": vapi_ready,
    })
