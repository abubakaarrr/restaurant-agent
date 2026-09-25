"""Gemini native audio session with server-owned restaurant operations.

Gemini handles live listening and conversation. A verified booking approval can
use a short server-authored OpenAI speech response when model tool choice is
unreliable. Captions are Gemini Live events except for that transaction reply.
"""
from __future__ import annotations
import asyncio
import base64
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import replace
from datetime import date
import json
import logging
import os
import re
import time
import uuid

from app.native_voice.adapter import NativeVoiceAdapter
from app.native_voice.protocol import MemoryRealtimeTransport
from app.native_voice.streaming import StreamingVoiceSession, CONVERSATION_RULES, ORDER_CHANGES, model_tools, restaurant_information, menu_answer, menu_suggestions, state_view
from app.pending_confirmation import active_released_confirmation, classify_affirmation, clear_pending_confirmation, requests_order_abandonment

from app.native_voice.quantity import party_count, item_count, wants_review

MODEL = "gemini-3.8-live"
ENDPOINT = "wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent"


def native_setup(domain, pace):
    from app.services.restaurant import _restaurant_now
    instructions = CONVERSATION_RULES + """
You listen and speak natively in this one Gemini Live session. Input/output
captions are for display; do not invent or change names, quantities or dishes.
For an unclear name ask how to spell it. Never replace an unknown dish with a
different menu item. Ask a specific question if the requested dish is not found.
Number prefixes on dishes are quantities: a couple/pair means two, half a dozen
means six. Two more increases an existing quantity by two. The second item
selects a line, not a quantity. Do not confuse counts with phone numbers or times.
Ask briefly if a count is unclear.
During collection or correction, address ONLY the changed detail and ask the
next missing question. Do not repeat the full order or reservation. Give a full
order readback when the caller asks for a summary or says that is all / ready
to place the order. For confirmed reservation changes, send ONLY changed fields
to update_confirmed_booking, never reconstruct the whole booking or create a
duplicate. The server speaks amendment approval prompts; do not repeat them.
Keep name and phone as separate fields. A name cannot contain a phone number.
After the caller spells a name, collapse the letters into one name, repeat the
result once, and ask whether it is correct. Never ask the same spelling question
more than twice. Preserve an already accepted name or phone unless the caller
explicitly corrects it and then confirms the old-to-new change.
Wait for the caller to finish a multi-part request; retain alternatives and
pre-orders while collecting missing details. Never interrupt their list.
Check requested pre-order dishes against menu tools before promising a pre-order.
For a reservation with food, call apply_order_changes with all requested items
before requesting table approval. The server retains them until the table exists.
Explain that table and food have separate confirmations. Never lose the food
request or ask the caller to repeat already captured items.
If a dish is unavailable, retain the reservation details, explain which dish
is unavailable, and ask the caller what they prefer. Never silently substitute.
Always speak English, including after accented English, Hindi/Urdu discourse
markers, or captions rendered in another script. Do not imitate their language.
If unclear, ask a short clarification in English. Read phone numbers digit by
digit in groups. Double means two copies and triple/treble means three copies.
An unclear repeat word requires clarification; never guess or drop a digit.
For menu recommendations always call get_full_menu or suggest_menu_for_preference.
Recommend ONLY items whose available flag is true for menu_checked_for. A catalog
entry is not proof of availability. Explain service-time restrictions accurately;
never describe closed service hours as every item being sold out.
Use a warm, concise natural reply. Never say 'Your request was applied'.
Tool status describes the database, not proof that you understood the caller.
When a tool returns readback_text, say that entire text EXACTLY, including the
question, then wait for the caller's later approval. Do not paraphrase it.
For EVERY reservation readback, call get_reservation_draft first, even if you
remember every detail. A summary from conversation memory cannot authorize a
booking. A draft is only a proposal: say 'requested reservation', not 'you have
a reservation'. Use get_order_summary for every order readback for the same reason.
Only say booked/confirmed after a successful create_booking/confirm_order result.
Do not say 'Thanks, <name>' unless that is the name the caller actually provided.
Do not greet yourself or treat your own speech as the caller's request.
When a confirmed reservation exists in this call, keep using that booking. For a
party-size or time change, check availability for the proposed slot and then
call update_confirmed_booking with caller_confirmed=false. Read back the
proposal, wait for a NEW caller yes, and call update_confirmed_booking with
caller_confirmed=true. Never call create_booking or update_reservation_draft
for a confirmed reservation. The server owns the booking reference.
For a confirmed dine-in pre-order, preserve every existing line and propose
only the requested change through apply_order_changes. Wait for a later yes
before applying it. Never add a second order to imitate an amendment.
Treat dietary and child requests as conversational context, not isolated tags.
If a guest is vegetarian, preserve who the note is about and offer a menu
recommendation; use suggest_menu_for_preference before naming dishes. If the
caller asks for a high chair without a quantity, ask how many are needed and
how many children are joining. High chairs are limited, not guaranteed.
Do not claim there is a play area: no verified play-area fact is available.
When a check may take time, briefly tell the caller what you are checking
before calling the tool. Do not claim the check succeeded until its result.
"""
    instructions += "\nRestaurant local date/time: " + _restaurant_now().isoformat()
    instructions += "\nInitial application order: " + json.dumps(state_view(domain.state), default=str)
    functions = [{"name": t["name"], "description": t["description"], "parametersJsonSchema": t["parameters"], "behavior": "BLOCKING"} for t in model_tools()]
    return {"setup": {
        "model": "models/" + MODEL,
        "generationConfig": {"responseModalities": ["AUDIO"], "speechConfig": {"voiceConfig": {"prebuiltVoiceConfig": {"voiceName": "Aoede"}}}},
        "systemInstruction": {"parts": [{"text": instructions}]},
        "inputAudioTranscription": {}, "outputAudioTranscription": {},
        "realtimeInputConfig": {"automaticActivityDetection": {"disabled": False, "startOfSpeechSensitivity": "START_SENSITIVITY_HIGH", "endOfSpeechSensitivity": "END_SENSITIVITY_LOW", "prefixPaddingMs": 300, "silenceDurationMs": 1000 if pace == "patient" else 650}},
        "tools": [{"functionDeclarations": functions}],
        "contextWindowCompression": {"slidingWindow": {}},
    }}


def is_call_farewell(text):
    return bool(re.fullmatch(r"\s*(?:(?:no|okay|ok|thank you|thanks)[,.!\s]+)*(?:bye|goodbye|bye bye|that's all goodbye)[.!\s]*", str(text), re.I))


def speech_key(text):
    return " ".join(re.findall(r"[a-z0-9]+", str(text).casefold()))


NUMBER_WORDS = {
    "zero": "0", "oh": "0", "one": "1", "two": "2", "three": "3",
    "four": "4", "five": "5", "six": "6", "seven": "7", "eight": "8",
    "nine": "9",
}
CARDINALS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}
SMALL_NUMBERS = {
    1: ("1", "one"), 2: ("2", "two"), 3: ("3", "three"),
    4: ("4", "four"), 5: ("5", "five"), 6: ("6", "six"),
    7: ("7", "seven"), 8: ("8", "eight"), 9: ("9", "nine"),
    10: ("10", "ten"), 11: ("11", "eleven"), 12: ("12", "twelve"),
}


def phone_runs(value):
    """Keep complete numeric runs, including adjoining digit words and repeats."""
    tokens = re.findall(r"[a-z]+|\d+", str(value).casefold())
    repeats = {"double": 2, "triple": 3, "treble": 3, "quadruple": 4}
    runs, current, i = [], "", 0
    while i < len(tokens):
        token = tokens[i]
        if token in repeats and i + 1 < len(tokens):
            digit = NUMBER_WORDS.get(tokens[i + 1], tokens[i + 1])
            if len(digit) == 1 and digit.isdigit():
                current += digit * repeats[token]
                i += 2
                continue
        digit = NUMBER_WORDS.get(token, token if token.isdigit() else "")
        if digit:
            current += digit
        elif current:
            runs.append(current)
            current = ""
        i += 1
    if current:
        runs.append(current)
    return runs


def phone_digits(value):
    return "".join(phone_runs(value))


def name_keys(value):
    """Include normal and spelled-letter forms (A B U -> abu)."""
    text = str(value)
    keys = {speech_key(text)}
    for match in re.finditer(r"(?:\b[a-zA-Z]\b[\s,.'-]*){2,}", text):
        collapsed = "".join(re.findall(r"[a-zA-Z]", match.group())).casefold()
        if collapsed:
            keys.add(collapsed)
    return {key for key in keys if key}


def valid_person_name(value):
    text = str(value).strip()
    return bool(2 <= len(text) <= 80 and re.fullmatch(r"[A-Za-z][A-Za-z .'-]*", text))


def _contains_key(text, value):
    source = f" {speech_key(text)} "
    return any(f" {key} " in source or key.replace(" ", "") in name_keys(text) for key in name_keys(value))


def _contains_number(text, value):
    try:
        number = int(value)
    except (TypeError, ValueError):
        return False
    return any(f" {token} " in f" {speech_key(text)} " for token in SMALL_NUMBERS.get(number, (str(number),)))


def reservation_readback_matches(outcome, spoken):
    """Accept a natural readback only when every booking fact is spoken."""
    if speech_key(outcome.confirmation_text) in speech_key(spoken):
        return True
    if outcome.name not in {"create_booking", "get_reservation_draft"}:
        return False
    if re.search(r"\b(?:reservation|table)\b.{0,24}\b(?:is|has been|was)\b.{0,12}\b(?:confirmed|booked|reserved)\b|\b(?:i(?:'ve| have)?|we(?:'ve| have)?)\s+(?:booked|confirmed|reserved)\b", spoken, re.I):
        return False
    proposed = outcome.result.get("proposed") if isinstance(outcome.result, Mapping) else {}
    proposed = proposed if isinstance(proposed, Mapping) else {}
    facts = outcome.facts if isinstance(outcome.facts, Mapping) else {}
    value = lambda key: proposed.get(key) or facts.get(key)
    customer_name = str(value("customer_name") or "").strip()
    customer_phone = phone_digits(value("customer_phone") or "")
    party_size = value("party_size")
    date_value = str(value("date") or "").strip()
    time_value = str(value("time") or "").strip()
    notes = str(value("notes") or "").strip()
    if not all((customer_name, customer_phone, party_size, date_value, time_value)):
        return False
    if not _contains_key(spoken, customer_name):
        return False
    spoken_phone = phone_digits(spoken)
    if customer_phone not in spoken_phone and (len(customer_phone) < 10 or customer_phone[-10:] not in spoken_phone):
        return False
    if not _contains_number(spoken, party_size):
        return False
    spoken_key = speech_key(spoken)
    try:
        day = date.fromisoformat(date_value)
        date_forms = (
            speech_key(date_value),
            speech_key(day.strftime("%B %d")),
            speech_key(day.strftime("%A")),
        )
    except ValueError:
        date_forms = (speech_key(date_value),)
    if not any(form and form in spoken_key for form in date_forms):
        return False
    match = re.fullmatch(r"(\d{1,2}):(\d{2})", time_value)
    if not match:
        return False
    hour, minute = int(match.group(1)), int(match.group(2))
    hour12 = hour % 12 or 12
    time_forms = {f"{hour} {minute:02d}", f"{hour12} {minute:02d}"}
    if hour12 in SMALL_NUMBERS:
        time_forms.add(f"{SMALL_NUMBERS[hour12][1]} {minute:02d}")
        if minute == 30:
            time_forms.add(f"{SMALL_NUMBERS[hour12][1]} thirty")
    if not any(form in spoken_key for form in time_forms):
        return False
    if notes and notes.casefold() not in {"none", "no notes"}:
        note_terms = [token for token in speech_key(notes).split() if len(token) >= 4]
        if note_terms and not any(token in spoken_key for token in note_terms):
            return False
    return bool(re.search(r"\b(confirm|book|proceed|correct|go ahead|shall i|would you like me)\b", spoken, re.I))


def extract_reservation_slots(text, *, today):
    """Capture only high-confidence non-name slots from caller speech.

    These values survive a clarification turn. Names stay model-mediated because
    proper-name recognition is exactly where the provider can be uncertain.
    """
    spoken = str(text or "")
    reservation_context = bool(re.search(r"\b(?:book|reservation|reserve|table|guests?|people|callback|phone)\b", spoken, re.I))
    if not reservation_context:
        return {}
    slots = {}
    phone = re.search(
        r"\b(?:callback|phone)(?:\s+number)?\s*(?:is|:)?\s*([+\d][\d\s().-]{6,}\d)",
        spoken,
        re.I,
    )
    if phone:
        digits = phone_digits(phone.group(1))
        if 10 <= len(digits) <= 15:
            slots["phone"] = digits
    party = re.search(
        r"\b(?:table\s+for|party\s+of|for)\s+(\d{1,2}|one|two|three|four|five|six|seven|eight|nine|ten)\s*(?:guests?|people|persons?)?\b",
        spoken,
        re.I,
    )
    if party:
        raw_party = party.group(1).casefold()
        value = int(raw_party) if raw_party.isdigit() else CARDINALS.get(raw_party, 0)
        if 1 <= value <= 10:
            slots["party_size"] = value
    corrected_party = re.search(
        r"\b(?:we(?:'ll| will)\s+be|make\s+(?:that|it)|change\s+(?:that|it)\s+to)\s+"
        r"(\d{1,2}|one|two|three|four|five|six|seven|eight|nine|ten)\s+"
        r"(?:guests?|people|persons?)\b",
        spoken,
        re.I,
    )
    if corrected_party:
        raw_party = corrected_party.group(1).casefold()
        value = int(raw_party) if raw_party.isdigit() else CARDINALS.get(raw_party, 0)
        if 1 <= value <= 10:
            slots["party_size"] = value
    clock = re.search(
        r"\b(?:at|for)\s+(\d{1,2})(?::(\d{2}))?\s*(a\.?m\.?|p\.?m\.?)\b",
        spoken,
        re.I,
    )
    if clock:
        hour, minute = int(clock.group(1)), int(clock.group(2) or 0)
        meridiem = clock.group(3).casefold()[0]
        if 1 <= hour <= 12 and 0 <= minute <= 59:
            hour = hour % 12 + (12 if meridiem == "p" else 0)
            slots["time"] = f"{hour:02d}:{minute:02d}"
    relative_day = re.search(
        r"\b(?:this|next)?\s*(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
        spoken,
        re.I,
    )
    if relative_day:
        weekday = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday").index(relative_day.group(1).casefold())
        delta = (weekday - today.weekday()) % 7
        if delta == 0:
            delta = 7
        from datetime import timedelta
        slots["date"] = (today + timedelta(days=delta)).isoformat()
    parsed = party_count(spoken)
    if parsed is not None and 1 <= parsed <= 10:
        slots["party_size"] = parsed
    return slots



def same_phone(left, right):
    left, right = phone_digits(left), phone_digits(right)
    def national(value):
        return value[1:] if len(value) == 11 and value.startswith("1") else value
    return bool(left and right) and national(left) == national(right)


def extract_identity_slots(text, *, reservation_context=False):
    """Retain explicitly supplied identity; never infer a name from other mentions."""
    text = str(text or "")
    slots = {}
    match = re.search(r"\b(?:under (?:the )?name|my name is|correct the name to|change the name to)\s+(.+?)(?=[.,!?]|\s+(?:for|on|at|this|next|tomorrow|phone|callback|spelled)\b|$)", text, re.I)
    if match and valid_person_name(match[1].strip()):
        slots["name"] = match[1].strip()
    if reservation_context or re.search(r"\b(?:phone|callback|number)\b", text, re.I):
        numbers = [value for value in phone_runs(text) if 10 <= len(value) <= 15]
        if len(numbers) == 1:
            slots["phone"] = numbers[0]
    return slots


def party_size_correction(text, current):
    """Resolve a relative guest-count correction against the current booking."""
    spoken = str(text or "")
    if re.search(r"\b(?:add|bring|include)\s+(?:one|1)\s+(?:more|additional)\s+(?:guest|person|friend)\b", spoken, re.I):
        return current + 1
    if re.search(r"\b(?:one|1)\s+(?:fewer|less)\s+(?:guest|person)\b", spoken, re.I):
        return max(1, current - 1)
    return party_count(spoken, current)


def canonical_available_menu_item(items, requested, spoken):
    """Resolve only a unique, speech-supported live menu item."""
    def tokens(value):
        words = speech_key(value).split()
        if words and words[-1] == "salad":
            words.pop()
        return [word[:-1] if len(word) > 3 and word.endswith("s") else word for word in words]

    wanted = tokens(requested)
    heard = tokens(spoken)
    if not wanted or not heard:
        return None
    heard_text = " " + " ".join(heard) + " "
    matches = []
    for item in items:
        if item.get("available") is not True:
            continue
        names = [item.get("name") or "", *(item.get("aliases") or [])]
        if any(tokens(name) == wanted and " " + " ".join(tokens(name)) + " " in heard_text for name in names):
            matches.append(item)
    return matches[0] if len(matches) == 1 else None


class GeminiLiveSession(StreamingVoiceSession):
    def __init__(self, *, domain, socket, send, pace):
        super().__init__(domain=domain, transport=MemoryRealtimeTransport(), speech_client=None, send=send, config={})
        self.socket = socket
        self.pace = pace
        self.provider_lock = asyncio.Lock()
        self.tool_queue = asyncio.Queue(maxsize=20)
        self.input_id = ""
        self.input_text = ""
        self.input_ready = asyncio.Event()
        self.input_finalized = False
        self.input_consumed = False
        self.input_updated = 0.0
        self.input_history = []
        self.cancelled_calls = set()
        self.function_results = {}
        self.model_turn_open = False
        self.block_output = False
        self.pending_identity_change = {}
        self.reservation_buffer = {}
        self.current_reservation_slots = {}
        self.pending_booking_update = None
        self.pending_booking_creation = None
        self.pending_order_amendment = None
        self.pending_dine_in_fulfillment = False
        self.ending_call = False
        self.staged_preorder = None
        self.auto_confirmation_result = None

    async def provider_send(self, event):
        async with self.provider_lock:
            await self.socket.send(json.dumps(event, separators=(",", ":")))

    async def start(self):
        await self.domain.start()
        await self.provider_send(native_setup(self.domain, self.pace))
        while True:
            event = json.loads(await asyncio.wait_for(self.socket.recv(), 20))
            if "setupComplete" in event:
                break
            if "error" in event:
                raise RuntimeError("gemini_setup_rejected")
        self.reader_task = asyncio.create_task(self.read_events())
        self.worker_task = asyncio.create_task(self.tools_worker())
        await self.emit({"type": "ready", "session_id": self.domain.session_id, "model": MODEL,
                         "pipeline": "Gemini Live conversation · verified booking speech", "input_sample_rate": 16000})

    async def append_audio(self, audio):
        if self.ending_call:
            return
        if not audio or len(audio) > 32000 or len(audio) % 2:
            raise ValueError("invalid_audio_chunk")
        await self.provider_send({"realtimeInput": {"audio": {"mimeType": "audio/pcm;rate=16000", "data": base64.b64encode(audio).decode()}}})

    async def interrupt(self, *, provider=False):
        self.epoch += 1
        self.domain.interruptions.interrupt()
        self.utterance = None
        self.model_turn_open = False
        self.block_output = not (provider and self.input_text and not self.input_finalized)
        await self.emit({"type": "interrupted", "epoch": self.epoch})
        if not provider:
            # A user pressing Stop is an explicit interruption, not an audio VAD guess.
            await self.provider_send({"clientContent": {"turns": [{"role": "user", "parts": [{"text": "Stop speaking and listen for my next request."}]}], "turnComplete": False}})

    async def input_caption(self, text):
        if self.ending_call or not text:
            return
        if not self.input_id or self.input_consumed:
            if self.utterance and not self.utterance.get("played"):
                await self.interrupt(provider=True)
            self.input_id = "gemini-input-" + uuid.uuid4().hex
            self.input_text = ""
            self.input_finalized = False
            self.input_consumed = False
            self.auto_confirmation_result = None
            self.input_ready = asyncio.Event()
            await self.emit({"type": "speech_started", "item_id": self.input_id})
        self.input_text += text
        self.block_output = False
        self.input_updated = time.monotonic()
        self.input_ready.set()
        await self.emit({"type": "transcript_delta", "item_id": self.input_id, "delta": text})

    async def finalize_input(self):
        if self.input_finalized:
            return
        await asyncio.wait_for(self.input_ready.wait(), 3)
        # Gemini input captions have no guaranteed ordering relative to tool calls.
        # Let the current caption burst finish before authorizing a domain operation.
        await asyncio.sleep(max(0, .12 - (time.monotonic() - self.input_updated)))
        if not self.input_text.strip():
            raise RuntimeError("caller_transcript_unavailable")
        self.domain._outcomes = []
        self.domain._completed_turn = None
        self.domain.turns.start(self.input_id)
        await self.domain._finalize_caller_turn(self.input_text.strip(), generation=self.domain.interruptions.generation)
        self.input_history.append(self.input_text.strip())
        affirmation = classify_affirmation(self.input_text)
        if affirmation == "negative":
            self.pending_order_amendment = None
            self.pending_booking_update = None
            self.pending_booking_creation = None
        from app.services.restaurant import _restaurant_now
        captured = extract_reservation_slots(self.input_text, today=_restaurant_now().date())
        captured.update(extract_identity_slots(self.input_text, reservation_context=bool(self.reservation_buffer or captured)))
        self.current_reservation_slots = captured
        if captured:
            self.reservation_buffer.update(captured)
        if re.search(r"\bpre[ -]?order\b", self.input_text, re.I) and not requests_order_abandonment(self.input_text) and not self.draft.get("preorder_request"):
            self.draft["preorder_request"] = self.input_text.strip()
        if self.draft.get("preorder_request") and self.staged_preorder is None and not self.domain.state.items:
            service = getattr(self.domain, "_native_service", None)
            moment = await self.menu_moment()
            request = self.draft["preorder_request"]
            if service is not None and moment is not None and not re.search(r"\b(?:without|extra|allerg|substitut|remove)\w*\b", request, re.I):
                menu = await service.list_menu(available_only=False, at=moment)
                actions = []
                planned = []
                for item in menu.get("items", []):
                    count = item_count(request, item.get("name", ""))
                    if count is not None and 1 <= count <= 20 and item.get("available"):
                        actions.append({"name":"add_order_item","arguments":{"item_name":item["name"],"quantity":count}})
                        planned.append({"name":item["name"],"quantity":count})
                if actions:
                    self.staged_preorder = {"actions":actions}
                    self.draft["preorder_items"] = planned
        self.input_finalized = True
        if captured and getattr(self.domain, "_native_service", None) is not None:
            identity = await self.reservation_identity()
            if identity.get("status") != "confirmed":
                result = await self.execute_function({"id":self.input_id + ":captured-draft",
                    "name":"update_reservation_draft", "args":dict(self.reservation_buffer)}, self.epoch)
                if result.get("readback_text"):
                    self.auto_confirmation_result = result
                    self.block_output = True
                    self.input_consumed = True
                    text = result["readback_text"]
                    planned = self.draft.get("preorder_items") or []
                    if planned:
                        text = "Requested food: " + ", ".join(f"{i['quantity']} {i['name']}" for i in planned) + ". The table and food need separate approvals. " + text
                    from openai import AsyncOpenAI
                    async with AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"]) as speech:
                        self.speech_client = speech
                        try:
                            await self.speak(text, self.epoch, item_id=self.input_id)
                        finally:
                            self.speech_client = None
        await self.emit({"type": "transcript", "item_id": self.input_id, "text": self.input_text.strip()})
        await self.emit({"type": "speech_stopped", "item_id": self.input_id})
        if is_call_farewell(self.input_text):
            self.ending_call = True
            self.block_output = True
            self.input_consumed = True
            await self.emit({"type":"call_closing"})
            from openai import AsyncOpenAI
            async with AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"]) as speech:
                self.speech_client = speech
                try:
                    await self.speak("Goodbye, and thank you for calling.", self.epoch, item_id=self.input_id)
                finally:
                    self.speech_client = None
            await self.emit({"type":"call_end_after_playback"})
        elif requests_order_abandonment(self.input_text):
            await self.abandon_pending_order()
        elif affirmation == "affirmative":
            pending = active_released_confirmation(self.domain.session_id)
            if pending and pending[0] in {"create_booking", "confirm_order", "update_confirmed_booking", "cancel_booking"}:
                await self.confirm_pending_action(*pending)
        elif self.domain.state.items and not re.search(r"\b(?:add|remove|change|swap|replace|cancel)\b", self.input_text, re.I) and (wants_review(self.input_text) or re.search(r"\b(?:read|repeat|review|summari[sz]e)\b.{0,55}\b(?:order|food|pre[ -]?order)\b", self.input_text, re.I)):
            result = await self.execute_function({"id":self.input_id + ":server-readback","name":"get_order_summary","args":{}}, self.epoch)
            text = result.get("readback_text")
            if text:
                self.auto_confirmation_result = result
                self.block_output = True
                self.input_consumed = True
                from openai import AsyncOpenAI
                async with AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"]) as speech:
                    self.speech_client = speech
                    try:
                        await self.speak(text, self.epoch, item_id=self.input_id)
                    finally:
                        self.speech_client = None

    async def abandon_pending_order(self):
        """Leave only this call's unconfirmed food order; never cancel its booking."""
        await self._execute_server_action("abandon_pending_order", {})

    async def confirm_pending_booking(self):
        """Compatibility entry point; the server-owned proposal remains authoritative."""
        pending = active_released_confirmation(self.domain.session_id)
        if pending and pending[0] == "create_booking":
            await self.confirm_pending_action(*pending)

    async def confirm_pending_action(self, action, record):
        """Execute the one heard, approved transaction without model tool choice."""
        payload = dict(record["payload"])
        if action == "create_booking":
            args = {
                "name": payload.get("customer_name", ""), "phone": payload.get("customer_phone", ""),
                "date": payload.get("date", ""), "time": payload.get("time", ""),
                "party_size": payload.get("party_size", 0), "notes": payload.get("notes", ""),
                "caller_confirmed": True,
            }
        elif action == "confirm_order":
            args = {
                "expected_draft_version": payload.get("draft_version", 0),
                "caller_approved_full_readback": True,
            }
        elif action == "update_confirmed_booking":
            allowed = {
                "booking_id", "date", "time", "party_size", "customer_name",
                "preferred_location", "seating_preference", "seating_backup", "seating_avoid",
                "dietary", "occasion", "extra_notes", "notes", "require_approval_for_paid_items",
            }
            args = {key: value for key, value in payload.items() if key in allowed}
            args["caller_confirmed"] = True
        elif action == "cancel_booking":
            args = {
                "booking_id": payload.get("booking_id", 0),
                "customer_name": payload.get("customer_name", ""),
                "customer_phone": payload.get("customer_phone", ""),
                "reason": payload.get("reason", ""), "caller_confirmed": True,
            }
        else:
            raise ValueError("unsupported_pending_action")

        await self._execute_server_action(action, args, requires_confirmation=True)

    async def _execute_server_action(self, action, args, *, requires_confirmation=False):
        """One result path for approval and explicit draft abandonment."""
        generation = self.domain.interruptions.generation
        args = dict(args)
        args["session_id"] = self.domain.session_id
        outcome = await self.domain._run_tool(self.input_id + ":" + action, action, args, generation=generation)
        outcome = self.domain._with_confirmation(outcome)
        if outcome.success and outcome.readback_verified:
            outcome = await self.domain._persist_committed_outcome(outcome)
            if action == "create_booking":
                self.pending_booking_creation = None
            elif action == "update_confirmed_booking":
                self.pending_booking_update = None
            elif action == "abandon_pending_order":
                self.pending_order_amendment = None
                self.heard_order = None
        elif requires_confirmation:
            # A failed or uncertain attempt is terminal for this spoken
            # proposal. A later retry requires fresh status and readback.
            clear_pending_confirmation(self.domain.session_id, action)
        await self.domain._persist_native_confirmation_state()
        self.domain._outcomes.append(outcome)
        old_version = self.domain.state.version
        synced = await self.domain._sync_order_memory(generation=None)
        if synced and synced.version != old_version:
            outcome = replace(outcome, state_version=synced.version)
            self.domain._outcomes[-1] = outcome
        self.auto_confirmation_result = self.domain._model_tool_output(outcome)
        await self.emit({"type": "tool", "name": action, "success": outcome.success,
                         "verified": outcome.readback_verified, "error": outcome.error})
        self.block_output = True
        self.input_consumed = True
        if outcome.success and outcome.readback_verified:
            sentence = outcome.confirmation_text or self.auto_confirmation_result.get("speech") or "The request is confirmed."
        else:
            subject = {
                "create_booking": "reservation", "confirm_order": "pre-order",
                "update_confirmed_booking": "reservation change", "cancel_booking": "cancellation",
                "abandon_pending_order": "pre-order abandonment",
            }[action]
            sentence = f"I could not verify the {subject}. I won't claim it is complete or retry it without checking the current status."
        if action == "create_booking" and outcome.success and outcome.readback_verified and self.staged_preorder:
            plan = dict(self.staged_preorder)
            plan["expected_order_revision"] = state_view(self.domain.state)["order_revision"]
            plan["retain_order_item_ids"] = [int(i.line_id) for i in self.domain.state.items if str(i.line_id).isdigit()]
            identity = await self.reservation_identity()
            plan["customer_name"] = identity.get("customer_name") or plan.get("customer_name")
            plan["customer_phone"] = identity.get("customer_phone") or plan.get("customer_phone")
            prepared, candidate = await self.apply_order_changes(plan, self.input_id + ":preorder", generation, self.epoch)
            self.auto_confirmation_result["preorder_draft"] = prepared
            if candidate and candidate.name == "get_order_summary":
                self.staged_preorder = None
                sentence += " Your pre-order is prepared as a draft, but is not yet confirmed. Shall I read it back for your approval?"
            else:
                sentence += " Your table is confirmed, but I could not finish preparing the pre-order. I still have your requested items; the food is not confirmed."
        from openai import AsyncOpenAI
        speech = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self.speech_client = speech
        try:
            await self.speak(sentence, self.epoch, item_id=self.input_id)
        finally:
            self.speech_client = None
            await speech.close()

    async def ensure_output(self):
        if self.model_turn_open and self.utterance:
            return
        async with self.domain_lock:
            if self.input_text and not self.input_finalized:
                await self.finalize_input()
        if self.block_output:
            return
        self.input_consumed = True
        self.model_turn_open = True
        self.utterance = {"token": uuid.uuid4().hex, "epoch": self.epoch, "text": "", "version": self.domain.state.version,
                          "outcomes": tuple(self.domain._outcomes), "bytes": 0, "started": None, "done": False, "played": False}
        await self.emit({"type": "assistant", "text": "", "token": self.utterance["token"], "epoch": self.epoch})

    async def read_events(self):
        try:
            while not self.closed:
                event = json.loads(await self.socket.recv())
                content = event.get("serverContent", {})
                if content.get("interrupted") and not self.ending_call:
                    await self.interrupt(provider=True)
                if "toolCallCancellation" in event and not self.ending_call:
                    self.cancelled_calls.update(event["toolCallCancellation"].get("ids", []))
                    await self.interrupt(provider=True)
                if content.get("inputTranscription", {}).get("text"):
                    await self.input_caption(content["inputTranscription"]["text"])
                if "toolCall" in event and not self.ending_call:
                    await self.tool_queue.put((event["toolCall"].get("functionCalls", []), self.epoch))
                parts = content.get("modelTurn", {}).get("parts", [])
                caption = content.get("outputTranscription", {}).get("text", "")
                if not self.block_output and (caption or any(p.get("inlineData") for p in parts)):
                    await self.ensure_output()
                    if self.block_output:
                        continue
                    u = self.utterance
                    if caption:
                        u["text"] += caption
                        await self.emit({"type": "assistant_delta", "token": u["token"], "epoch": self.epoch, "text": u["text"]})
                    for part in parts:
                        data = part.get("inlineData")
                        if data and str(data.get("mimeType", "")).startswith("audio/pcm"):
                            raw = base64.b64decode(data["data"])
                            if u["started"] is None:
                                u["started"] = time.monotonic()
                                await self.emit({"type": "timing", "token": u["token"], "provider": MODEL, "caption_to_first_audio_ms": round((time.monotonic() - self.input_updated) * 1000) if self.input_updated else None})
                            u["bytes"] += len(raw)
                            await self.emit({"type": "audio", "token": u["token"], "epoch": self.epoch, "audio": data["data"]})
                if not self.block_output and (content.get("generationComplete") or content.get("turnComplete")):
                    if self.utterance and not self.utterance["done"]:
                        self.utterance["done"] = True
                        await self.emit({"type": "audio_done", "token": self.utterance["token"], "epoch": self.epoch})
                if content.get("turnComplete"):
                    self.model_turn_open = False
                if "goAway" in event:
                    await self.emit({"type": "notice", "message": "Gemini is ending this session shortly. Finish this test and start a new call."})
                if "error" in event:
                    raise RuntimeError("gemini_provider_error")
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if not self.closed:
                await self.emit({"type": "error", "message": "The Gemini voice connection stopped. Please reconnect.", "code": type(exc).__name__})

    def name_supported(self, args):
        names = [str(args[key]).strip() for key in ("customer_name", "name") if args.get(key)]
        source = " ".join(self.input_history + [self.input_text])
        return all(valid_person_name(name) and _contains_key(source, name) for name in names)

    async def reservation_identity(self):
        service = getattr(self.domain, "_native_service", None)
        if service is None or not hasattr(service, "load_call_state"):
            return {}
        persisted = await service.load_call_state(self.domain.session_id)
        state = persisted.get("state") or {}
        draft = state.get("reservation_draft") or state
        return dict(draft) if isinstance(draft, Mapping) else {}

    @staticmethod
    def contextual_note(value, spoken):
        note = str(value or "").strip()
        if not note:
            return note, None
        if re.fullmatch(r"(?:a\s+)?vegetarian(?:\s+guest)?", note, re.I):
            if re.search(r"\bone of (?:my|our) (?:friends|guests)\b", spoken, re.I):
                return "One guest is vegetarian.", None
            return "A guest is vegetarian.", None
        if re.fullmatch(r"(?:(?:one|two|three|\d+)\s+)?high\s*chairs?(?:\s+requested)?", note, re.I):
            count = re.search(r"\b(\d+|one|two|three)\s+high\s*chairs?\b", spoken, re.I)
            if not count:
                return note, "Ask how many high chairs are needed and how many children are joining before saving a specific request. High chairs are limited and not guaranteed."
            quantity = CARDINALS.get(count.group(1).casefold(), count.group(1))
            return f"Request for {quantity} high chair(s) for children; subject to availability.", None
        return note, None

    @staticmethod
    def order_amendment_summary(args):
        phrases = []
        for action in args.get("actions") or []:
            name = action.get("name")
            values = action.get("arguments") or {}
            if name == "add_order_item":
                phrases.append(f"add {values.get('quantity') or 1} {values.get('item_name') or 'item'}")
            elif name == "update_order_item":
                phrases.append(f"change order line {values.get('order_item_id')} to quantity {values.get('quantity')}")
            elif name == "remove_order_item":
                phrases.append(f"remove order line {values.get('order_item_id')}")
            else:
                phrases.append(name.replace("_", " ") if isinstance(name, str) else "change the order")
        return ", ".join(phrases)

    async def menu_moment(self):
        from datetime import datetime
        from zoneinfo import ZoneInfo
        from app.restaurant_knowledge import get_restaurant_knowledge
        state = self.domain.state
        details = state.fulfillment_details or {}
        moment = details.get("fulfillment_at")
        if state.fulfillment in {"pickup", "delivery"} or re.search(r"\b(?:right now|today|pickup|delivery|takeaway)\b", self.input_text, re.I):
            if not moment:
                return None
        elif not moment:
            draft = {**(await self.reservation_identity()), **self.reservation_buffer}
            if draft.get("date") and draft.get("time"):
                moment = str(draft["date"]) + "T" + str(draft["time"])
        if not moment:
            return None
        try:
            parsed = datetime.fromisoformat(str(moment))
        except ValueError:
            return None
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=ZoneInfo(get_restaurant_knowledge().identity["timezone"]))

    async def validate_identity_fields(self, args, *, confirmation_only):
        existing = await self.reservation_identity()
        proposed_name = str(args.get("name") or args.get("customer_name") or "").strip()
        if proposed_name:
            if not valid_person_name(proposed_name):
                return {"status": "needs_clarification", "error": "invalid_name_field", "instruction": "The proposed name is not a valid person name and may be a phone number. Ask for the name only; do not guess or write it."}
            old_name = str(existing.get("customer_name") or "").strip()
            if old_name and speech_key(old_name) != speech_key(proposed_name):
                if confirmation_only and speech_key(self.pending_identity_change.get("name", "")) == speech_key(proposed_name):
                    self.pending_identity_change.pop("name", None)
                elif _contains_key(self.input_text, proposed_name) and re.search(r"\b(?:no|actually|correction|correct|change|name is|under (?:the )?name|spell)\b", self.input_text, re.I):
                    # A draft correction is not a booking commit. Confirmed-booking
                    # changes still pass the domain's proposal/readback boundary.
                    self.pending_identity_change.pop("name", None)
                    self.reservation_buffer["name"] = proposed_name
                else:
                    return {"status": "needs_clarification", "error": "identity_change_unverified", "instruction": f"Keep the saved name {old_name}. Do not replace it with {proposed_name} unless the caller explicitly corrects and confirms the change."}
            elif not old_name and not self.name_supported({"name": proposed_name}):
                return {"status": "needs_clarification", "error": "name_not_in_current_speech", "instruction": "That name is not supported by the caller's speech. Ask once for the caller to say or spell the name; do not substitute or guess."}
            elif not old_name and not _contains_key(self.input_text, proposed_name) and speech_key(self.reservation_buffer.get("name", "")) != speech_key(proposed_name):
                return {"status": "needs_clarification", "error": "name_not_in_current_speech", "instruction": "The caller did not provide this name in the current turn. Ask for the name only; do not reuse a guess from an earlier transcript."}


        phone_key = "phone" if args.get("phone") is not None else "customer_phone" if args.get("customer_phone") is not None else ""
        if phone_key:
            proposed_phone = phone_digits(args.get(phone_key) or "")
            old_phone = phone_digits(existing.get("customer_phone") or "")
            current_numbers = [value for value in phone_runs(self.input_text) if len(value) >= 7]
            if re.search(r"\btetra\b", self.input_text, re.I):
                return {"status":"needs_clarification","error":"ambiguous_digit_repeat","instruction":"Ask how many times the digit should repeat, then ask for that group one digit at a time. Do not guess or change the phone."}
            if current_numbers and (len(current_numbers) != 1 or not same_phone(current_numbers[0], proposed_phone)):
                return {"status":"needs_clarification","error":"phone_transcript_mismatch","instruction":"The proposed phone does not match every digit heard. Do not drop, add or truncate digits. Ask for the number in groups, one digit at a time, and read all digits back."}
            if len(proposed_phone) == 11 and not proposed_phone.startswith("1") and "+" not in self.input_text and not old_phone:
                return {"status":"needs_clarification","error":"phone_country_code_required","instruction":"This has eleven digits and is not a US number with country code 1. Ask whether it is an international number and request its country code explicitly. Do not repeatedly request the same digits or invent a country code."}
            if not 10 <= len(proposed_phone) <= 15:
                return {"status": "needs_clarification", "error": "invalid_phone_field", "instruction": "The callback number is incomplete. Ask for the phone number again in small groups; keep it separate from the name and do not write a partial number."}
            args[phone_key] = proposed_phone
            if old_phone and not same_phone(old_phone, proposed_phone):
                if confirmation_only and self.pending_identity_change.get("phone") == proposed_phone:
                    self.pending_identity_change.pop("phone", None)
                elif proposed_phone in phone_digits(self.input_text) and re.search(r"\b(?:actually|correction|correct|change|phone|number)\b", self.input_text, re.I):
                    self.pending_identity_change["phone"] = proposed_phone
                    return {"status": "needs_confirmation", "error": "identity_change_requires_confirmation", "instruction": "The callback number differs from the saved number. Read only the last four digits of both and ask which number to keep; do not update it yet."}
                else:
                    return {"status": "needs_clarification", "error": "identity_change_unverified", "instruction": "Keep the saved callback number. Do not replace it unless the caller explicitly corrects and confirms the new number."}
            elif not old_phone and proposed_phone not in phone_digits(self.input_text) and not same_phone(proposed_phone, self.reservation_buffer.get("phone", "")):
                return {"status": "needs_clarification", "error": "phone_not_in_current_speech", "instruction": "That phone number is not supported by the current caller speech. Ask for the callback number again in small groups; do not guess."}
        if proposed_name and _contains_key(self.input_text, proposed_name):
            self.reservation_buffer["name"] = proposed_name
        if phone_key and proposed_phone in phone_digits(self.input_text):
            self.reservation_buffer["phone"] = proposed_phone
        return None

    async def execute_function(self, call, epoch):
        name, args, call_id = call["name"], dict(call.get("args") or {}), call["id"]
        if self.ending_call or epoch != self.epoch or call_id in self.cancelled_calls:
            return {"status": "cancelled", "instruction": "The caller interrupted. Listen to their correction."}
        if call_id in self.function_results:
            return self.function_results[call_id]
        await self.finalize_input()
        if self.auto_confirmation_result is not None:
            return self.auto_confirmation_result
        if name in {"create_booking", "confirm_order", "cancel_booking", "update_confirmed_booking"} and call.get("args", {}).get("caller_confirmed") and classify_affirmation(self.input_text) != "affirmative":
            return {"status":"needs_confirmation","error":"approval_not_unambiguous","instruction":"No technical failure occurred and nothing was committed. Ask a brief explicit approval question for the existing proposal. Do not claim a system error, promise a background retry, or change the draft."}
        if self.ending_call or epoch != self.epoch or call_id in self.cancelled_calls:
            return {"status": "cancelled", "instruction": "The caller interrupted before execution. Listen to their correction."}
        generation = self.domain.interruptions.generation
        if name == "check_table_availability" and not any(o.name == "check_table_availability" for o in self.domain._outcomes):
            args.update({k:v for k,v in self.current_reservation_slots.items() if k in {"date","time","party_size"}})
        confirmation_only = classify_affirmation(self.input_text) == "affirmative"
        existing_reservation = await self.reservation_identity()
        menu_at = await self.menu_moment()
        executor = getattr(getattr(self.domain, "tool_bridge", None), "executor", None)
        if executor is not None:
            executor.menu_at = menu_at
        confirmed_booking_id = int(existing_reservation.get("booking_id") or 0) if existing_reservation.get("status") == "confirmed" else 0
        if confirmed_booking_id and name == "create_booking" and re.search(r"\b(?:change|update|modify|make|add|more|fewer|instead)\b", self.input_text, re.I):
            name = "update_confirmed_booking"
            args = {k:v for k,v in args.items() if k in {"date","time","party_size","dietary","occasion","extra_notes","seating_preference"}}
        if confirmed_booking_id and name == "create_booking":
            return {"status": "rejected", "error": "booking_already_confirmed", "instruction": "This call already owns a confirmed reservation. Use update_confirmed_booking for changes; never create a duplicate booking."}
        if confirmation_only and self.pending_booking_creation and name in {"create_booking", "get_reservation_draft"}:
            name = "create_booking"
            args = {**self.pending_booking_creation, "caller_confirmed": True}
        if confirmed_booking_id and name == "update_reservation_draft":
            name = "update_confirmed_booking"
            args["caller_confirmed"] = False
        if name == "update_confirmed_booking" and confirmed_booking_id:
            supplied = int(args.get("booking_id") or 0)
            if supplied and supplied != confirmed_booking_id:
                return {"status": "rejected", "error": "booking_scope_mismatch", "instruction": "The requested booking does not belong to this call. Keep the current reservation unchanged."}
            args["booking_id"] = confirmed_booking_id
            if confirmation_only and self.pending_booking_update:
                args = {**self.pending_booking_update, "booking_id": confirmed_booking_id, "caller_confirmed": True}
            else:
                args["caller_confirmed"] = False
                current_party = int(existing_reservation.get("party_size") or 0)
                relative_party = party_size_correction(self.input_text, current_party)
                explicit_party = self.current_reservation_slots.get("party_size")
                if relative_party is not None and explicit_party is not None and relative_party != explicit_party:
                    return {"status": "needs_clarification", "error": "party_size_conflict", "instruction": f"The current booking is for {current_party} guests. The caller said both one more guest and {explicit_party} guests. Ask for the intended final party size before checking availability or updating the booking."}
                if explicit_party is not None:
                    args["party_size"] = explicit_party
                elif relative_party is not None:
                    args["party_size"] = relative_party
                if "party_size" in args and not 1 <= int(args["party_size"]) <= 10:
                    return {"status":"needs_clarification","error":"party_size_out_of_range","instruction":"Ask for the intended final party size between one and ten. Keep the existing reservation."}
            if not confirmation_only:
                args = {k:v for k,v in args.items() if k in {"booking_id","caller_confirmed"} or k not in existing_reservation or str(v) != str(existing_reservation[k])}
            if any(args.get(k) and str(args[k]) != str(existing_reservation.get(k) or "") for k in ("party_size","date","time")):
                date_value = str(args.get("date") or existing_reservation.get("date") or "")
                time_value = str(args.get("time") or existing_reservation.get("time") or "")
                if not date_value or not time_value:
                    return {"status": "needs_clarification", "error": "booking_slot_missing", "instruction": "Check the current booking date and time before proposing a party-size change."}
                availability = await self.domain._run_tool(
                    call_id + ":availability", "check_table_availability",
                    {"session_id": self.domain.session_id, "date": date_value, "time": time_value, "party_size": int(args.get("party_size") or existing_reservation.get("party_size") or 0)},
                    generation=generation,
                )
                self.domain._outcomes.append(availability)
                if not availability.success or not availability.readback_verified or not (availability.result or {}).get("available"):
                    return {"status": "unavailable", "error": availability.error or "capacity_unavailable", "instruction": "The proposed party size has not been verified as available. Keep the existing reservation and offer only verified alternatives.", "alternatives": (availability.result or {}).get("alternatives", []) if isinstance(availability.result, Mapping) else []}
        for note_key in ("dietary", "extra_notes", "note"):
            if note_key in args and args[note_key] is not None:
                contextual, ask = self.contextual_note(args[note_key], self.input_text)
                if ask:
                    return {"status": "needs_clarification", "error": "high_chair_quantity_missing", "instruction": ask}
                args[note_key] = contextual
        identity_problem = await self.validate_identity_fields(args, confirmation_only=confirmation_only)
        if identity_problem:
            return identity_problem
        if name == "update_reservation_draft" and self.reservation_buffer:
            for key, value in self.reservation_buffer.items():
                if key in self.current_reservation_slots or not re.search(
                    {"date": r"\b(?:date|day|monday|tuesday|wednesday|thursday|friday|saturday|sunday|tomorrow)\b",
                     "time": r"\b(?:time|at\s+\d|a\.?m\.?|p\.?m\.?)\b",
                     "party_size": r"\b(?:guest|people|person|party|table for)\b",
                     "phone": r"\b(?:phone|callback|number)\b"}.get(key, r"$^"),
                    self.input_text, re.I,
                ):
                    args[key] = value
        if name == "apply_order_changes":
            for action in args.get("actions", []):
                if not isinstance(action, dict):continue
                a = action.get("arguments") or {}
                if action.get("name") not in {"add_order_item","update_order_item"}:continue
                line = next((i for i in self.domain.state.items if str(i.line_id) == str(a.get("order_item_id"))), None)
                item_name = a.get("item_name") or (getattr(line,"item_name","") if line else "")
                quantity = item_count(self.input_text, item_name, getattr(line,"quantity",None))
                if quantity is not None:
                    if not 1 <= quantity <= 20:
                        return {"status":"needs_clarification","error":"quantity_out_of_range","instruction":"Ask for a quantity between one and twenty. Nothing was changed."}
                    a["quantity"] = quantity
        confirmed_order = self.domain.state.status == "confirmed"
        approved_order_amendment = False
        if name in {"apply_order_changes", "confirm_order", "get_order_summary"} and confirmation_only and self.pending_order_amendment:
            name = "apply_order_changes"
            args = dict(self.pending_order_amendment)
            args["actions"] = [
                {"name": action["name"], "arguments": {**action["arguments"], "caller_confirmed": True}}
                for action in args["actions"]
            ]
            self.pending_order_amendment = None
            approved_order_amendment = True
        elif name == "apply_order_changes" and confirmed_order and not confirmation_only:
            actions = args.get("actions")
            if not isinstance(actions, list) or not actions:
                return {"status": "rejected", "error": "invalid_order_changes", "instruction": "No order change was made. Ask what the caller wants to change."}
            current_order = state_view(self.domain.state)
            filtered = []
            for action in actions:
                if not isinstance(action, dict) or action.get("name") not in ORDER_CHANGES or not isinstance(action.get("arguments"), dict):
                    return {"status": "rejected", "error": "invalid_order_changes", "instruction": "No order change was made. Ask for a specific item or note change."}
                if action["name"] == "set_order_fulfillment":
                    requested = str(action["arguments"].get("fulfillment_type") or "")
                    if requested != str(current_order.get("fulfillment") or "") or re.search(r"\b(?:change|switch|move)\b.{0,35}\b(?:pickup|delivery|dine.?in|fulfillment)\b", self.input_text, re.I):
                        return {"status": "rejected", "error": "confirmed_fulfillment_change_unsupported", "instruction": "A confirmed order's fulfillment cannot be changed here. Keep the existing order unchanged and explain that staff must help with that request."}
                    continue
                filtered.append(action)
            if len(filtered) != 1:
                return {"status": "needs_clarification", "error": "confirmed_order_one_change_at_a_time", "instruction": "Handle one confirmed pre-order change at a time so a later action cannot fail after an earlier write. Ask which change to make first; no change was made."}
            if filtered[0]["name"] == "add_order_item":
                service = getattr(self.domain, "_native_service", None)
                if service is None:
                    return {"status": "failed", "error": "menu_unavailable", "instruction": "I cannot verify the live menu right now. No order change was made."}
                available_menu = await service.list_menu(available_only=True, **({"at": menu_at} if menu_at else {}))
                candidate = canonical_available_menu_item(
                    available_menu.get("items", []), filtered[0]["arguments"].get("item_name", ""), self.input_text,
                )
                if candidate is None:
                    return {"status": "needs_clarification", "error": "menu_item_unverified", "instruction": "I could not uniquely match that dish to an available menu item. Ask the caller for the exact name before proposing an order change."}
                filtered[0] = {**filtered[0], "arguments": {**filtered[0]["arguments"], "item_name": candidate["name"]}}
            args["actions"] = filtered
            existing = {int(item.line_id) for item in self.domain.state.items if item.status != "removed" and str(item.line_id).isdigit()}
            retained = args.get("retain_order_item_ids")
            changed = [action["arguments"].get("order_item_id") for action in filtered if action["name"] in {"remove_order_item", "update_order_item"}]
            if (args.get("expected_order_revision") != current_order["order_revision"] or
                    not isinstance(retained, list) or any(type(item) is not int for item in retained + changed) or
                    len(set(changed)) != len(changed) or set(retained) & set(changed) or
                    set(retained) | set(changed) != existing):
                return {"status": "rejected", "error": "order_preservation_plan_invalid", "current_order": current_order,
                        "instruction": "No change was made. Keep every unaffected order line and use the current order revision before proposing the amendment."}
            self.pending_order_amendment = dict(args)
            return {"status": "needs_confirmation", "instruction": "This is a proposal to amend the existing confirmed pre-order. Read the requested change back and ask for an explicit yes in a later caller turn. Nothing has changed yet.", "proposed_change": self.order_amendment_summary(args), "current_order": state_view(self.domain.state)}
        identity_confirmation = confirmation_only and any(
            key in args for key in ("name", "customer_name", "phone", "customer_phone")
        )
        if confirmation_only and name in ORDER_CHANGES | {"apply_order_changes", "update_reservation_draft", "remember_request"} and not identity_confirmation and not approved_order_amendment:
            return {"status": "rejected", "error": "confirmation_turn_cannot_change_order", "instruction": "Caller agreed to the previous readback. Confirm the eligible pending request, or obtain a fresh summary if none is eligible. Do not rebuild or change the order."}
        if name == "get_order_summary" and self.staged_preorder and not self.domain.state.items:
            return {"status":"staged_not_ordered","proposed_preorder":self.draft.get("preorder_items", []),
                    "instruction":"These requested foods are retained as a proposal, not a placed order. The table must be confirmed before the food draft can be created. Do not report order_not_found as a system failure or claim the food is confirmed."}
        if name == "remember_request":
            self.draft = {**{k:v for k,v in self.draft.items() if k.startswith("preorder_")}, "request": str(args.get("request", ""))[:8000], "missing": args.get("missing", []), "source_turn": self.input_id}
            result = {"status": "tentative_only", "draft": self.draft}
        elif name == "restaurant_information":
            result = {"status": "verified_information", "answer": restaurant_information(str(args.get("query", "")))}
        elif name == "suggest_menu_for_preference":
            service = getattr(self.domain, "_native_service", None)
            if service is None:
                result = {"status": "failed", "instruction": "I cannot verify current menu availability. Do not recommend a dish by memory."}
            else:
                result = await menu_suggestions(service, str(args.get("preference") or ""), at=menu_at)
        elif name == "answer_menu_question":
            outcome = await self.domain._run_tool(call_id, "check_menu_item_availability", {"item_name": args.get("item_name", "")}, generation=generation)
            self.domain._outcomes.append(outcome)
            result = {"status": "verified_information" if outcome.success and outcome.readback_verified else "failed", "answer": menu_answer(outcome, args.get("topic", "details"))}
            await self.emit({"type": "tool", "name": name, "success": outcome.success, "verified": outcome.readback_verified, "error": outcome.error})
        elif name == "apply_order_changes":
            if not confirmed_booking_id and self.draft.get("preorder_request") and self.reservation_buffer.get("date"):
                actions = args.get("actions") or []
                service = getattr(self.domain, "_native_service", None)
                if not service or not actions or any(a.get("name") not in {"add_order_item", "set_order_fulfillment", "set_order_notes"} for a in actions):
                    return {"status":"needs_clarification","instruction":"The pre-order is not saved. Prepare the requested items without removing any existing order."}
                if self.domain.state.items or any(a.get("name") == "set_order_fulfillment" and a.get("arguments", {}).get("fulfillment_type") != "dine_in" for a in actions):
                    return {"status":"needs_clarification","instruction":"Keep the existing pickup/delivery separate. Ask which order this request belongs to."}
                planned = []
                for action in actions:
                    if action["name"] != "add_order_item":
                        continue
                    values = action.get("arguments") or {}
                    match = await service.find_menu_item(str(values.get("item_name") or ""), **({"at":menu_at} if menu_at else {}))
                    item = match.get("match")
                    if not item or not item.get("available"):
                        return {"status":"needs_clarification","instruction":"That requested dish is not verified as available for the reservation time. Preserve all requested details and ask about that dish only.","dish":values.get("item_name"),"menu_checked_for":str(menu_at)}
                    planned.append({"name":item["name"],"quantity":values.get("quantity",1)})
                if not planned:
                    return {"status":"needs_clarification","instruction":"Supply the food items already requested; no food was saved."}
                self.staged_preorder = {**args, "actions":[a for a in actions if a["name"] != "set_order_fulfillment"]}
                self.draft["preorder_items"] = planned
                return {"status":"staged_not_ordered","proposed_preorder":planned,"instruction":"These items are retained for this reservation, not ordered. Read the proposed food back before the reservation approval, explaining that the table is confirmed first and food needs its own final confirmation. Do not ask the caller to repeat the items. Obtain the reservation readback now."}
            if (confirmed_booking_id and not confirmed_order and
                    re.search(r"\b(?:dine.?in|pre.?order|reservation)\b", self.input_text, re.I) and
                    not re.search(r"\b(?:pickup|delivery)\b", self.input_text, re.I)):
                actions = args.get("actions")
                if isinstance(actions, list):
                    non_fulfillment = [action for action in actions if isinstance(action, dict) and action.get("name") != "set_order_fulfillment"]
                    wants_fulfillment = len(non_fulfillment) != len(actions) or self.pending_dine_in_fulfillment
                    if wants_fulfillment and not non_fulfillment and not self.domain.state.items:
                        self.pending_dine_in_fulfillment = True
                        return {"status": "deferred", "instruction": "The reservation is confirmed, but no order item exists yet. Add the requested menu item first; dine-in fulfillment will be set immediately afterward. Do not retry fulfillment on its own."}
                    if wants_fulfillment or any(action.get("name") == "add_order_item" for action in non_fulfillment):
                        args["actions"] = [*non_fulfillment, {"name": "set_order_fulfillment", "arguments": {"fulfillment_type": "dine_in"}}]
                        self.pending_dine_in_fulfillment = False
            result, candidate = await self.apply_order_changes(args, call_id, generation, epoch)
            if candidate and (candidate.name != "get_order_summary" or wants_review(self.input_text)):
                result["readback_text"] = candidate.confirmation_text
            elif candidate:
                clear_pending_confirmation(self.domain.session_id, "confirm_order")
                await self.domain._persist_native_confirmation_state()
                result.pop("speech", None)
                result.pop("readback_text", None)
                result["exact_speech_required"] = False
                result["status"] = "draft_updated"
                if isinstance(result.get("facts"), dict):
                    result["facts"].pop("confirmation", None)
                result["instruction"] = "Only requested draft changes were applied. Briefly acknowledge those specific changes. Do not recite the full order. Ask the next missing detail or whether the caller wants anything else. Full readback and a later yes are required at checkout."

        else:
            args["session_id"] = self.domain.session_id
            if name == "create_booking" and confirmation_only:
                await self._execute_server_action(name, args, requires_confirmation=True)
                return self.auto_confirmation_result
            if name == "confirm_order" and self.heard_order:
                args["expected_draft_version"] = self.heard_order["draft_version"]
            outcome = await self.domain._run_tool(call_id, name, args, generation=generation)
            # Preparing a reservation and obtaining its authoritative readback
            # are one controller operation, not two optional model decisions.
            needs_reservation_readback = (
                name == "update_reservation_draft" and outcome.success and outcome.readback_verified
            ) or (
                name == "create_booking" and outcome.error == "confirmation_readback_not_released"
            )
            if name == "update_reservation_draft" and outcome.success and outcome.readback_verified:
                self.reservation_buffer.update({k:v for k,v in args.items() if k in {"name","phone","date","time","party_size"}})
            if needs_reservation_readback and generation == self.domain.interruptions.generation:
                self.domain._outcomes.append(outcome)
                outcome = await self.domain._run_tool(call_id + ":readback", "get_reservation_draft", {"session_id": self.domain.session_id}, generation=generation)
            outcome = self.domain._with_confirmation(outcome)
            if outcome.success and outcome.readback_verified:
                outcome = await self.domain._persist_committed_outcome(outcome)
            await self.domain._persist_native_confirmation_state()
            self.domain._outcomes.append(outcome)
            old = self.domain.state.version
            synced = await self.domain._sync_order_memory(generation=None)
            if synced and synced.version != old:
                outcome = replace(outcome, state_version=synced.version)
                self.domain._outcomes[-1] = outcome
            result = self.domain._model_tool_output(outcome)
            if outcome.error and any(word in outcome.error for word in ("confirmation", "affirmation", "approval")):
                result["status"] = "needs_confirmation"
                result["instruction"] = "This is an approval/readback requirement, not a system outage. Nothing is committed. Obtain a fresh authoritative readback, let it finish, and ask for approval. Do not promise to work in the background or repeatedly retry."
            if result.get("speech") == "Your request was applied.":
                result.pop("speech", None)
                result["exact_speech_required"] = False
                result["instruction"] = "Only draft fields were updated. Ask for the next missing detail naturally; this is not a confirmed reservation/order."
            if outcome.pending and outcome.confirmation_text:
                result["readback_text"] = outcome.confirmation_text
                result["instruction"] = "This is a pending proposal, not a confirmed booking/order. Speak readback_text exactly, then wait for a NEW caller turn approving it. Never claim it is already booked."
                if outcome.name == "get_reservation_draft" and isinstance(outcome.result, Mapping):
                    proposed = outcome.result.get("proposed") or {}
                    if isinstance(proposed, Mapping):
                        self.pending_booking_creation = {
                            "name": proposed.get("customer_name", ""), "phone": proposed.get("customer_phone", ""),
                            "date": proposed.get("date", ""), "time": proposed.get("time", ""),
                            "party_size": proposed.get("party_size", 0), "notes": proposed.get("notes", ""),
                        }
                if name == "update_confirmed_booking":
                    result["reservation_amendment_prompt"] = True
                    self.pending_booking_update = {key: value for key, value in args.items() if key not in {"session_id", "caller_confirmed"}}
            elif name == "update_confirmed_booking" and outcome.success and outcome.readback_verified:
                self.pending_booking_update = None
            elif name == "create_booking" and outcome.success and outcome.readback_verified:
                self.pending_booking_creation = None
            await self.emit({"type": "tool", "name": name, "success": outcome.success, "verified": outcome.readback_verified, "error": outcome.error})
        result["current_order"] = state_view(self.domain.state)
        result["captured_reservation_details"] = dict(self.reservation_buffer)
        result["retained_preorder"] = self.draft.get("preorder_items") or self.draft.get("preorder_request")
        result["menu_checked_for"] = menu_at.isoformat() if menu_at else "current restaurant time"
        current_reservation = await self.reservation_identity()
        result["current_reservation"] = {
            key: current_reservation.get(key)
            for key in ("booking_id", "status", "customer_name", "date", "time", "party_size", "dietary", "extra_notes")
            if current_reservation.get(key) not in (None, "")
        }
        self.function_results[call_id] = result
        if self.model_turn_open and self.utterance and self.utterance["epoch"] == epoch:
            self.utterance["version"] = self.domain.state.version
            self.utterance["outcomes"] = tuple(self.domain._outcomes)
        await self.emit({"type": "state", "order": state_view(self.domain.state), "draft": self.draft})
        return result

    async def tools_worker(self):
        while not self.closed:
            calls, epoch = await self.tool_queue.get()
            responses = []
            try:
                async with self.domain_lock:
                    for call in calls:
                        try:
                            result = await self.execute_function(call, epoch)
                        except Exception as exc:
                            result = {"status": "failed", "error": type(exc).__name__, "instruction": "No success is established. Ask for the missing detail or explain a connection problem; never claim completion."}
                        logging.getLogger(__name__).info(
                            "voice_tool session=%s tool=%s status=%s error=%s",
                            self.domain.session_id, call.get("name"), result.get("status"), result.get("error", ""))
                        responses.append({"id": call["id"], "name": call["name"], "response": result})
                # Speak authoritative amendment approval text server-side.
                proposal = next((r["response"].get("readback_text") for r in reversed(responses) if r["response"].get("readback_text") and r["response"].get("reservation_amendment_prompt") and self.pending_booking_update), None)
                if proposal and epoch == self.epoch:
                    self.block_output = True
                    for response in responses:
                        response["response"]["instruction"] = "The server is speaking the exact approval prompt. Do not repeat it. Wait for the caller."
                await self.provider_send({"toolResponse": {"functionResponses": responses}})
                if proposal and epoch == self.epoch:
                    from openai import AsyncOpenAI
                    async with AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"]) as speech:
                        self.speech_client = speech
                        try: await self.speak(proposal, epoch, item_id=self.input_id)
                        finally: self.speech_client = None
            finally:
                self.tool_queue.task_done()

    async def played(self, token):
        u = self.utterance
        if not u or token != u["token"] or not u["done"] or u["played"] or u["epoch"] != self.epoch or u["started"] is None:
            return False
        remaining = u["started"] + u["bytes"] / 48000 - time.monotonic()
        if remaining > .05:
            await self.emit({"type": "diagnostic", "stage": "playback_ack_too_early", "token": token,
                             "epoch": u["epoch"], "retry_after_ms": round(remaining * 1000)})
            return False
        async with self.domain_lock:
            if self.utterance is not u or u["epoch"] != self.epoch:
                return False
            u["played"] = True
            for outcome in u["outcomes"]:
                if not outcome.pending or not outcome.confirmation_text or u["version"] != self.domain.state.version or outcome.state_version != u["version"]:
                    continue
                if not reservation_readback_matches(outcome, u["text"]):
                    await self.emit({"type": "diagnostic", "stage": "confirmation", "code": "spoken_readback_not_matched"})
                    continue
                previous = self.domain._outcomes
                self.domain._outcomes = [outcome]
                try:
                    await self.domain._release_pending_readbacks(outcome.confirmation_text, u["version"])
                finally:
                    self.domain._outcomes = previous
                if outcome.name == "get_order_summary":
                    self.heard_order = {"draft_version": outcome.facts.get("draft_version"), "state_version": u["version"]}
            await self.emit({"type": "listening"})
        return True

    async def close(self):
        self.closed = True
        self.epoch += 1
        self.domain.interruptions.interrupt()
        self.utterance = None
        await self.socket.close()
        if self.reader_task:
            self.reader_task.cancel()
            with suppress(asyncio.CancelledError): await self.reader_task
        if self.worker_task:
            async with self.domain_lock:
                self.worker_task.cancel()
            with suppress(asyncio.CancelledError): await self.worker_task
        await self.domain.close()


async def connect_gemini_session(*, send, pace="natural", microphone="near_field"):
    from app.config import get_settings
    import websockets
    settings = get_settings()
    if settings.is_production or not settings.native_voice_realtime_enabled:
        raise RuntimeError("gemini_requires_development")
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError("gemini_api_key_missing")
    socket = await websockets.connect(ENDPOINT, additional_headers={"x-goog-api-key": key}, open_timeout=20, max_size=8000000)
    domain = NativeVoiceAdapter(session_id="gemini-" + uuid.uuid4().hex, transport=MemoryRealtimeTransport())
    return GeminiLiveSession(domain=domain, socket=socket, send=send, pace=pace)
