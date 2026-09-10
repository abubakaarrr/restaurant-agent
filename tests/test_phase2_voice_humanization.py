from __future__ import annotations

import asyncio
import csv
import json
from pathlib import Path

import pytest
from fastapi import WebSocketDisconnect

import app.retell_handler as handler
from app.behavior import BehaviorState, TurnObservation, reduce_behavior
from app.call_flags import clear_call_control, request_end_call, request_transfer
from app.spoken_delivery import (
    ResponseGenerationGate,
    SpokenTextBuffer,
    sanitize_spoken_text,
    spoken_text_violations,
)
from scripts.phase2_voice_evaluation import (
    load_plan,
    prepare,
    run_local_scenarios,
    summarize,
)


REQUIRED_SCENARIOS = {
    "greeting",
    "simple_faq",
    "menu_clarification",
    "allergy_question",
    "reservation_availability",
    "order_correction",
    "frustration",
    "interruption",
    "silence_incomplete_input",
    "unknown_request",
    "transfer_unavailable",
    "personal_identity",
}


def test_all_phase2_scenarios_execute_as_plain_concise_grounded_speech() -> None:
    plan = load_plan()
    results = run_local_scenarios(plan)

    assert {row["scenario_id"] for row in results} == REQUIRED_SCENARIOS
    assert all(row["passed"] for row in results)
    for result in results:
        assert result["source_refs"]
        for output in result["outputs"]:
            assert spoken_text_violations(output) == ()
            assert output.count("?") <= 1

    by_id = {row["scenario_id"]: row for row in results}
    assert by_id["greeting"]["outputs"] == [
        "Hi, you've reached Harbor & Hearth Kitchen. This is Avery. "
        "How can I help you today?"
    ]
    assert "1842 Market Street" in by_id["simple_faq"]["outputs"][0]
    assert "Which one did you mean?" in by_id["menu_clarification"]["outputs"][0]
    allergy = by_id["allergy_question"]["outputs"][0].casefold()
    assert "gluten" in allergy
    assert "can't guarantee zero cross-contact" in allergy
    reservation = by_id["reservation_availability"]["outputs"][0].casefold()
    assert "isn't booked until" in reservation
    correction = by_id["order_correction"]["outputs"][0].casefold()
    assert "one hearth burger" not in correction  # quantity stays explicit and machine-checkable
    assert "1 hearth burger, not 2" in correction
    assert all(word not in correction for word in ("updated", "changed", "added"))
    assert "virtual host" in by_id["personal_identity"]["outputs"][0].casefold()
    unknown = by_id["unknown_request"]["outputs"][0].casefold()
    assert "don't have that answer" in unknown
    assert "transfer" not in unknown and "connect" not in unknown
    interruption = by_id["interruption"]
    trace = interruption["handler_delivery_trace"]
    new_turn = next(
        index
        for index, event in enumerate(trace)
        if event == {"event": "customer_turn", "response_id": 71}
    )
    deliveries = [
        event for event in trace[new_turn + 1 :] if event["event"] == "assistant_delivery"
    ]
    assert deliveries
    assert all(event["response_id"] == 71 for event in deliveries)
    assert all(not event["end_call"] for event in deliveries)
    assert all(event["transfer_number"] is None for event in deliveries)


def test_first_interruption_immediately_raises_sensitivity_and_invalidates_old_id() -> None:
    gate = ResponseGenerationGate()
    gate.begin(3)
    assert gate.allows(3)

    gate.begin(4)
    directive = reduce_behavior(
        BehaviorState(),
        TurnObservation(text="Actually, make it seven.", interrupted=True),
    ).directive

    assert gate.allows(4)
    assert not gate.allows(3)
    assert directive.interruption_sensitivity == 0.9
    assert "interrupted" in directive.reasons
    assert "recent_interruptions" in directive.reasons


class InterruptingWebSocket:
    def __init__(self) -> None:
        self.sent: list[str] = []
        self.old_chunk_sent = asyncio.Event()
        self.receive_count = 0

    async def receive_text(self) -> str:
        self.receive_count += 1
        if self.receive_count == 1:
            return json.dumps(
                {
                    "interaction_type": "response_required",
                    "response_id": 70,
                    "transcript": [{"role": "user", "content": "Tell me the old answer"}],
                }
            )
        if self.receive_count == 2:
            await asyncio.wait_for(self.old_chunk_sent.wait(), timeout=1)
            return json.dumps(
                {
                    "interaction_type": "response_required",
                    "response_id": 71,
                    "transcript": [
                        {"role": "user", "content": "Tell me the old answer"},
                        {"role": "assistant", "content": "Old answer started"},
                        {"role": "user", "content": "Actually, make it seven"},
                    ],
                }
            )
        await asyncio.sleep(0.05)
        raise WebSocketDisconnect()

    async def send_text(self, payload: str) -> None:
        self.sent.append(payload)
        message = json.loads(payload)
        if (
            message.get("response_type") == "response"
            and message.get("response_id") == 70
            and message.get("content")
        ):
            self.old_chunk_sent.set()


class BehaviorStateInterruptWebSocket:
    def __init__(self) -> None:
        self.sent: list[str] = []
        self.old_send_started = asyncio.Event()
        self.new_response_complete = asyncio.Event()
        self.receive_count = 0

    async def receive_text(self) -> str:
        self.receive_count += 1
        if self.receive_count == 1:
            return json.dumps(
                {
                    "interaction_type": "response_required",
                    "response_id": 70,
                    "transcript": [
                        {"role": "user", "content": "Connect me to a person"}
                    ],
                }
            )
        if self.receive_count == 2:
            await asyncio.wait_for(self.old_send_started.wait(), timeout=1)
            return json.dumps(
                {
                    "interaction_type": "response_required",
                    "response_id": 71,
                    "transcript": [
                        {"role": "user", "content": "Connect me to a person"},
                        {"role": "user", "content": "What is your address?"},
                    ],
                }
            )
        await asyncio.wait_for(self.new_response_complete.wait(), timeout=1)
        raise WebSocketDisconnect()

    async def send_text(self, payload: str) -> None:
        message = json.loads(payload)
        if (
            message.get("response_type") == "response"
            and message.get("response_id") == 70
        ):
            self.old_send_started.set()
            await asyncio.Event().wait()
        self.sent.append(payload)
        if (
            message.get("response_type") == "response"
            and message.get("response_id") == 71
            and message.get("content_complete")
        ):
            self.new_response_complete.set()


@pytest.mark.asyncio
async def test_protocol_delivers_no_old_content_after_customer_interruption(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_load(_call_id: str) -> BehaviorState:
        return BehaviorState()

    async def fake_save(_call_id: str, _state: BehaviorState) -> None:
        return None

    async def fake_caller_turn(*_args, **_kwargs):
        return {"handled": False, "kind": "caller_turn", "affirmation": None}

    async def fake_stream(_call_id: str, user_text: str, *_args, **_kwargs):
        if "old answer" in user_text.casefold():
            yield "Old answer started."
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                # Model an upstream iterator that mishandles cancellation. The
                # handler's generation gate must still block its next chunk.
                pass
            yield "STALE CONTENT."
        else:
            yield "You want seven o'clock instead."

    monkeypatch.setattr(handler, "load_behavior_state", fake_load)
    monkeypatch.setattr(handler, "save_behavior_state", fake_save)
    monkeypatch.setattr(handler, "process_caller_turn", fake_caller_turn)
    monkeypatch.setattr(handler, "stream_agent_tokens", fake_stream)
    traces: list[tuple[str, dict]] = []

    def capture_trace(_call_id: str, event_type: str, **kwargs) -> None:
        traces.append((event_type, kwargs))

    monkeypatch.setattr(handler, "_record_background", capture_trace)

    websocket = InterruptingWebSocket()
    await handler.handle_retell_connection(websocket, "phase2-interruption")
    responses = [
        json.loads(payload)
        for payload in websocket.sent
        if json.loads(payload).get("response_type") == "response"
    ]

    first_new = next(
        index for index, row in enumerate(responses) if row["response_id"] == 71
    )
    assert all(row["response_id"] != 70 for row in responses[first_new:])
    assert not any("STALE CONTENT" in row.get("content", "") for row in responses)
    assert responses[-1]["response_id"] == 71
    assert responses[-1]["content_complete"] is True
    suppressed = [row for row in traces if row[0] == "stale_response_suppressed"]
    assert suppressed == [
        (
            "stale_response_suppressed",
            {
                "response_id": 70,
                "payload": {"active_response_id": 71, "stage": "stream_chunk"},
            },
        )
    ]


@pytest.mark.asyncio
async def test_cancelled_behavior_state_does_not_control_the_new_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    saved_states: list[BehaviorState] = []

    async def fake_load(_call_id: str) -> BehaviorState:
        return BehaviorState()

    async def fake_save(_call_id: str, state: BehaviorState) -> None:
        saved_states.append(state)

    async def fake_caller_turn(*_args, **_kwargs):
        return {"handled": False, "kind": "caller_turn", "affirmation": None}

    async def fake_stream(*_args, **_kwargs):
        yield "We're at 1842 Market Street."

    monkeypatch.setattr(handler, "load_behavior_state", fake_load)
    monkeypatch.setattr(handler, "save_behavior_state", fake_save)
    monkeypatch.setattr(handler, "process_caller_turn", fake_caller_turn)
    monkeypatch.setattr(handler, "stream_agent_tokens", fake_stream)
    monkeypatch.setattr(handler, "_record_background", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        "app.behavior.resolve_handoff_destination",
        lambda _reason: {
            "owner": "staff",
            "channel": "voice_transfer",
            "transfer_number": "+15035550149",
            "can_transfer": True,
        },
    )

    websocket = BehaviorStateInterruptWebSocket()
    await handler.handle_retell_connection(websocket, "phase2-state-interruption")
    await asyncio.sleep(0)
    responses = [
        json.loads(payload)
        for payload in websocket.sent
        if json.loads(payload).get("response_type") == "response"
    ]

    assert responses
    assert all(row["response_id"] == 71 for row in responses)
    assert "1842 Market Street" in "".join(
        row.get("content", "") for row in responses
    )
    assert all("transfer_number" not in row for row in responses)
    assert all("end_call" not in row for row in responses)
    assert saved_states
    assert all(state.terminal_control is None for state in saved_states)


@pytest.mark.parametrize("control", ["transfer", "end"])
@pytest.mark.asyncio
async def test_cancelled_turn_control_cannot_leak_into_next_response(
    monkeypatch: pytest.MonkeyPatch,
    control: str,
) -> None:
    call_id = f"phase2-cancelled-{control}"

    async def fake_load(_call_id: str) -> BehaviorState:
        return BehaviorState()

    async def fake_save(_call_id: str, _state: BehaviorState) -> None:
        return None

    async def fake_caller_turn(*_args, **_kwargs):
        return {"handled": False, "kind": "caller_turn", "affirmation": None}

    async def fake_stream(_call_id: str, user_text: str, *_args, **_kwargs):
        if "old answer" in user_text.casefold():
            if control == "transfer":
                request_transfer(call_id, "human_requested", "+14155550123")
            else:
                request_end_call(call_id)
            yield "Old answer started."
            await asyncio.Event().wait()
        else:
            yield "The new answer."

    monkeypatch.setattr(handler, "load_behavior_state", fake_load)
    monkeypatch.setattr(handler, "save_behavior_state", fake_save)
    monkeypatch.setattr(handler, "process_caller_turn", fake_caller_turn)
    monkeypatch.setattr(handler, "stream_agent_tokens", fake_stream)
    monkeypatch.setattr(handler, "_record_background", lambda *args, **kwargs: None)

    try:
        websocket = InterruptingWebSocket()
        await handler.handle_retell_connection(websocket, call_id)
        completions = [
            json.loads(payload)
            for payload in websocket.sent
            if json.loads(payload).get("response_type") == "response"
            and json.loads(payload).get("content_complete")
        ]
        new_completion = next(
            row for row in completions if row["response_id"] == 71
        )
        assert "transfer_number" not in new_completion
        assert "transfer_caller_id" not in new_completion
        assert "end_call" not in new_completion
    finally:
        clear_call_control(call_id)


def test_spoken_list_sanitizer_preserves_grounded_numbers() -> None:
    confirmation = "You want 1 Hearth Burger, not 2. Pickup is at 7:00."
    assert sanitize_spoken_text(confirmation) == confirmation

    numbered = "Options: 1. Hearth Burger 2. Herb Chicken Plate"
    sanitized = sanitize_spoken_text(numbered)
    assert "1," in sanitized and "2," in sanitized
    assert "1." not in sanitized and "2." not in sanitized
    assert spoken_text_violations(sanitized) == ()

    order = "Order: 1. Burger. 2. Salad. Confirmation 123. Please keep it."
    cleaned_order = sanitize_spoken_text(order)
    assert "1, Burger" in cleaned_order and "2, Salad" in cleaned_order
    assert "Confirmation 123." in cleaned_order

    hours = "We're open Tuesday - Thursday, 5 - 10 p.m."
    assert sanitize_spoken_text(hours) == hours

    mixed = "Hours: - pickup, 5 - 10 p.m. - delivery"
    cleaned_mixed = sanitize_spoken_text(mixed)
    assert "5 - 10 p.m." in cleaned_mixed
    assert "5. 10" not in cleaned_mixed
    assert spoken_text_violations(cleaned_mixed) == ()

    inline = sanitize_spoken_text("You can choose - fries - salad.")
    assert inline == "You can choose. fries. salad."
    assert spoken_text_violations(inline) == ()

    meal_periods = sanitize_spoken_text(
        "Hours: - lunch, 11 a.m. - 2 p.m. - dinner."
    )
    assert meal_periods == "Hours: lunch, 11 a.m. - 2 p.m. dinner."
    assert ".." not in meal_periods
    assert spoken_text_violations(meal_periods) == ()

    uppercase_hours = sanitize_spoken_text(
        "Hours: - Tuesday, 11:30 AM - 10 PM - Friday."
    )
    assert "11:30 AM - 10 PM" in uppercase_hours
    assert "11:30 AM. 10 PM" not in uppercase_hours
    assert spoken_text_violations(uppercase_hours) == ()

    inline_ordered = sanitize_spoken_text("You can choose 1. burger 2. salad.")
    assert inline_ordered == "You can choose 1, burger 2, salad."
    assert spoken_text_violations(inline_ordered) == ()
    assert "Confirmation 123." in sanitize_spoken_text(
        "Confirmation 123. Please keep it."
    )

    later_sequence = sanitize_spoken_text(
        "Remaining choices: 3. soup 4. salad."
    )
    assert later_sequence == "Remaining choices: 3, soup 4, salad."
    assert spoken_text_violations(later_sequence) == ()

    grounded_numbers = (
        "Confirmation 123. Pickup is 7:00. We're open 5 - 10 p.m."
    )
    assert sanitize_spoken_text(grounded_numbers) == grounded_numbers


def test_response_transport_sanitizes_inline_unordered_lists() -> None:
    event = json.loads(
        handler._response_event(
            72,
            "Your sides are: - fries - salad",
            complete=True,
        )
    )
    assert event["content"] == "Your sides are: fries. salad"
    assert spoken_text_violations(event["content"]) == ()

    emphasis = json.loads(handler._response_event(73, "_Special_", complete=True))
    assert emphasis["content"] == "Special"
    assert spoken_text_violations(emphasis["content"]) == ()

    inline = json.loads(
        handler._response_event(74, "You can choose - fries - salad.", complete=True)
    )
    assert inline["content"] == "You can choose. fries. salad."
    assert spoken_text_violations(inline["content"]) == ()

    thematic_text = "Today's specials\n---\nHearth Burger."
    assert "markdown" in spoken_text_violations(thematic_text)
    thematic = json.loads(handler._response_event(75, thematic_text, complete=True))
    assert thematic["content"] == "Today's specials\nHearth Burger."
    assert spoken_text_violations(thematic["content"]) == ()

    blockquote_text = ">Today's special"
    assert "markdown" in spoken_text_violations(blockquote_text)
    blockquote = json.loads(
        handler._response_event(76, blockquote_text, complete=True)
    )
    assert blockquote["content"] == "Today's special"
    assert spoken_text_violations(blockquote["content"]) == ()

    glyph_text = "Options: • fries • salad."
    assert "markdown" in spoken_text_violations(glyph_text)
    glyphs = json.loads(handler._response_event(77, glyph_text, complete=True))
    assert glyphs["content"] == "Options: fries. salad."
    assert spoken_text_violations(glyphs["content"]) == ()

    setext_text = "Today's specials\n===\nHearth Burger."
    assert "markdown" in spoken_text_violations(setext_text)
    setext = json.loads(handler._response_event(78, setext_text, complete=True))
    assert setext["content"] == "Today's specials\nHearth Burger."
    assert spoken_text_violations(setext["content"]) == ()

    for response_id, marker in enumerate(("-", "--"), start=79):
        short_setext_text = f"Today's specials\n{marker}\nHearth Burger."
        assert "markdown" in spoken_text_violations(short_setext_text)
        short_setext = json.loads(
            handler._response_event(response_id, short_setext_text, complete=True)
        )
        assert short_setext["content"] == "Today's specials\nHearth Burger."
        assert spoken_text_violations(short_setext["content"]) == ()

    nested_quote_text = ">>Today's special is Hearth Burger."
    assert "markdown" in spoken_text_violations(nested_quote_text)
    nested_quote = json.loads(
        handler._response_event(81, nested_quote_text, complete=True)
    )
    assert nested_quote["content"] == "Today's special is Hearth Burger."
    assert spoken_text_violations(nested_quote["content"]) == ()

    fenced_text = "~~~menu\nToday's special is Hearth Burger.\n~~~"
    assert "markdown" in spoken_text_violations(fenced_text)
    fenced = json.loads(handler._response_event(82, fenced_text, complete=True))
    assert fenced["content"] == "Today's special is Hearth Burger.\n"
    assert spoken_text_violations(fenced["content"]) == ()

    single_glyph_text = "Today's side is • fries."
    assert "markdown" in spoken_text_violations(single_glyph_text)
    single_glyph = json.loads(
        handler._response_event(83, single_glyph_text, complete=True)
    )
    assert single_glyph["content"] == "Today's side is fries."
    assert spoken_text_violations(single_glyph["content"]) == ()


def test_stream_buffer_sanitizes_split_markdown_and_list_syntax() -> None:
    plain = SpokenTextBuffer()
    assert plain.feed("I") == ()
    assert plain.feed(" can") == ()
    early = plain.feed(" help")
    assert early
    assert "".join((*early, *plain.feed("."), *plain.flush())) == "I can help."

    buffer = SpokenTextBuffer()
    chunks = [
        "Options:",
        " -",
        " Fries",
        " -",
        " Salad.",
        " **Special**.",
        " See [our menu]",
        "(https://example.com).",
    ]
    delivered = [part for chunk in chunks for part in buffer.feed(chunk)]
    delivered.extend(buffer.flush())

    assert "".join(delivered) == "Options: Fries. Salad. Special. See our menu."
    assert all(spoken_text_violations(part) == () for part in delivered)

    inline = SpokenTextBuffer()
    assert inline.feed("You can choose - crispy fries.") == ()
    inline_delivery = inline.feed(" - salad.")
    assert "".join((*inline_delivery, *inline.flush())) == (
        "You can choose. crispy fries. salad."
    )

    ordered = SpokenTextBuffer()
    assert ordered.feed("You can choose 1. burger.") == ()
    ordered_delivery = ordered.feed(" 2. salad.")
    assert "".join((*ordered_delivery, *ordered.flush())) == (
        "You can choose 1, burger. 2, salad."
    )


class ListWebSocket:
    def __init__(self, messages: list[dict]) -> None:
        self.messages = [json.dumps(row) for row in messages]
        self.sent: list[str] = []

    async def receive_text(self) -> str:
        if self.messages:
            return self.messages.pop(0)
        raise WebSocketDisconnect()

    async def send_text(self, payload: str) -> None:
        self.sent.append(payload)


@pytest.mark.asyncio
async def test_initial_empty_turn_greets_once_and_later_incomplete_input_repairs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_load(_call_id: str) -> BehaviorState:
        return BehaviorState()

    async def fake_save(_call_id: str, _state: BehaviorState) -> None:
        return None

    monkeypatch.setattr(handler, "load_behavior_state", fake_load)
    monkeypatch.setattr(handler, "save_behavior_state", fake_save)
    monkeypatch.setattr(handler, "_record_background", lambda *args, **kwargs: None)
    websocket = ListWebSocket(
        [
            {"interaction_type": "response_required", "response_id": 1, "transcript": []},
            {"interaction_type": "response_required", "response_id": 2, "transcript": []},
        ]
    )

    await handler.handle_retell_connection(websocket, "phase2-incomplete")
    responses = [
        json.loads(payload)
        for payload in websocket.sent
        if json.loads(payload).get("response_type") == "response"
    ]
    assert len(responses) == 2
    assert "you've reached" in responses[0]["content"]
    assert "didn't catch that" in responses[1]["content"]
    assert "you've reached" not in responses[1]["content"]


def test_offline_evaluation_is_reproducible_and_missing_metrics_stay_null(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    result = prepare(first)
    prepare(second)

    for name in (
        "local-scenario-results.jsonl",
        "provider-measurements.csv",
        "blind-rating-form.csv",
        "randomization-key.csv",
        "results.json",
    ):
        assert (first / name).read_bytes() == (second / name).read_bytes()

    assert result["prepared"] == {
        "scenario_count": 12,
        "turns_per_arm": 120,
        "blind_comparisons": 120,
        "randomization_seed": 20260909,
    }
    assert result["arms"]["retell_native_clone"]["completed_scenarios"] == 0
    assert result["human_preference"]["rater_count"] == 0
    assert result["baseline_vs_clone"]["clone_score"] is None
    assert result["baseline_vs_clone"]["baseline_score"] is None
    for arm_id, arm in result["arms"].items():
        assert arm["provider_audio_score"] is None
        assert arm["clone_audio_artifact_ref"] is None
        assert result["human_preference"]["arm_scores"][arm_id][
            "preference_wins"
        ] is None
        assert arm["event_counts"]["timeout_count"] is None
        for metric in arm["latency_ms"].values():
            assert metric["availability"] == "missing"
            assert metric["median_ms"] is None
            assert metric["p95_ms"] is None


def test_aggregator_uses_declared_latency_and_blinded_score_mappings(
    tmp_path: Path,
) -> None:
    output = tmp_path / "evaluation"
    prepare(output)
    plan = load_plan()
    local_results = run_local_scenarios(plan)
    with (output / "provider-measurements.csv").open(newline="", encoding="utf-8") as handle:
        measurements = list(csv.DictReader(handle))
    for row in measurements:
        if row["scenario_id"] != "greeting" or row["repeat"] != "1":
            continue
        first_audio = 1500 if row["arm"] == "current_retell_baseline" else 1600
        row.update(
            {
                "status": "completed",
                "missing_reason": "",
                "customer_speech_end_ms": "1000",
                "agent_request_start_ms": "1010",
                "agent_text_first_token_ms": "1110",
                "provider_request_start_ms": "1120",
                "provider_response_ms": "1320",
                "first_audio_playback_ms": str(first_audio),
                "final_audio_completion_ms": str(first_audio + 900),
                "stale_response_incidents": "0",
                "error_count": "0",
                "timeout_count": "0",
                "fallback_count": "0",
            }
        )
    with (output / "blind-rating-form.csv").open(newline="", encoding="utf-8") as handle:
        ratings = list(csv.DictReader(handle))
    with (output / "randomization-key.csv").open(newline="", encoding="utf-8") as handle:
        keys = list(csv.DictReader(handle))
    rating = ratings[0]
    rating.update(
        {
            "sample_a_ref": "safe-local-reference-a",
            "sample_b_ref": "safe-local-reference-b",
            "rater_id": "rater-001",
            "preferred_sample": "B",
        }
    )
    for dimension in plan["preference_dimensions"]:
        rating[f"sample_a_{dimension}"] = "4"
        rating[f"sample_b_{dimension}"] = "5"

    result = summarize(plan, local_results, measurements, ratings, keys)
    assert result["arms"]["current_retell_baseline"]["latency_ms"]["first_audio"][
        "median_ms"
    ] == 500
    assert result["arms"]["retell_native_clone"]["latency_ms"]["first_audio"][
        "median_ms"
    ] == 600
    assert result["baseline_vs_clone"]["first_audio_delta_ms"] == 100
    assert result["human_preference"]["rater_count"] == 1
    mapping = keys[0]
    assert result["human_preference"]["preferred_arm"] == mapping["sample_b_arm"]
    assert result["human_preference"]["arm_scores"][mapping["sample_a_arm"]][
        "overall_mean"
    ] == 4
    assert result["human_preference"]["arm_scores"][mapping["sample_b_arm"]][
        "overall_mean"
    ] == 5
