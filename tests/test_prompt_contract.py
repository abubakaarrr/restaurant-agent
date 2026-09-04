from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def test_legacy_prompt_requires_truthful_identity_and_confirmed_writes() -> None:
    prompt = (ROOT / "app" / "prompts" / "system.md").read_text(encoding="utf-8")
    assert "Never volunteer that you are AI" in prompt
    assert "Never repeat that script later" in prompt
    assert "virtual host. How can I help?" not in prompt
    assert "I'm the restaurant's automated host" not in prompt
    assert "caller_confirmed=true" in prompt
    assert "caller_approved_full_readback=true" in prompt
    assert prompt.index("call `get_order_summary`") < prompt.index(
        "call `confirm_order`"
    )
    assert "Never call `confirm_order` before" in prompt
    assert "add_guest_note" in prompt
    assert "ordinary special requests" in prompt
    assert "update_reservation_draft" in prompt
    assert "update_confirmed_booking" in prompt
    assert "log_unknown_question" in prompt
    assert "Latest user message wins" in prompt
    assert "Forget the fifth person" in prompt
    assert "Never transfer for a name spelling" in prompt
    assert "Do you serve water when I arrive" in prompt
    assert "Never call `request_handoff` for a name fix" in prompt
    assert "When an occasion is mentioned, acknowledge it warmly" in prompt
    assert "You're down as Hamza" in prompt
    assert "Never repeat that script later" in prompt


def test_managed_flow_prompts_scope_transactions_and_handoff() -> None:
    global_prompt = (
        ROOT / "app" / "prompts" / "retell" / "global.md"
    ).read_text(encoding="utf-8")
    reservation = (
        ROOT / "app" / "prompts" / "retell" / "reservation.md"
    ).read_text(encoding="utf-8")
    order = (ROOT / "app" / "prompts" / "retell" / "order.md").read_text(
        encoding="utf-8"
    )
    handoff = (
        ROOT / "app" / "prompts" / "retell" / "handoff.md"
    ).read_text(encoding="utf-8")

    assert "Never infer age, accent, disability" in global_prompt
    assert "explicit yes" in reservation
    assert "idempotency" in reservation
    assert "exact draft version" in order
    assert "Never accept" not in handoff  # wording is fixed and auditable
    assert "never provide" in handoff
