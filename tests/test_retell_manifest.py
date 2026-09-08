import json
from pathlib import Path
from urllib.parse import urlparse

from app.main import app


ROOT = Path(__file__).resolve().parent.parent


def test_retell_manifest_references_real_routes_and_prompts() -> None:
    manifest = json.loads(
        (ROOT / "config" / "retell-agent.pilot.json").read_text(encoding="utf-8")
    )
    assert manifest["directly_importable"] is False
    assert manifest["phase1_limitations"]["managed_cancellation_reversal"].startswith(
        "unsupported"
    )
    assert all(
        tool["name"] != "process_caller_turn"
        for tool in manifest["conversation_flow_review"]["tool_endpoint_placeholders"]
    )

    routes = {
        (route.path, method)
        for route in app.routes
        for method in (getattr(route, "methods", None) or set())
    }
    for tool in manifest["conversation_flow_review"]["tool_endpoint_placeholders"]:
        path = urlparse(tool["url"].replace("<PILOT_API_HOST>", "example.com")).path
        assert (path, tool["method"]) in routes
        assert "X-Voice-Tool-Secret" in tool["authentication_header"]
        if tool["name"] in {
            "create_reservation",
            "cancel_booking",
            "add_guest_note",
            "add_order_item",
            "set_order_fulfillment",
            "set_order_notes",
            "update_order_item",
            "remove_order_item",
            "confirm_order",
            "update_confirmed_booking",
        }:
            assert "idempotency_header" in tool

    webhook_path = urlparse(
        manifest["lifecycle_webhooks"]["webhook_url"].replace(
            "<PILOT_API_HOST>", "example.com"
        )
    ).path
    assert (webhook_path, "POST") in routes

    for relative_path in manifest["conversation_flow_review"][
        "version_controlled_prompts"
    ]:
        assert (ROOT / relative_path).is_file()
