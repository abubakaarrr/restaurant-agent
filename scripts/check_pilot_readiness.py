#!/usr/bin/env python3
"""Fail-closed readiness check that never prints secret values."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

from dotenv import dotenv_values


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from app.config import settings  # noqa: E402
from app.security import is_e164  # noqa: E402


def check() -> dict[str, Any]:
    env = {
        key: value
        for key, value in dotenv_values(ROOT / ".env").items()
        if value is not None
    }
    env.update(os.environ)
    bakeoff_path = ROOT / "artifacts" / "voice-bakeoff" / "report.json"
    bakeoff_complete = False
    if bakeoff_path.is_file():
        try:
            bakeoff = json.loads(bakeoff_path.read_text(encoding="utf-8"))
            bakeoff_complete = bool(
                bakeoff.get("status") == "complete"
                and (bakeoff.get("decision") or {}).get("selected_arm")
            )
        except (json.JSONDecodeError, OSError):
            bakeoff_complete = False
    checks = {
        "retell_api_key": bool(settings.retell_api_key),
        "retell_agent_id": bool(settings.retell_agent_id),
        "voice_tool_secret": len(settings.voice_tool_secret) >= 24,
        "staff_transfer_number": is_e164(settings.staff_transfer_number),
        "retell_phone_number": is_e164(settings.retell_phone_number),
        "widget_public_key": bool(settings.retell_public_key),
        "widget_domains": bool(settings.widget_domain_list),
        "recaptcha_site_key": bool(settings.recaptcha_site_key),
        "elevenlabs_api_key_for_challenger": bool(env.get("ELEVENLABS_API_KEY")),
        "elevenlabs_voice_or_agent": bool(
            env.get("ELEVENLABS_VOICE_ID") or env.get("ELEVENLABS_AGENT_ID")
        ),
        "baseline_report": (ROOT / "artifacts" / "retell-baseline.json").is_file(),
        "completed_bakeoff_report": bakeoff_complete,
        "live_writes_safely_disabled": not settings.voice_live_writes_enabled,
    }
    external_activation = (
        checks["retell_api_key"]
        and checks["retell_agent_id"]
        and checks["voice_tool_secret"]
        and checks["staff_transfer_number"]
        and checks["retell_phone_number"]
    )
    website_ready = (
        checks["widget_public_key"]
        and checks["widget_domains"]
        and checks["recaptcha_site_key"]
    )
    bakeoff_ready = (
        checks["elevenlabs_api_key_for_challenger"]
        and checks["elevenlabs_voice_or_agent"]
        and checks["completed_bakeoff_report"]
    )
    return {
        "checks": checks,
        "external_activation_ready": external_activation,
        "website_activation_ready": website_ready,
        "voice_bakeoff_complete": bakeoff_ready,
        "code_and_local_tests": "Run pytest and integration suite separately.",
    }


def main() -> int:
    report = check()
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if all(
        (
            report["external_activation_ready"],
            report["website_activation_ready"],
            report["voice_bakeoff_complete"],
        )
    ) else 2


if __name__ == "__main__":
    raise SystemExit(main())
