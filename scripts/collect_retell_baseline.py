#!/usr/bin/env python3
"""Collect a PII-free latency baseline from existing Retell call history."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
import sys
from pathlib import Path
from statistics import fmean
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from app.config import settings


API_URL = "https://api.retellai.com/v3/list-calls"


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    position = (len(values) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return values[lower]
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def latency_values(call: dict[str, Any], key: str) -> list[float]:
    metric = (call.get("latency") or {}).get(key) or {}
    values = metric.get("values") or []
    return [
        float(value)
        for value in values
        if isinstance(value, (int, float)) and math.isfinite(float(value))
    ]


def build_report(
    calls: list[dict[str, Any]],
    agent_id: str,
    agent: dict[str, Any] | None = None,
) -> dict[str, Any]:
    e2e = [value for call in calls for value in latency_values(call, "e2e")]
    llm = [value for call in calls for value in latency_values(call, "llm")]
    websocket_rtt = [
        value
        for call in calls
        for value in latency_values(call, "llm_websocket_network_rtt")
    ]
    tts = [value for call in calls for value in latency_values(call, "tts")]
    ended = [call for call in calls if call.get("call_status") == "ended"]
    analyzed = [
        call
        for call in calls
        if isinstance(call.get("call_analysis") or call.get("analysis"), dict)
    ]
    successful = sum(
        bool((call.get("call_analysis") or call.get("analysis") or {}).get("call_successful"))
        for call in analyzed
    )

    def stats(values: list[float]) -> dict[str, Any]:
        return {
            "samples": len(values),
            "mean_ms": fmean(values) if values else None,
            "p50_ms": percentile(values, 0.50),
            "p95_ms": percentile(values, 0.95),
        }

    safe_agent = None
    if agent:
        response_engine = agent.get("response_engine") or {}
        safe_agent = {
            "response_engine_type": response_engine.get("type"),
            "voice_id_fingerprint": (
                str(agent.get("voice_id"))[:8] + "..."
                if agent.get("voice_id")
                else None
            ),
            "voice_model": agent.get("voice_model"),
            "language": agent.get("language"),
            "responsiveness": agent.get("responsiveness"),
            "interruption_sensitivity": agent.get("interruption_sensitivity"),
            "enable_backchannel": agent.get("enable_backchannel"),
            "denoising_mode": agent.get("denoising_mode"),
            "data_storage_setting": agent.get("data_storage_setting"),
        }
    return {
        "source": "Retell v3 list-calls (latest available calls for configured agent)",
        "collected_at": datetime.now(timezone.utc).isoformat(),
        "agent_id_fingerprint": agent_id[:6] + "..." + agent_id[-4:],
        "contains_pii": False,
        "calls_returned": len(calls),
        "ended_calls": len(ended),
        "analyzed_calls": len(analyzed),
        "successful_analyzed_calls": successful,
        "success_rate": successful / len(analyzed) if analyzed else None,
        "agent_configuration": safe_agent,
        "latency": {
            "e2e": stats(e2e),
            "llm": stats(llm),
            "llm_websocket_network_rtt": stats(websocket_rtt),
            "tts": stats(tts),
        },
        "limitations": [
            "Existing calls may use the legacy custom-LLM agent and are not a managed-flow bakeoff.",
            "Retell e2e excludes the final network trip from Retell to the caller.",
            "No transcript, phone number, recording URL, or caller metadata is retained.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=100)
    args = parser.parse_args()
    api_key = (os.environ.get("RETELL_API_KEY") or settings.retell_api_key).strip()
    agent_id = (os.environ.get("RETELL_AGENT_ID") or settings.retell_agent_id).strip()
    if not api_key or not agent_id:
        raise SystemExit("RETELL_API_KEY and RETELL_AGENT_ID are required")
    limit = min(max(args.limit, 1), 1000)
    with httpx.Client(timeout=30) as client:
        response = client.post(
            API_URL,
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "filter_criteria": {"agent": [{"agent_id": agent_id}]},
                "sort_order": "descending",
                "limit": limit,
            },
        )
        agent_response = client.get(
            f"https://api.retellai.com/get-agent/{agent_id}",
            headers={"Authorization": f"Bearer {api_key}"},
        )
    if response.is_error:
        raise SystemExit(f"Retell list-calls failed with HTTP {response.status_code}")
    payload = response.json()
    calls = payload.get("items")
    if not isinstance(calls, list):
        raise SystemExit("Retell response did not contain an items array")
    agent = agent_response.json() if not agent_response.is_error else None
    report = build_report(calls, agent_id, agent)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"wrote PII-free baseline for {len(calls)} calls to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
