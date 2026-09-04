from __future__ import annotations

import csv
from pathlib import Path

import pytest

from scripts.voice_bakeoff import (
    initialize_scorecards,
    summarize,
)


def test_bakeoff_refuses_placeholder_rows(tmp_path: Path) -> None:
    measurements = tmp_path / "measurements.csv"
    preferences = tmp_path / "preferences.csv"
    initialize_scorecards(measurements, preferences)
    with pytest.raises(ValueError, match="real provider call_id"):
        summarize(measurements, preferences)


def test_challenger_requires_gates_and_blind_preference_margin(
    tmp_path: Path,
) -> None:
    measurements = tmp_path / "measurements.csv"
    preferences = tmp_path / "preferences.csv"
    initialize_scorecards(measurements, preferences)

    with measurements.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
        fields = list(rows[0])
    with measurements.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, row in enumerate(rows):
            row.update(
                {
                    "call_id": f"real-call-{index}",
                    "end_of_user_speech_ms": "1000",
                    "first_agent_audio_ms": "1700",
                    "task_success": "true",
                    "transfer_required": (
                        "true"
                        if row["scenario_id"]
                        in {"explicit_human", "severe_allergy", "complaint", "tool_outage"}
                        else "false"
                    ),
                    "transfer_success": "true",
                    "duplicate_writes": "0",
                    "ungrounded_answers": "0",
                    "naturalness_score_1_to_5": (
                        "4.8"
                        if row["arm"] == "direct_elevenagents_v3"
                        else "4.2"
                    ),
                    "cost_usd": "0.15",
                }
            )
            if row["scenario_id"] == "fast_interruption":
                row["barge_in_started_ms"] = "2000"
                row["agent_audio_stopped_ms"] = "2250"
            writer.writerow(row)

    with preferences.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("comparison_id", "retell_arm", "winner"),
        )
        writer.writeheader()
        for index in range(20):
            writer.writerow(
                {
                    "comparison_id": f"blind-{index}",
                    "retell_arm": "retell_elevenlabs_flash",
                    "winner": "direct_elevenagents_v3",
                }
            )
            writer.writerow(
                {
                    "comparison_id": f"blind-expressive-{index}",
                    "retell_arm": "retell_platform_expressive",
                    "winner": "direct_elevenagents_v3",
                }
            )

    report = summarize(measurements, preferences)
    assert all(result["all_gates_pass"] for result in report["arms"].values())
    assert report["decision"]["selected_arm"] == "direct_elevenagents_v3"
    assert report["decision"]["challenger_promoted"] is True
