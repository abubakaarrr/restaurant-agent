#!/usr/bin/env python3
"""Create and score a provider-neutral, evidence-only voice bakeoff."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SCENARIOS = ROOT / "config" / "voice-bakeoff-scenarios.json"
MEASUREMENT_FIELDS = (
    "arm",
    "scenario_id",
    "repeat",
    "call_id",
    "end_of_user_speech_ms",
    "first_agent_audio_ms",
    "barge_in_started_ms",
    "agent_audio_stopped_ms",
    "task_success",
    "transfer_required",
    "transfer_success",
    "duplicate_writes",
    "ungrounded_answers",
    "naturalness_score_1_to_5",
    "cost_usd",
    "reviewer_notes",
)
PREFERENCE_FIELDS = ("comparison_id", "retell_arm", "winner")


def load_spec(path: Path = DEFAULT_SCENARIOS) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def initialize_scorecards(
    measurements_path: Path,
    preferences_path: Path,
    *,
    repeats: int = 4,
    spec_path: Path = DEFAULT_SCENARIOS,
) -> None:
    spec = load_spec(spec_path)
    measurements_path.parent.mkdir(parents=True, exist_ok=True)
    with measurements_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MEASUREMENT_FIELDS)
        writer.writeheader()
        for arm in spec["arms"]:
            for scenario in spec["scenarios"]:
                for repeat in range(1, repeats + 1):
                    writer.writerow(
                        {
                            "arm": arm,
                            "scenario_id": scenario["id"],
                            "repeat": repeat,
                        }
                    )
    with preferences_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=PREFERENCE_FIELDS)
        writer.writeheader()


def _required_float(row: dict[str, str], field: str) -> float:
    try:
        value = float(row[field])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Missing numeric {field} for call {row.get('call_id')}") from exc
    if not math.isfinite(value):
        raise ValueError(f"Non-finite {field} for call {row.get('call_id')}")
    return value


def _optional_float(row: dict[str, str], field: str) -> float | None:
    value = (row.get(field) or "").strip()
    return _required_float(row, field) if value else None


def _bool(row: dict[str, str], field: str, *, optional: bool = False) -> bool | None:
    value = (row.get(field) or "").strip().casefold()
    if optional and not value:
        return None
    if value in {"1", "true", "yes", "pass"}:
        return True
    if value in {"0", "false", "no", "fail"}:
        return False
    raise ValueError(f"Invalid boolean {field} for call {row.get('call_id')}")


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def summarize(
    measurements_path: Path,
    preferences_path: Path,
    *,
    spec_path: Path = DEFAULT_SCENARIOS,
) -> dict[str, Any]:
    spec = load_spec(spec_path)
    expected_arms = set(spec["arms"])
    rows = list(csv.DictReader(measurements_path.open(encoding="utf-8")))
    if not rows:
        raise ValueError("Measurement scorecard is empty")
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        arm = row.get("arm", "")
        if arm not in expected_arms:
            raise ValueError(f"Unknown bakeoff arm: {arm}")
        if not (row.get("call_id") or "").strip():
            raise ValueError(
                "Scorecard is incomplete; every row needs a real provider call_id"
            )
        grouped[arm].append(row)

    gates = spec["release_gates"]
    arm_results: dict[str, Any] = {}
    for arm in spec["arms"]:
        arm_rows = grouped.get(arm, [])
        first_audio: list[float] = []
        barge_stop: list[float] = []
        task_successes = 0
        required_transfers = 0
        transfer_successes = 0
        duplicate_writes = 0
        ungrounded_answers = 0
        naturalness: list[float] = []
        costs: list[float] = []
        for row in arm_rows:
            first_audio.append(
                _required_float(row, "first_agent_audio_ms")
                - _required_float(row, "end_of_user_speech_ms")
            )
            barge_started = _optional_float(row, "barge_in_started_ms")
            audio_stopped = _optional_float(row, "agent_audio_stopped_ms")
            if barge_started is not None or audio_stopped is not None:
                if barge_started is None or audio_stopped is None:
                    raise ValueError(
                        f"Incomplete barge-in timing for call {row['call_id']}"
                    )
                barge_stop.append(audio_stopped - barge_started)
            task_successes += int(bool(_bool(row, "task_success")))
            transfer_required = bool(_bool(row, "transfer_required"))
            if transfer_required:
                required_transfers += 1
                transfer_successes += int(bool(_bool(row, "transfer_success")))
            duplicate_writes += int(_required_float(row, "duplicate_writes"))
            ungrounded_answers += int(_required_float(row, "ungrounded_answers"))
            naturalness.append(_required_float(row, "naturalness_score_1_to_5"))
            costs.append(_required_float(row, "cost_usd"))

        call_count = len(arm_rows)
        p50 = _percentile(first_audio, 0.50)
        p95 = _percentile(first_audio, 0.95)
        barge_p95 = _percentile(barge_stop, 0.95)
        task_rate = task_successes / call_count if call_count else 0.0
        transfer_rate = (
            transfer_successes / required_transfers
            if required_transfers
            else 1.0
        )
        passed = {
            "minimum_calls": call_count >= gates["minimum_calls_per_arm"],
            "first_audio_p50": p50 is not None
            and p50 <= gates["first_audio_p50_ms"],
            "first_audio_p95": p95 is not None
            and p95 <= gates["first_audio_p95_ms"],
            "barge_in_p95": barge_p95 is not None
            and barge_p95 <= gates["barge_in_stop_p95_ms"],
            "task_success": task_rate >= gates["task_success_rate"],
            "transfer_success": transfer_rate >= gates["transfer_success_rate"],
            "duplicate_writes": duplicate_writes
            <= gates["maximum_duplicate_writes"],
            "grounding": ungrounded_answers
            <= gates["maximum_ungrounded_answers"],
        }
        arm_results[arm] = {
            "calls": call_count,
            "first_audio_p50_ms": p50,
            "first_audio_p95_ms": p95,
            "barge_in_stop_p95_ms": barge_p95,
            "task_success_rate": task_rate,
            "transfer_success_rate": transfer_rate,
            "duplicate_writes": duplicate_writes,
            "ungrounded_answers": ungrounded_answers,
            "naturalness_mean": statistics.fmean(naturalness),
            "total_cost_usd": sum(costs),
            "cost_per_successful_call_usd": (
                sum(costs) / task_successes if task_successes else None
            ),
            "gates": passed,
            "all_gates_pass": all(passed.values()),
        }

    preferences = list(csv.DictReader(preferences_path.open(encoding="utf-8")))
    valid_preferences = [
        row
        for row in preferences
        if row.get("winner") in expected_arms
        and row.get("retell_arm")
        in {"retell_elevenlabs_flash", "retell_platform_expressive"}
    ]
    best_retell = max(
        ("retell_elevenlabs_flash", "retell_platform_expressive"),
        key=lambda arm: (
            arm_results[arm]["all_gates_pass"],
            arm_results[arm]["naturalness_mean"],
        ),
    )
    relevant = [
        row for row in valid_preferences if row["retell_arm"] == best_retell
    ]
    challenger_wins = sum(
        row["winner"] == "direct_elevenagents_v3" for row in relevant
    )
    preference_rate = challenger_wins / len(relevant) if relevant else 0.0
    required_rate = 0.5 + gates["challenger_preference_margin_points"] / 100
    challenger_promoted = (
        arm_results["direct_elevenagents_v3"]["all_gates_pass"]
        and bool(relevant)
        and preference_rate >= required_rate
    )
    selected = "direct_elevenagents_v3" if challenger_promoted else best_retell
    return {
        "status": "complete",
        "arms": arm_results,
        "blind_preferences": {
            "comparisons_against_best_retell": len(relevant),
            "challenger_win_rate": preference_rate,
            "required_win_rate": required_rate,
        },
        "decision": {
            "selected_arm": selected,
            "challenger_promoted": challenger_promoted,
            "reason": (
                "ElevenAgents cleared all reliability gates and exceeded the blind "
                "preference margin."
                if challenger_promoted
                else "Retell remains the default because the challenger promotion rule was not met."
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    init_parser = subparsers.add_parser("init")
    init_parser.add_argument("--measurements", type=Path, required=True)
    init_parser.add_argument("--preferences", type=Path, required=True)
    init_parser.add_argument("--repeats", type=int, default=4)
    summary_parser = subparsers.add_parser("summarize")
    summary_parser.add_argument("--measurements", type=Path, required=True)
    summary_parser.add_argument("--preferences", type=Path, required=True)
    summary_parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.command == "init":
        if args.repeats < 4:
            raise SystemExit("Use at least four repeats per scenario")
        initialize_scorecards(
            args.measurements,
            args.preferences,
            repeats=args.repeats,
        )
        print(f"created {args.measurements}")
        print(f"created {args.preferences}")
        return 0

    report = summarize(args.measurements, args.preferences)
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
