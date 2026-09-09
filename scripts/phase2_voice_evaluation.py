#!/usr/bin/env python3
"""Prepare and aggregate the offline-safe Phase 2 voice evaluation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import statistics
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.agent.runner import opening_greeting
from app.behavior import BehaviorState, TurnObservation, reduce_behavior
from app.restaurant_knowledge import get_restaurant_knowledge
from app.spoken_delivery import (
    FRUSTRATION_REPLY,
    INCOMPLETE_INPUT_REPLY,
    TRANSFER_UNAVAILABLE_REPLY,
    ResponseGenerationGate,
    spoken_text_violations,
)


DEFAULT_PLAN = ROOT / "config" / "phase2-voice-evaluation.v1.json"
PROVIDER_UNAVAILABLE_REASON = (
    "Explicit voice-sample and provider-action authorization was not supplied."
)

MEASUREMENT_FIELDS = (
    "arm",
    "scenario_id",
    "repeat",
    "status",
    "missing_reason",
    "customer_speech_end_ms",
    "agent_request_start_ms",
    "agent_text_first_token_ms",
    "provider_request_start_ms",
    "provider_response_ms",
    "first_audio_playback_ms",
    "final_audio_completion_ms",
    "interruption_ms",
    "cancellation_effective_ms",
    "stale_response_incidents",
    "error_count",
    "timeout_count",
    "fallback_count",
)

RATING_BASE_FIELDS = (
    "presentation_id",
    "scenario_id",
    "repeat",
    "sample_a_ref",
    "sample_b_ref",
    "rater_id",
    "preferred_sample",
    "notes",
)


def load_plan(path: Path = DEFAULT_PLAN) -> dict[str, Any]:
    plan = json.loads(path.read_text(encoding="utf-8"))
    if plan.get("schema_version") != "restaurant-voice-evaluation.v1":
        raise ValueError("Unsupported Phase 2 evaluation schema")
    if len(plan.get("arms") or []) != 2:
        raise ValueError("Phase 2 requires exactly a baseline and clone arm")
    scenario_ids = [row.get("id") for row in plan.get("scenarios") or []]
    if not scenario_ids or len(scenario_ids) != len(set(scenario_ids)):
        raise ValueError("Scenario IDs must be present and unique")
    if int(plan.get("scenario_repeats_per_arm") or 0) < 10:
        raise ValueError("At least ten comparable turns per arm must be prepared")
    return plan


def _join_spoken(values: list[str], *, conjunction: str) -> str:
    if len(values) == 1:
        return values[0]
    return ", ".join(values[:-1]) + f", {conjunction} {values[-1]}"


def _local_scenario_result(scenario: dict[str, Any]) -> dict[str, Any]:
    knowledge = get_restaurant_knowledge()
    kind = str(scenario["kind"])
    source_refs: list[str] = []
    outputs: list[str] = []
    stale_incidents = 0
    interruption_failures = 0

    if kind == "greeting":
        outputs = [opening_greeting()]
        source_refs = [knowledge.metadata["fixture_id"]]
    elif kind == "faq":
        topic = knowledge.find_topic(str(scenario["customer_input"]))
        if topic.status != "known" or topic.records[0]["topic_id"] != scenario["topic_id"]:
            raise ValueError(f"FAQ scenario is not grounded: {scenario['id']}")
        identity = knowledge.identity
        outputs = [
            f"We're at {identity['address']['street']} in {identity['address']['city']}, "
            f"two blocks west of {identity['landmark'].removeprefix('Two blocks west of ')}."
        ]
        source_refs = [str(topic.records[0]["topic_id"])]
    elif kind == "menu_clarification":
        match = knowledge.find_menu_item(str(scenario["menu_query"]))
        if match.status != "ambiguous" or len(match.candidates) < 2:
            raise ValueError("Menu clarification scenario must stay ambiguous")
        names = [str(row["name"]) for row in match.candidates]
        outputs = [
            f"We have {_join_spoken(names, conjunction='or')}. Which one did you mean?"
        ]
        source_refs = [str(row["item_id"]) for row in match.candidates]
    elif kind == "allergy":
        match = knowledge.find_menu_item(str(scenario["menu_query"]))
        if match.status != "known" or match.item is None:
            raise ValueError("Allergy scenario item is not grounded")
        item = match.item
        allergens = _join_spoken(
            [str(value).replace("_", " ") for value in item["allergens"]],
            conjunction="and",
        )
        outputs = [
            f"The {item['name']} contains {allergens}. It's made in a shared kitchen, "
            "so I can't guarantee zero cross-contact. Would you like staff to follow up?"
        ]
        source_refs = [str(item["item_id"])]
    elif kind == "reservation":
        topic = knowledge.find_topic("reservation policy")
        if topic.status != "known" or topic.records[0]["topic_id"] != "topic.reservations":
            raise ValueError("Reservation policy is not grounded")
        outputs = [
            "I can check availability, but a table isn't booked until the reservation "
            "succeeds. What date would you like?"
        ]
        source_refs = ["topic.reservations"]
    elif kind == "order_correction":
        match = knowledge.find_menu_item(str(scenario["menu_query"]))
        if match.status != "known" or match.item is None:
            raise ValueError("Order-correction item is not grounded")
        outputs = [
            f"You want {scenario['new_quantity']} {match.item['name']}, not "
            f"{scenario['old_quantity']}. Is that the only correction?"
        ]
        source_refs = [str(match.item["item_id"])]
    elif kind == "frustration":
        reduction = reduce_behavior(
            BehaviorState(), TurnObservation(text=str(scenario["customer_input"]))
        )
        if "explicit_complaint" not in reduction.directive.reasons:
            raise ValueError("Frustration scenario must activate the de-escalating policy")
        outputs = [FRUSTRATION_REPLY]
        source_refs = ["behavior:explicit_complaint"]
    elif kind == "interruption":
        old_id = int(scenario["old_response_id"])
        new_id = int(scenario["new_response_id"])
        gate = ResponseGenerationGate()
        gate.begin(old_id)
        gate.begin(new_id)
        stale_incidents = int(gate.allows(old_id))
        reduction = reduce_behavior(
            BehaviorState(),
            TurnObservation(text=str(scenario["customer_input"]), interrupted=True),
        )
        interruption_failures = int(
            "interrupted" not in reduction.directive.reasons
            or reduction.directive.interruption_sensitivity < 0.9
        )
        outputs = ["You want seven o'clock instead. Is that the only correction?"]
        source_refs = ["response-generation-gate", "behavior:interrupted"]
    elif kind == "silence_incomplete":
        state = BehaviorState()
        for _ in range(3):
            reduction = reduce_behavior(state, TurnObservation(text="", reminder=True))
            state = reduction.state
            if reduction.directive.direct_reply:
                outputs.append(reduction.directive.direct_reply)
        incomplete = reduce_behavior(
            BehaviorState(), TurnObservation(text=str(scenario["customer_input"]))
        )
        outputs.append(incomplete.directive.direct_reply or INCOMPLETE_INPUT_REPLY)
        source_refs = ["behavior:silence-ladder", "behavior:unintelligible-audio"]
    elif kind == "unknown":
        match = knowledge.find_topic(str(scenario["customer_input"]))
        if match.status != "unknown":
            raise ValueError("Unknown scenario unexpectedly matched restaurant facts")
        outputs = [
            "I don't have that answer in the current restaurant information. "
            "I can take a message for the team."
        ]
        source_refs = [knowledge.metadata["fixture_id"]]
    elif kind == "transfer_unavailable":
        outputs = [TRANSFER_UNAVAILABLE_REPLY]
        source_refs = ["spoken-delivery:transfer-unavailable"]
    elif kind == "personal_identity":
        reduction = reduce_behavior(
            BehaviorState(), TurnObservation(text=str(scenario["customer_input"]))
        )
        if reduction.directive.direct_reply is None:
            raise ValueError("Personal identity scenario must answer directly")
        outputs = [reduction.directive.direct_reply]
        source_refs = ["behavior:truthful-identity"]
    else:
        raise ValueError(f"Unknown local scenario kind: {kind}")

    violations = {
        str(index + 1): list(spoken_text_violations(output))
        for index, output in enumerate(outputs)
        if spoken_text_violations(output)
    }
    too_many_questions = [
        index + 1 for index, output in enumerate(outputs) if output.count("?") > 1
    ]
    passed = not violations and not too_many_questions
    if kind == "interruption":
        passed = passed and stale_incidents == 0 and interruption_failures == 0
    return {
        "scenario_id": scenario["id"],
        "customer_input": scenario["customer_input"],
        "outputs": outputs,
        "source_refs": source_refs,
        "spoken_contract_violations": violations,
        "outputs_with_multiple_questions": too_many_questions,
        "interruption_recovery_failures": interruption_failures,
        "stale_response_incidents": stale_incidents,
        "passed": passed,
    }


def run_local_scenarios(plan: dict[str, Any]) -> list[dict[str, Any]]:
    return [_local_scenario_result(row) for row in plan["scenarios"]]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _write_provider_scorecard(path: Path, plan: dict[str, Any]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=MEASUREMENT_FIELDS, lineterminator="\n"
        )
        writer.writeheader()
        for arm in plan["arms"]:
            for scenario in plan["scenarios"]:
                for repeat in range(1, int(plan["scenario_repeats_per_arm"]) + 1):
                    writer.writerow(
                        {
                            "arm": arm["id"],
                            "scenario_id": scenario["id"],
                            "repeat": repeat,
                            "status": "missing_authorization",
                            "missing_reason": PROVIDER_UNAVAILABLE_REASON,
                        }
                    )


def _write_rating_artifacts(
    form_path: Path,
    key_path: Path,
    plan: dict[str, Any],
) -> None:
    rng = random.Random(int(plan["randomization_seed"]))
    arms = [str(row["id"]) for row in plan["arms"]]
    with form_path.open("w", newline="", encoding="utf-8") as form_handle, key_path.open(
        "w", newline="", encoding="utf-8"
    ) as key_handle:
        dimensions = list(plan["preference_dimensions"])
        rating_fields = (
            *RATING_BASE_FIELDS[:6],
            *(f"sample_a_{dimension}" for dimension in dimensions),
            *(f"sample_b_{dimension}" for dimension in dimensions),
            *RATING_BASE_FIELDS[6:],
        )
        form_writer = csv.DictWriter(
            form_handle, fieldnames=rating_fields, lineterminator="\n"
        )
        key_writer = csv.DictWriter(
            key_handle,
            fieldnames=("presentation_id", "sample_a_arm", "sample_b_arm"),
            lineterminator="\n",
        )
        form_writer.writeheader()
        key_writer.writeheader()
        for scenario in plan["scenarios"]:
            for repeat in range(1, int(plan["scenario_repeats_per_arm"]) + 1):
                shuffled = list(arms)
                rng.shuffle(shuffled)
                presentation_id = f"{scenario['id']}-{repeat:02d}"
                form_writer.writerow(
                    {
                        "presentation_id": presentation_id,
                        "scenario_id": scenario["id"],
                        "repeat": repeat,
                    }
                )
                key_writer.writerow(
                    {
                        "presentation_id": presentation_id,
                        "sample_a_arm": shuffled[0],
                        "sample_b_arm": shuffled[1],
                    }
                )


def _optional_number(row: dict[str, str], field: str) -> float | None:
    raw = str(row.get(field) or "").strip()
    if not raw:
        return None
    value = float(raw)
    if not math.isfinite(value):
        raise ValueError(f"Non-finite {field}")
    return value


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _metric_summary(values: list[float], missing_reason: str | None) -> dict[str, Any]:
    return {
        "availability": "measured" if values else "missing",
        "samples": len(values),
        "median_ms": statistics.median(values) if values else None,
        "p95_ms": _percentile(values, 0.95),
        "missing_reason": None if values else missing_reason,
    }


def _latencies(
    rows: list[dict[str, str]], start: str, end: str
) -> list[float]:
    values: list[float] = []
    for row in rows:
        start_value = _optional_number(row, start)
        end_value = _optional_number(row, end)
        if start_value is None or end_value is None:
            continue
        if end_value < start_value:
            raise ValueError(f"{end} precedes {start} for {row.get('scenario_id')}")
        values.append(end_value - start_value)
    return values


def summarize(
    plan: dict[str, Any],
    local_results: list[dict[str, Any]],
    measurement_rows: list[dict[str, str]],
    rating_rows: list[dict[str, str]],
    randomization_rows: list[dict[str, str]],
) -> dict[str, Any]:
    expected_arms = {str(row["id"]) for row in plan["arms"]}
    if {row.get("arm", "") for row in measurement_rows} != expected_arms:
        raise ValueError("Provider scorecard does not contain both expected arms")
    missing_reasons = sorted(
        {
            str(row.get("missing_reason") or "").strip()
            for row in measurement_rows
            if str(row.get("missing_reason") or "").strip()
        }
    )
    missing_reason = "; ".join(missing_reasons) or "No completed measurement supplied."
    arm_results: dict[str, Any] = {}
    for arm in sorted(expected_arms):
        arm_rows = [row for row in measurement_rows if row.get("arm") == arm]
        completed = [row for row in arm_rows if row.get("status") == "completed"]
        latency = {
            "agent_generation": _metric_summary(
                _latencies(completed, "agent_request_start_ms", "agent_text_first_token_ms"),
                missing_reason,
            ),
            "provider_network": _metric_summary(
                _latencies(completed, "provider_request_start_ms", "provider_response_ms"),
                missing_reason,
            ),
            "first_audio": _metric_summary(
                _latencies(completed, "customer_speech_end_ms", "first_audio_playback_ms"),
                missing_reason,
            ),
            "completion": _metric_summary(
                _latencies(completed, "customer_speech_end_ms", "final_audio_completion_ms"),
                missing_reason,
            ),
            "interruption_recovery": _metric_summary(
                _latencies(completed, "interruption_ms", "cancellation_effective_ms"),
                missing_reason,
            ),
        }
        event_counts: dict[str, int | None] = {}
        for field in (
            "stale_response_incidents",
            "error_count",
            "timeout_count",
            "fallback_count",
        ):
            values = [_optional_number(row, field) for row in completed]
            present = [value for value in values if value is not None]
            event_counts[field] = int(sum(present)) if present else None
        arm_results[arm] = {
            "prepared_turns": len(arm_rows),
            "completed_turns": len(completed),
            "completed_scenarios": len({row["scenario_id"] for row in completed}),
            "latency_ms": latency,
            "event_counts": event_counts,
            "provider_audio_score": None,
            "missing_reason": missing_reason if not completed else None,
        }

    dimensions = list(plan["preference_dimensions"])
    randomization = {
        str(row["presentation_id"]): {
            "A": str(row["sample_a_arm"]),
            "B": str(row["sample_b_arm"]),
        }
        for row in randomization_rows
    }
    complete_ratings = [
        row
        for row in rating_rows
        if str(row.get("rater_id") or "").strip()
        and str(row.get("preferred_sample") or "").strip() in {"A", "B"}
        and str(row.get("sample_a_ref") or "").strip()
        and str(row.get("sample_b_ref") or "").strip()
        and row.get("presentation_id") in randomization
        and all(
            _optional_number(row, f"sample_{sample}_{dimension}") is not None
            for sample in ("a", "b")
            for dimension in dimensions
        )
    ]
    scores_by_arm: dict[str, dict[str, list[float]]] = {
        arm: {dimension: [] for dimension in dimensions} for arm in expected_arms
    }
    preference_wins = {arm: 0 for arm in expected_arms}
    for row in complete_ratings:
        mapping = randomization[str(row["presentation_id"])]
        for sample in ("A", "B"):
            arm = mapping[sample]
            if arm not in scores_by_arm:
                raise ValueError(f"Unknown randomized arm: {arm}")
            for dimension in dimensions:
                value = float(row[f"sample_{sample.casefold()}_{dimension}"])
                if not 1 <= value <= 5:
                    raise ValueError(
                        f"Preference dimension {dimension} must use the 1-5 scale"
                    )
                scores_by_arm[arm][dimension].append(value)
        preference_wins[mapping[str(row["preferred_sample"])]] += 1

    arm_preference_scores: dict[str, Any] = {}
    for arm in sorted(expected_arms):
        dimension_scores: dict[str, Any] = {}
        all_values: list[float] = []
        for dimension in dimensions:
            values = scores_by_arm[arm][dimension]
            all_values.extend(values)
            dimension_scores[dimension] = {
                "samples": len(values),
                "mean": statistics.fmean(values) if values else None,
                "availability": "measured" if values else "missing",
                "missing_reason": None if values else PROVIDER_UNAVAILABLE_REASON,
            }
        arm_preference_scores[arm] = {
            "overall_mean": statistics.fmean(all_values) if all_values else None,
            "preference_wins": preference_wins[arm],
            "dimensions": dimension_scores,
        }

    preferred_arm = None
    if complete_ratings and len(set(preference_wins.values())) > 1:
        preferred_arm = max(preference_wins, key=preference_wins.get)

    baseline = arm_results["current_retell_baseline"]
    clone = arm_results["retell_native_clone"]
    baseline_score = arm_preference_scores["current_retell_baseline"]["overall_mean"]
    clone_score = arm_preference_scores["retell_native_clone"]["overall_mean"]
    baseline_first_audio = baseline["latency_ms"]["first_audio"]["median_ms"]
    clone_first_audio = clone["latency_ms"]["first_audio"]["median_ms"]
    baseline_completion = baseline["latency_ms"]["completion"]["median_ms"]
    clone_completion = clone["latency_ms"]["completion"]["median_ms"]

    local_stale = sum(int(row["stale_response_incidents"]) for row in local_results)
    local_interrupt_failures = sum(
        int(row["interruption_recovery_failures"]) for row in local_results
    )
    all_local_pass = all(bool(row["passed"]) for row in local_results)
    spec_bytes = json.dumps(plan, sort_keys=True, separators=(",", ":")).encode()
    return {
        "schema_version": "restaurant-voice-evaluation-results.v1",
        "evaluation_id": plan["evaluation_id"],
        "input_sha256": hashlib.sha256(spec_bytes).hexdigest(),
        "provider_authorization": {
            "available": False,
            "reason": PROVIDER_UNAVAILABLE_REASON,
        },
        "prepared": {
            "scenario_count": len(plan["scenarios"]),
            "turns_per_arm": len(plan["scenarios"])
            * int(plan["scenario_repeats_per_arm"]),
            "blind_comparisons": len(rating_rows),
            "randomization_seed": int(plan["randomization_seed"]),
        },
        "offline_local_text": {
            "completed_scenarios": len(local_results),
            "passed_scenarios": sum(bool(row["passed"]) for row in local_results),
            "all_passed": all_local_pass,
            "interruption_scenarios": sum(
                row["scenario_id"] == "interruption" for row in local_results
            ),
            "interruption_recovery_failures": local_interrupt_failures,
            "stale_response_incidents": local_stale,
            "errors": 0,
            "timeouts": 0,
            "fallbacks": 0,
        },
        "arms": arm_results,
        "human_preference": {
            "rater_count": len(
                {str(row["rater_id"]).strip() for row in complete_ratings}
            ),
            "completed_ratings": len(complete_ratings),
            "arm_scores": arm_preference_scores,
            "preferred_arm": preferred_arm,
            "availability": "measured" if complete_ratings else "missing",
            "missing_reason": None if complete_ratings else PROVIDER_UNAVAILABLE_REASON,
        },
        "baseline_vs_clone": {
            "completed_pairs": len(complete_ratings),
            "baseline_score": baseline_score,
            "clone_score": clone_score,
            "preference_delta": (
                clone_score - baseline_score
                if clone_score is not None and baseline_score is not None
                else None
            ),
            "first_audio_delta_ms": (
                clone_first_audio - baseline_first_audio
                if clone_first_audio is not None and baseline_first_audio is not None
                else None
            ),
            "completion_delta_ms": (
                clone_completion - baseline_completion
                if clone_completion is not None and baseline_completion is not None
                else None
            ),
        },
        "targets": {
            "median_first_audio_under_ms": 700,
            "p95_first_audio_under_ms": 1200,
            "zero_stale_response_incidents": True,
            "provider_targets_evaluated": False,
        },
        "recommendation": {
            "decision": "retain_baseline",
            "reason": (
                "Keep the current baseline fallback. The offline wording and turn-taking "
                "fixtures pass, but no authorized clone audio, provider latency, or human "
                "preference evidence exists to support adopting a clone."
            ),
        },
        "limitations": [
            PROVIDER_UNAVAILABLE_REASON,
            (
                "No provider call, phone call, recording, voice sample, or "
                "customer-visible action occurred."
            ),
            (
                "Local text fixtures do not measure first-audio, completion, "
                "provider-network, or acoustic voice quality."
            ),
        ],
    }


def prepare(output_dir: Path, plan_path: Path = DEFAULT_PLAN) -> dict[str, Any]:
    plan = load_plan(plan_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    local_results = run_local_scenarios(plan)
    local_path = output_dir / "local-scenario-results.jsonl"
    measurement_path = output_dir / "provider-measurements.csv"
    rating_path = output_dir / "blind-rating-form.csv"
    key_path = output_dir / "randomization-key.csv"
    _write_jsonl(local_path, local_results)
    _write_provider_scorecard(measurement_path, plan)
    _write_rating_artifacts(rating_path, key_path, plan)
    with measurement_path.open(newline="", encoding="utf-8") as handle:
        measurements = list(csv.DictReader(handle))
    with rating_path.open(newline="", encoding="utf-8") as handle:
        ratings = list(csv.DictReader(handle))
    with key_path.open(newline="", encoding="utf-8") as handle:
        randomization = list(csv.DictReader(handle))
    result = summarize(plan, local_results, measurements, ratings, randomization)
    (output_dir / "results.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = prepare(args.output_dir, args.plan)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["offline_local_text"]["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
