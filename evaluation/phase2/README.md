# Phase 2 offline voice evaluation

This directory is the reproducible, provider-safe preparation for the Phase 2
baseline-versus-Retell-native-clone comparison. The checked-in run used no
credentials, voice sample, audio, recording, provider API, phone call, or live
system.

Regenerate every artifact from the versioned scenario plan:

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/phase2_voice_evaluation.py \
  --output-dir evaluation/phase2
```

The command runs the twelve deterministic local text scenarios, creates ten
measurement rows per scenario for each provider arm, prepares 120 blinded A/B
rating presentations with a fixed randomization seed, and aggregates only rows
whose status is `completed`. Blank provider values remain missing; they are
never interpreted as zero.

`provider-measurements.csv` preserves these definitions:

- Agent generation latency is first text token minus agent request start.
- Provider/network latency is provider response minus provider request start.
- First-audio latency is first audible playback minus customer speech end.
- Completion latency is final audible completion minus customer speech end.
- Interruption recovery is cancellation effective minus interruption start.
- A stale-response incident is old text or audio delivered after a newer
  customer turn begins.
- Errors are generation, provider, transport, or playback failures. Timeouts
  are stages exceeding their configured deadline. Fallbacks count only an
  actual switch to the current baseline.

For a later authorized bakeoff, an operator records non-secret sample
references in `blind-rating-form.csv` and keeps `randomization-key.csv` hidden
from raters until scoring is complete. Each rater scores both samples from 1 to
5 on naturalness, warmth, clarity, pronunciation, expressiveness, confidence,
conversational pacing, and whether it feels like a real human. Provider actions
remain prohibited until separate sample and account-action authorization is
documented.

The current `results.json` intentionally reports zero completed provider
scenarios and zero raters. Provider audio scores, preference scores, and all
provider latency values are explicit `null` values with a missing reason. The
offline result recommends retaining the baseline fallback because there is no
authorized clone evidence to justify adoption.

## Run evidence

The combined focused and affected-suite validation command for this run was:

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/phase2_voice_evaluation.py \
  --output-dir evaluation/phase2 >/dev/null && \
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q \
  tests/test_phase2_voice_humanization.py \
  tests/test_retell_protocol.py \
  tests/test_behavior.py \
  tests/test_stale_reply_extraction.py
```

It regenerated all five evaluation artifacts successfully. The focused Phase 2
tests and the existing Retell protocol, behavior, and stale-reply suites all
passed: 67 tests in 17.55 seconds. The run prepared 12 local scenarios, 120
turns per provider arm, and 120 blinded comparisons. All 12 local scenarios
passed with zero stale-response incidents. Provider scenario completions,
raters, and ratings remained zero; clone-audio references, audio and preference
results, provider latency, errors, timeouts, and fallbacks remained explicitly
missing or `null`.

Changed files relative to the Phase 2 starting commit:

- `README.md`
- `app/behavior.py`
- `app/call_flags.py`
- `app/prompts/retell/global.md`
- `app/prompts/retell/handoff.md`
- `app/prompts/system.md`
- `app/retell_handler.py`
- `app/services/restaurant.py`
- `app/spoken_delivery.py`
- `app/transfer_availability.py`
- `config/phase2-voice-evaluation.v1.json`
- `evaluation/phase2/README.md`
- `evaluation/phase2/blind-rating-form.csv`
- `evaluation/phase2/local-scenario-results.jsonl`
- `evaluation/phase2/provider-measurements.csv`
- `evaluation/phase2/randomization-key.csv`
- `evaluation/phase2/results.json`
- `scripts/phase2_handler_delivery_trace.py`
- `scripts/phase2_voice_evaluation.py`
- `tests/test_phase2_voice_humanization.py`
- `tests/test_retell_protocol.py`

This run did not access staging or production, enable live writes, call a
provider or phone, use credentials or recordings, or upload, clone, synthesize,
select, assign, mutate, or delete any voice. The limitation is unchanged: local
text and handler traces provide no authorized acoustic or provider-performance
evidence, so the recommendation remains to retain the baseline.
