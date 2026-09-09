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
