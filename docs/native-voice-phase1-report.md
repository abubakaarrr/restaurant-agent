# Native voice Phase 1 completion evidence

Date: 2026-09-21

## Result and scope

The development-only OpenAI Realtime adapter now completes synthetic spoken orders and bookings through authoritative restaurant services, verified database readbacks, and a separate caller approval. Compound order choices survive a contact-information follow-up. Cancelled audio is isolated, unknown menu items receive clarification, and closed dates are stated from authoritative results.

Source freeze: `08b98ab3bfb082728fe5f162d1941eea107cae21`.
Branch: `fm/restaurant-agent-native-voice-phase-1-a1`.
Integration target: `development`.
Required preserved ancestor `26fefdce08696bd30079dbf2d5408f9dbd2fe47c` remains reachable. No reset, rebase, discarded fix commits, or replacement checkout was used. Subsequent evidence-only commits do not require another source-freeze campaign.

This is development acceptance, not a production-readiness claim. Production routing, Retell, telephony, deployment and customer data remain untouched.

## What changed

| Finding | Implemented boundary |
| --- | --- |
| NV-019 | Complete canonical order effects and notes are read back; confirmation is bound to the released pending payload's database draft version. Unchanged state synchronization no longer invalidates newer evidence. Missing or wrong stored effects are rejected. |
| NV-020 | Finalized turn IDs, versioned state and operation identities prevent provisional/replayed tool effects; committed results survive interruption/restart. |
| NV-021 | Missing or incomplete assistant transcripts suppress consequential audio without fabricating a transcript or success. |
| NV-022 | Response generations isolate cancellation; WebSocket cancellation uses supported events and clears local buffered audio. |
| NV-023 | Canonical menu options, required groups and applied effects reach the model. Removal options are checked with the same customization resolver used by the write service. |
| NV-024 | Negative availability and restaurant closure survive the bridge. Negative slot speech is server-authored; unknown menu results request clarification. |
| NV-025 | Model-supplied session IDs and booking details do not establish authorization. Booking access requires existing server-bound session ownership. The identity-bootstrap shortcut was removed after a real-database cross-session test exposed it. |

Live integration also corrected the required output PCM rate, an insufficient response-token cap, session-ID generation by the model, raw dictionary readbacks, mutation of readback payloads during fact projection, ambiguous persisted booking notes, and the existing booking reschedule regression. Canonical speech generation uses explicit text context and remains checked before audio release.

## Validation

Counts overlap; they are not a count of distinct tests.

| Suite | Result | Source evidence |
| --- | --- | --- |
| Native tests with disposable PostgreSQL | 90 passed | Exact patch committed as `08b98ab3`; includes six complete public booking lifecycle variants and matching-identity cross-session denial |
| Existing database/service/security tests | 28 passed | `6e421931`; later changes only affect native closure speech and denial of unverified booking access |
| Full offline suite, network disabled | 352 passed, 54 skipped | `08b98ab3` |
| Independent review | Passed scoped final corrections | User-approved replacement for timed-out NoMistakes review |
| Whitespace and tracked-secret checks | Passed | No approved API key or long secret-token pattern in tracked candidate files |

Offline skips include explicitly enabled database/infrastructure tests; the enabled database suites are reported separately. The existing Starlette/AnyIO deprecation warning remains nonblocking.

Commands used through the isolated validation runner:

```text
python -m pytest tests/test_native_voice_phase1.py tests/test_native_voice_public_flow.py -q --tb=short
python -m pytest tests/test_database_integration.py tests/test_app_security_integration.py tests/test_connection.py -q --tb=short
python -m pytest tests -q --tb=short
```

Native runs use RUN_DB_INTEGRATION=1, APP_ENV=development, a marked disposable NATIVE_VOICE_DATABASE_URL and explicit native write flags. The ordinary DATABASE_URL is disabled. Legacy database regressions use a separate disposable database. Offline tests run with RUN_DB_INTEGRATION=0 and networking disabled.

NoMistakes run `01M31XQW8A7SMVBQXSXF6ZVWBS` timed out without a verdict. It is **not reported as passing**. The user authorized independent review followed by live acceptance and PR creation. Custody was recovered with the prescribed recovery operation; no additional NoMistakes pipeline was started during these corrections.

## Live acceptance

All audio and customer records were synthetic. Model: `gpt-realtime`; 24 kHz mono PCM. The restaurant clock was explicitly fixed to September 23, 2026 at 18:00 in the restaurant timezone, allowing deterministic service-hour fixtures. Only the approved OpenAI key was selected from the existing secret file; production database settings were not loaded.

[Machine-readable evidence](../evaluation/native-voice-phase1/acceptance-summary.json) contains source heads, report hashes, utterances, field checks, tool outcomes and timings. Later source changes were restricted to the affected paths, so only affected scenarios were rerun.

| Scenario | Result |
| --- | --- |
| Simple pickup order | Correct item/side/quantity/fulfillment, exact audible readback, later approval, verified database confirmation |
| 26.349-second compound order | All nine expected checks passed after a contact follow-up: quantity corrected from two to one; fries; onion-jam removal; cut-in-half note; no-utensils note; sesame allergy; pickup; one line; no unresolved fields. Exact readback and later confirmation passed. No allergen-safety guarantee was spoken. |
| Unknown item | Audible clarification; no invented availability, price or item |
| Closed restaurant | Exact audible closure for September 28, 2026; no booking mutation |
| Interruption | Generation advanced, cancelled response released no audio, following turn responded |
| Booking | Audible full proposal, separate approval, PostgreSQL confirmed two guests for September 25 at 19:00 |
| Cross-session booking attack | A fresh session supplying the exact existing ID/name/phone was denied with no booking data returned |
| Missing/late transcripts and adversarial event ordering | Deterministic protocol tests; intentionally induced provider transcript omission is not claimed |

Grounding verdicts were reviewed against the actual authoritative tool outputs. The original grounding harness deliberately leaves `acceptance=false` pending manual review; the summary records the reviewed verdict and its exact criteria rather than hiding that convention.

## Limits and follow-up

- Full model audio is buffered until speech verification. Accepted full order readbacks took approximately **7.7–9.4 seconds** after caller audio commit; confirmation replies took approximately **2.1–2.2 seconds**. This does not establish the desired low-latency human experience.
- This bounded synthetic campaign is not a human, accent, noisy-room, concurrency or statistical reliability benchmark.
- One simple-order attempt at `6e421931` ended with a RuntimeError before any write. Its earlier harness omitted the terminal cause. An instrumented rerun passed; the original cause remains unclassified. The failed report and empty-operation state were preserved.
- Existing-booking access from a new session requires trusted ownership provisioning. Merely stating a name, phone or booking ID is intentionally insufficient.
- No production audio transport or user-facing voice UI was added in this phase, and no production readiness or deployment is claimed.

All prior live attempts, local audio and validation logs remain under the local acceptance workspace. The committed summary contains selected accepted evidence and identifies the preserved attempts. No merge or deployment is authorized by this report.
