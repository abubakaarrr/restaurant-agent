# CI/CD and deployment checks

This repository includes a minimal CI workflow in `.github/workflows/ci.yml` for
foundational build, test, and deployment-readiness checks.

## CI workflow summary

- Trigger: pull requests and pushes to `main`.
- Jobs:
  - `checks` — compile + local checks and non-DB suite.
  - `integration` — isolated PostgreSQL run (`pgvector/pgvector:pg16`) + database
    integration tests.

## Required commands and assumptions

- `python -m pip install -r requirements-dev.txt` must succeed.
- `python -m pytest` requires tests and migration scripts to be runnable in this
  environment.
- `docker build` must be available in GitHub-hosted runners.

The workflow intentionally does not publish images and does not include provider
mutations.

## Required secrets / environments / approvals

- Baseline CI does not require repository secrets.
- Staging release execution is **manual and off-pipeline**. `scripts/staging_release.py`
  enforces an explicit `--release` gate before it emits a command plan.
- Do not execute staging release commands without explicit Captain approval.

## Artifact evidence captured

- `ci-image-evidence/<sha>/build-meta.txt`
- `ci-image-evidence/<sha>/image.json`

These artifacts include the generated image reference and the Docker image metadata.

## Docker image reference strategy

CI builds the image with an immutable SHA tag:
`ghcr.io/<owner>/<repo>:<commit_sha>`.

## Health and smoke checks

- `/health` must respond successfully after rollout.
- Release smoke checks in `scripts/staging_release.py` include health checks and
  container status checks.

## Rollback support

- The staging plan captures the previous web image into
  `releases/<sha>/previous_image.txt`.
- Rollback command example:
  `RESTAURANT_IMAGE_TAG=$(cat releases/<sha>/previous_image.txt) docker compose up -d --no-build db web`.

