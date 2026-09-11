#!/usr/bin/env python3
"""Generate a safe, explicit staging release plan.

This script is intentionally read-only by default. Staging writes happen only
when `--release` is used with an explicit review of the emitted commands.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import shlex
from pathlib import Path

STAGING_ALIAS = "staging"
STAGING_HOSTNAME = "agent.servicesground.com"
SSH_PORT = 717
DEFAULT_REMOTE_DIR = "/opt/restaurant-agent"
DEFAULT_IMAGE_REPO = "ghcr.io/abubakaarrr/restaurant-agent"

REQUIRED_NON_EMPTY_KEYS = [
    "POSTGRES_PASSWORD",
    "DATABASE_URL",
    "VOICE_TOOL_SECRET",
    "DASHBOARD_API_KEY",
    "SESSION_SECRET",
    "RETELL_API_KEY",
    "RETELL_AGENT_ID",
    "RETELL_PUBLIC_KEY",
    "LOGIN_PASSWORD",
    "ALLOWED_ORIGINS",
    "APP_ENV",
]

REQUIRED_FALSE_KEYS = ["VOICE_LIVE_WRITES_ENABLED"]


class ReleaseSafetyError(RuntimeError):
    """Raised when staging commands would violate safe-operating assumptions."""


def _parse_env_file(path: Path) -> dict[str, str]:
    data: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        data[key.strip()] = value.strip()
    return data


def _is_placeholder(value: str) -> bool:
    lowered = value.lower()
    return "replace_with" in lowered or "replace-with" in lowered or "placeholder" in lowered


def _is_false(value: str) -> bool:
    return value.strip().lower() in {"0", "false", "off", "no"}


def _assert_release_required(release: bool) -> None:
    if not release:
        raise ReleaseSafetyError(
            "Refusing to generate staging commands without --release. "
            "Use --release to run a gated plan."
        )


def _assert_staging_env(values: dict[str, str]) -> None:
    if values.get("APP_ENV", "").strip().lower() != "production":
        raise ReleaseSafetyError("APP_ENV must be set to production for staging plan.")

    for key in REQUIRED_NON_EMPTY_KEYS:
        if key not in values or not values[key].strip():
            raise ReleaseSafetyError(f"Missing required env key: {key}")
        if _is_placeholder(values[key]):
            raise ReleaseSafetyError(f"Key {key} contains placeholder text")

    if "*" in values.get("ALLOWED_ORIGINS", ""):
        raise ReleaseSafetyError("ALLOWED_ORIGINS must not include '*'")

    for key in REQUIRED_FALSE_KEYS:
        if key not in values or not _is_false(values[key]):
            raise ReleaseSafetyError(
                f"{key} must be explicitly false during staging dry-run/release prep"
            )


@dataclass(frozen=True)
class ReleasePlan:
    sha: str
    image_ref: str
    remote_alias: str
    remote_dir: str
    commands: tuple[str, ...]


def build_staging_plan(
    sha: str,
    release: bool,
    *,
    env_file: Path = Path(".env.production.example"),
    remote_alias: str = STAGING_ALIAS,
    remote_dir: str = DEFAULT_REMOTE_DIR,
    image_repo: str = DEFAULT_IMAGE_REPO,
    bootstrap_db: bool = False,
) -> ReleasePlan:
    _assert_release_required(release)
    values = _parse_env_file(env_file)
    _assert_staging_env(values)

    image_ref = f"{image_repo}:{sha}"
    migration_cmd = (
        "python scripts/migrate.py --initialize-schema"
        if bootstrap_db
        else "python scripts/migrate.py"
    )

    ssh_prefix = f"ssh -p {SSH_PORT} {remote_alias}"

    def cmd(command: str) -> str:
        return f"{ssh_prefix} bash -lc {shlex.quote(command)}"

    commands: list[str] = [
        cmd(
            "set -euo pipefail && "
            f"mkdir -p {remote_dir}/releases/{sha} && "
            f"printf '%s\\n' '{sha}' > {remote_dir}/releases/{sha}/requested_sha.txt"
        ),
        cmd(
            f"printf '%s\\n' '{image_ref}' > "
            f"{remote_dir}/releases/{sha}/release_image.txt"
        ),
        cmd(
            f"cd {remote_dir} && docker inspect $(docker compose ps -q web) "
            f"--format '{{{{.Config.Image}}}}' "
            f"> {remote_dir}/releases/{sha}/previous_image.txt || true"
        ),
        cmd(f"cd {remote_dir} && docker pull {image_ref}"),
        cmd(
            f"cd {remote_dir} && "
            f"RESTAURANT_IMAGE_TAG={image_ref} docker compose up -d --no-build db web"
        ),
        cmd(
            f"cd {remote_dir} && "
            f"RESTAURANT_IMAGE_TAG={image_ref} docker compose run --rm web {migration_cmd}"
        ),
        cmd(
            f"cd {remote_dir} && "
            f"RESTAURANT_IMAGE_TAG={image_ref} docker compose run --rm web python db/seed.py"
        ),
        cmd(
            f"cd {remote_dir} && curl -fsS https://{STAGING_HOSTNAME}/health "
            f"> {remote_dir}/releases/{sha}/smoke-health.json"
        ),
        cmd(f"cd {remote_dir} && docker compose ps web"),
    ]

    return ReleasePlan(
        sha=sha,
        image_ref=image_ref,
        remote_alias=remote_alias,
        remote_dir=remote_dir,
        commands=tuple(commands),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release", action="store_true", help="Enable staged release plan generation")
    parser.add_argument("--sha", required=True, help="Deploy commit SHA")
    parser.add_argument("--env-file", default=".env.production.example", type=Path)
    parser.add_argument(
        "--remote-alias",
        default=STAGING_ALIAS,
        help="SSH alias for the staging host",
    )
    parser.add_argument("--remote-dir", default=DEFAULT_REMOTE_DIR)
    parser.add_argument(
        "--image-repo",
        default=DEFAULT_IMAGE_REPO,
        help="Image repository used for immutable SHA tagging",
    )
    parser.add_argument("--bootstrap-db", action="store_true", help="Run migrate --initialize-schema")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    plan = build_staging_plan(
        args.sha,
        args.release,
        env_file=args.env_file,
        remote_alias=args.remote_alias,
        remote_dir=args.remote_dir,
        image_repo=args.image_repo,
        bootstrap_db=args.bootstrap_db,
    )
    print(f"staging alias: {plan.remote_alias}")
    print(f"remote root: {plan.remote_dir}")
    print(f"release SHA: {plan.sha}")
    print(f"release image: {plan.image_ref}")
    print("commands:")
    for command in plan.commands:
        print(f"- {command}")
    print("rollback hint: reuse previous image path from releases/<sha>/previous_image.txt")
    print(f"hostname: https://{STAGING_HOSTNAME}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
