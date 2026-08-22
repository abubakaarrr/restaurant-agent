#!/usr/bin/env python3
"""Add missing local application secrets to .env without printing values."""

from __future__ import annotations

import argparse
import secrets
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = ROOT / ".env"
SECRET_KEYS = (
    "VOICE_TOOL_SECRET",
    "RETELL_WS_TOKEN",
    "SESSION_SECRET",
    "DASHBOARD_API_KEY",
)


def apply() -> list[str]:
    if not ENV_FILE.exists():
        raise RuntimeError("Create .env from .env.example first")
    original = ENV_FILE.read_text(encoding="utf-8")
    lines = original.splitlines()
    positions: dict[str, int] = {}
    values: dict[str, str] = {}
    for index, line in enumerate(lines):
        if not line or line.lstrip().startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        positions[key] = index
        values[key] = value.strip()

    changed: list[str] = []
    for key in SECRET_KEYS:
        if len(values.get(key, "")) >= 24:
            continue
        rendered = f"{key}={secrets.token_urlsafe(48)}"
        if key in positions:
            lines[positions[key]] = rendered
        else:
            lines.append(rendered)
        changed.append(key)

    if changed:
        temporary = ENV_FILE.with_suffix(".tmp")
        temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
        temporary.replace(ENV_FILE)
    return changed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write missing values. Without this flag, make no changes.",
    )
    args = parser.parse_args()
    if not args.apply:
        print("dry-run: use --apply to add missing local secrets; values are never printed")
        return 0
    changed = apply()
    print(
        "updated keys: " + ", ".join(changed)
        if changed
        else "all local secrets were already configured"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
