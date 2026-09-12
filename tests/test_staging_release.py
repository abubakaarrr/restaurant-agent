from pathlib import Path
import shlex

import pytest

from scripts import staging_release


def _remote_script(command: str) -> str:
    ssh_args = shlex.split(command)
    assert ssh_args[:4] == ["ssh", "-p", "717", "staging"]
    assert len(ssh_args) == 5
    bash_args = shlex.split(ssh_args[4])
    assert bash_args[:2] == ["bash", "-lc"]
    assert len(bash_args) == 3
    return bash_args[2]


def _production_env_file(tmp_path: Path) -> Path:
    env = tmp_path / ".env.production.example"
    env.write_text(
        "\n".join(
            [
                "POSTGRES_PASSWORD=StrongPassword123!",
                "DATABASE_URL=postgresql://postgres:StrongPassword123!@127.0.0.1:5432/restaurant_agent",
                "OPENAI_API_KEY=sk-proj-placeholder",
                "RETELL_API_KEY=secret-retell",
                "RETELL_AGENT_ID=agent-123",
                "RETELL_CHAT_AGENT_ID=",
                "RETELL_PUBLIC_KEY=public-key",
                "RETELL_PHONE_NUMBER=+15035550148",
                "STAFF_TRANSFER_NUMBER=+15035550148",
                "RETELL_COUNTRY_CODE=US",
                "RETELL_AREA_CODE=",
                "RETELL_NUMBER_PROVIDER=twilio",
                "RETELL_AGENT_VERSION=latest_published",
                "RETELL_FALLBACK_NUMBER=",
                "VOICE_TOOL_SECRET=super-long-voice-tool-secret",
                "RETELL_WS_TOKEN=super-long-retell-ws-token",
                "VOICE_LIVE_WRITES_ENABLED=false",
                "ENABLE_LEGACY_RETELL_CUSTOM_LLM=false",
                "ENABLE_LEGACY_VAPI=false",
                "ENABLE_PUBLIC_WEB_CALLS=false",
                "WIDGET_ENABLED=false",
                "WIDGET_MODE=hybrid",
                "WIDGET_ALLOWED_DOMAINS=https://agent.servicesground.com",
                "WIDGET_TITLE=Talk to our restaurant host",
                "WIDGET_LOGO_URL=",
                "WIDGET_COLOR=",
                "WIDGET_FAB_TEXT=How can we help?",
                "CALLBACK_COUNTRIES=US,CA",
                "CALLBACK_TERMS_URL=https://agent.servicesground.com/privacy",
                "RECAPTCHA_SITE_KEY=site-key",
                "APP_ENV=production",
                "ALLOWED_ORIGINS=https://agent.servicesground.com",
                "LOGIN_USERNAME=admin",
                "LOGIN_PASSWORD=admin-secret-password",
                "SESSION_SECRET=this-is-a-very-long-session-secret",
                "DASHBOARD_API_KEY=dashboard-secret-with-enough-length",
            ]
        ),
        encoding="utf-8",
    )
    return env


def test_staging_plan_requires_release_flag(tmp_path: Path) -> None:
    env = _production_env_file(tmp_path)
    with pytest.raises(staging_release.ReleaseSafetyError):
        staging_release.build_staging_plan("a" * 40, release=False, env_file=env)


def test_staging_plan_requires_safe_production_values(tmp_path: Path) -> None:
    env = tmp_path / ".env.production.example"
    env.write_text("APP_ENV=development\n", encoding="utf-8")
    with pytest.raises(staging_release.ReleaseSafetyError, match="APP_ENV must be set to production"):
        staging_release.build_staging_plan("a" * 40, release=True, env_file=env)


def test_staging_plan_reuses_application_security_validation(tmp_path: Path) -> None:
    env = _production_env_file(tmp_path)
    env.write_text(
        env.read_text(encoding="utf-8").replace(
            "DASHBOARD_API_KEY=dashboard-secret-with-enough-length",
            "DASHBOARD_API_KEY=short",
        ),
        encoding="utf-8",
    )
    with pytest.raises(staging_release.ReleaseSafetyError, match="DASHBOARD_API_KEY"):
        staging_release.build_staging_plan("a" * 40, release=True, env_file=env)


def test_staging_plan_requires_immutable_commit_sha(tmp_path: Path) -> None:
    env = _production_env_file(tmp_path)
    with pytest.raises(staging_release.ReleaseSafetyError, match="40-character hexadecimal"):
        staging_release.build_staging_plan("latest", release=True, env_file=env)


def test_staging_plan_generates_expected_commands(tmp_path: Path) -> None:
    env = _production_env_file(tmp_path)
    plan = staging_release.build_staging_plan(
        "a" * 40,
        release=True,
        env_file=env,
        remote_dir="/opt/restaurant-agent",
        image_repo="ghcr.io/abubakaarrr/restaurant-agent",
        bootstrap_db=False,
    )

    remote_scripts = [_remote_script(command) for command in plan.commands]
    all_scripts = "\n".join(remote_scripts)
    assert plan.image_ref == "ghcr.io/abubakaarrr/restaurant-agent:" + "a" * 40
    assert len(plan.commands) == 1
    assert "RESTAURANT_IMAGE_TAG=" + plan.image_ref in all_scripts
    assert "python scripts/migrate.py" in all_scripts
    assert "docker compose up -d --no-build db web" in all_scripts
    assert "curl -fsS https://agent.servicesground.com/health" in all_scripts
    assert all(script.startswith("set -euo pipefail && ") for script in remote_scripts)


def test_staging_plan_bootstrap_uses_initialize_schema_when_requested(tmp_path: Path) -> None:
    env = _production_env_file(tmp_path)
    plan = staging_release.build_staging_plan(
        "a" * 40,
        release=True,
        env_file=env,
        bootstrap_db=True,
    )
    assert any(
        "python scripts/migrate.py --initialize-schema" in _remote_script(command)
        for command in plan.commands
    )


def test_staging_plan_quotes_shell_parameters(tmp_path: Path) -> None:
    env = _production_env_file(tmp_path)
    plan = staging_release.build_staging_plan(
        "a" * 40,
        release=True,
        env_file=env,
        remote_dir="/opt/restaurant-agent; touch /tmp/should-not-run",
        image_repo="ghcr.io/example/agent; touch /tmp/should-not-run",
    )

    tokens = [
        token
        for command in plan.commands
        for token in shlex.split(_remote_script(command))
    ]
    assert "/opt/restaurant-agent; touch /tmp/should-not-run/releases/" + "a" * 40 in tokens
    assert "ghcr.io/example/agent; touch /tmp/should-not-run:" + "a" * 40 in tokens
    assert "touch" not in tokens


def test_staging_plan_rejects_non_staging_destination(tmp_path: Path) -> None:
    env = _production_env_file(tmp_path)
    with pytest.raises(TypeError):
        staging_release.build_staging_plan(
            "a" * 40,
            release=True,
            env_file=env,
            remote_alias="production",
        )
