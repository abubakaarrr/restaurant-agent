from pathlib import Path

import pytest

from scripts import staging_release


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
                "DASHBOARD_API_KEY=dashboard-secret",
            ]
        ),
        encoding="utf-8",
    )
    return env


def test_staging_plan_requires_release_flag(tmp_path: Path) -> None:
    env = _production_env_file(tmp_path)
    with pytest.raises(staging_release.ReleaseSafetyError):
        staging_release.build_staging_plan("abcdef", release=False, env_file=env)


def test_staging_plan_requires_safe_production_values(tmp_path: Path) -> None:
    env = tmp_path / ".env.production.example"
    env.write_text("APP_ENV=development\n", encoding="utf-8")
    with pytest.raises(staging_release.ReleaseSafetyError, match="APP_ENV must be set to production"):
        staging_release.build_staging_plan("abcdef", release=True, env_file=env)


def test_staging_plan_generates_expected_commands(tmp_path: Path) -> None:
    env = _production_env_file(tmp_path)
    plan = staging_release.build_staging_plan(
        "abc123",
        release=True,
        env_file=env,
        remote_alias="staging",
        remote_dir="/opt/restaurant-agent",
        image_repo="ghcr.io/abubakaarrr/restaurant-agent",
        bootstrap_db=False,
    )

    all_commands = " ".join(plan.commands)
    assert plan.image_ref == "ghcr.io/abubakaarrr/restaurant-agent:abc123"
    assert "ssh -p 717 staging" in all_commands
    assert "RESTAURANT_IMAGE_TAG='ghcr.io/abubakaarrr/restaurant-agent:abc123'" in all_commands
    assert "python scripts/migrate.py" in all_commands
    assert "docker compose up -d --no-build db web" in all_commands
    assert "curl -fsS https://agent.servicesground.com/health" in all_commands


def test_staging_plan_bootstrap_uses_initialize_schema_when_requested(tmp_path: Path) -> None:
    env = _production_env_file(tmp_path)
    plan = staging_release.build_staging_plan(
        "abc123",
        release=True,
        env_file=env,
        bootstrap_db=True,
    )
    assert "--initialize-schema" in " ".join(plan.commands)


def test_staging_plan_quotes_shell_parameters(tmp_path: Path) -> None:
    env = _production_env_file(tmp_path)
    plan = staging_release.build_staging_plan(
        "abc123; touch /tmp/should-not-run",
        release=True,
        env_file=env,
        remote_alias="staging; touch /tmp/should-not-run",
        remote_dir="/opt/restaurant-agent; touch /tmp/should-not-run",
        image_repo="ghcr.io/example/agent; touch /tmp/should-not-run",
    )

    all_commands = " ".join(plan.commands)
    assert "'staging; touch /tmp/should-not-run'" in all_commands
    assert "'/opt/restaurant-agent; touch /tmp/should-not-run'" in all_commands
    assert "ghcr.io/example/agent; touch /tmp/should-not-run" in all_commands
    assert "; touch /tmp/should-not-run &&" not in all_commands
