from pathlib import Path

import scripts.bootstrap_local_secrets as bootstrap


def test_bootstrap_adds_missing_secrets_without_overwriting_existing(
    tmp_path: Path,
    monkeypatch,
) -> None:
    env_file = tmp_path / ".env"
    existing = "x" * 48
    env_file.write_text(
        f"SESSION_SECRET={existing}\nVOICE_TOOL_SECRET=\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(bootstrap, "ENV_FILE", env_file)

    changed = bootstrap.apply()
    content = env_file.read_text(encoding="utf-8")
    assert "VOICE_TOOL_SECRET=" in content
    assert "RETELL_WS_TOKEN=" in content
    assert "DASHBOARD_API_KEY=" in content
    assert f"SESSION_SECRET={existing}" in content
    assert "SESSION_SECRET" not in changed
    assert bootstrap.apply() == []
