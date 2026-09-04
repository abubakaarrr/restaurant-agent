from app.retell_ws_auth import (
    remember_retell_call,
    retell_ws_authorized,
)


def test_recently_minted_call_is_authorized_without_query_token(monkeypatch) -> None:
    monkeypatch.setattr("app.retell_ws_auth.settings.retell_ws_token", "a" * 32)
    remember_retell_call("call_d411ec0ae37491dd493e6709fc9")
    assert retell_ws_authorized("call_d411ec0ae37491dd493e6709fc9", "")
    assert not retell_ws_authorized("call_someone_else", "")


def test_matching_query_token_authorizes_unknown_call(monkeypatch) -> None:
    monkeypatch.setattr("app.retell_ws_auth.settings.retell_ws_token", "a" * 32)
    assert retell_ws_authorized("call_inbound", "a" * 32)
    assert not retell_ws_authorized("call_inbound", "b" * 32)
