from app.db_pool import pool_kwargs


def test_local_dsn_uses_ipv4_and_disables_ssl() -> None:
    kwargs = pool_kwargs("postgresql://postgres:x@localhost:5434/restaurant_agent")
    assert "127.0.0.1" in kwargs["dsn"]
    assert "localhost" not in kwargs["dsn"]
    assert kwargs["ssl"] is False


def test_explicit_ssl_require_is_preserved() -> None:
    kwargs = pool_kwargs(
        "postgresql://postgres:x@localhost:5434/restaurant_agent?sslmode=require"
    )
    assert "127.0.0.1" in kwargs["dsn"]
    assert "ssl" not in kwargs


def test_remote_host_is_unchanged() -> None:
    url = "postgresql://postgres:x@db:5432/restaurant_agent"
    kwargs = pool_kwargs(url)
    assert kwargs["dsn"] == url
    assert "ssl" not in kwargs
