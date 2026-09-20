"""Database boundary for the development-only native voice adapter."""

from __future__ import annotations

import contextvars
import ipaddress
import os
import socket
from urllib.parse import urlparse

from app.config import settings


class NativeVoiceDatabaseGuardError(RuntimeError):
    """Raised when native voice is not pointed at an approved disposable database."""


_active_database_url: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "native_voice_database_url", default=None
)


def _database_identity(url: str) -> tuple[frozenset[str], int, str]:
    parsed = urlparse(url)
    host = parsed.hostname
    if not host:
        raise NativeVoiceDatabaseGuardError("native_voice_database_host_required")
    try:
        port = parsed.port or 5432
    except ValueError as exc:
        raise NativeVoiceDatabaseGuardError("native_voice_database_port_invalid") from exc
    try:
        addresses = {
            str(ipaddress.ip_address(result[4][0]))
            for result in socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
        }
    except (OSError, ValueError) as exc:
        raise NativeVoiceDatabaseGuardError("native_voice_database_host_unresolved") from exc
    if not addresses:
        raise NativeVoiceDatabaseGuardError("native_voice_database_host_unresolved")
    database = (parsed.path or "").lstrip("/")
    if not database:
        raise NativeVoiceDatabaseGuardError("native_voice_database_name_required")
    return frozenset(addresses), port, database


def validate_native_voice_database() -> str:
    url = os.getenv("NATIVE_VOICE_DATABASE_URL", "").strip()
    if not url:
        raise NativeVoiceDatabaseGuardError("native_voice_database_url_required")
    configured_databases = {
        os.getenv("DATABASE_URL", "").strip(),
        str(settings.database_url or "").strip(),
    }
    native_identity = _database_identity(url)
    for configured_url in configured_databases:
        if configured_url:
            configured_identity = _database_identity(configured_url)
            same_server = bool(native_identity[0] & configured_identity[0])
            same_database = native_identity[1:] == configured_identity[1:]
            if same_server and same_database:
                raise NativeVoiceDatabaseGuardError("native_voice_database_must_be_separate")
    if settings.is_production or os.getenv("APP_ENV", "development").casefold() == "production":
        raise NativeVoiceDatabaseGuardError("native_voice_database_production_forbidden")
    if os.getenv("NATIVE_VOICE_DATABASE_WRITE_ENABLED", "").casefold() != "true":
        raise NativeVoiceDatabaseGuardError("native_voice_database_writes_disabled")
    if not os.getenv("NATIVE_VOICE_DATABASE_MARKER", "").strip():
        raise NativeVoiceDatabaseGuardError("native_voice_database_marker_required")
    return url


async def verify_native_voice_database_connection(pool: object) -> None:
    marker = os.getenv("NATIVE_VOICE_DATABASE_MARKER", "").strip()
    if not marker:
        raise NativeVoiceDatabaseGuardError("native_voice_database_marker_required")
    try:
        row = await pool.fetchrow(
            "SELECT inet_server_addr()::text AS server_host, "
            "inet_server_port() AS server_port, current_database() AS database_name, "
            "current_setting('app.native_voice_disposable_marker', true) AS marker"
        )
    except Exception as exc:
        raise NativeVoiceDatabaseGuardError("native_voice_database_identity_unreadable") from exc
    if row is None:
        raise NativeVoiceDatabaseGuardError("native_voice_database_identity_unreadable")
    try:
        actual_marker = row["marker"]
        server_host = str(row["server_host"] or "")
        server_port = int(row["server_port"])
        database_name = str(row["database_name"] or "")
        server_address = str(ipaddress.ip_address(server_host))
        expected_url = active_native_voice_database_url() or os.getenv("NATIVE_VOICE_DATABASE_URL", "").strip()
        expected_addresses, expected_port, expected_database = _database_identity(expected_url)
        normal_urls = {
            os.getenv("DATABASE_URL", "").strip(),
            str(settings.database_url or "").strip(),
        }
        if any(
            configured_url
            and (server_address, server_port, database_name)
            == (address, port, database)
            for configured_url in normal_urls
            for address, port, database in [_database_identity(configured_url)]
        ):
            raise NativeVoiceDatabaseGuardError("native_voice_database_must_be_separate")
    except (KeyError, TypeError, ValueError) as exc:
        raise NativeVoiceDatabaseGuardError("native_voice_database_identity_invalid") from exc
    if (
        not isinstance(actual_marker, str)
        or not actual_marker
        or actual_marker != marker
        or server_port != expected_port
        or database_name != expected_database
        or server_address not in expected_addresses
    ):
        raise NativeVoiceDatabaseGuardError("native_voice_database_marker_mismatch")


def activate_native_voice_database() -> contextvars.Token[str | None]:
    return _active_database_url.set(validate_native_voice_database())


def deactivate_native_voice_database(token: contextvars.Token[str | None]) -> None:
    _active_database_url.reset(token)


def active_native_voice_database_url() -> str | None:
    return _active_database_url.get()
