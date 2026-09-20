"""Database boundary for the development-only native voice adapter."""

from __future__ import annotations

import contextvars
import hashlib
import os

from app.config import settings


class NativeVoiceDatabaseGuardError(RuntimeError):
    """Raised when native voice is not pointed at an approved disposable database."""


_active_database_url: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "native_voice_database_url", default=None
)


def validate_native_voice_database() -> str:
    url = os.getenv("NATIVE_VOICE_DATABASE_URL", "").strip()
    if not url:
        raise NativeVoiceDatabaseGuardError("native_voice_database_url_required")
    configured_databases = {
        os.getenv("DATABASE_URL", "").strip(),
        str(settings.database_url or "").strip(),
    }
    if url in configured_databases:
        raise NativeVoiceDatabaseGuardError("native_voice_database_must_be_separate")
    if settings.is_production or os.getenv("APP_ENV", "development").casefold() == "production":
        raise NativeVoiceDatabaseGuardError("native_voice_database_production_forbidden")
    if os.getenv("NATIVE_VOICE_DATABASE_WRITE_ENABLED", "").casefold() != "true":
        raise NativeVoiceDatabaseGuardError("native_voice_database_writes_disabled")
    expected = hashlib.sha256(url.encode("utf-8")).hexdigest()
    if os.getenv("NATIVE_VOICE_DISPOSABLE_DATABASE_FINGERPRINT", "").strip() != expected:
        raise NativeVoiceDatabaseGuardError("native_voice_database_fingerprint_required")
    return url


def activate_native_voice_database() -> contextvars.Token[str | None]:
    return _active_database_url.set(validate_native_voice_database())


def deactivate_native_voice_database(token: contextvars.Token[str | None]) -> None:
    _active_database_url.reset(token)


def active_native_voice_database_url() -> str | None:
    return _active_database_url.get()
