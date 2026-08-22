"""Application settings loaded from environment variables.

Production defaults deliberately fail closed.  The legacy Vapi and custom-LLM
Retell adapters remain available as rollback paths, but the managed Retell
Conversation Flow should use the authenticated tool API.
"""

from functools import lru_cache
import re
from urllib.parse import urlparse

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Database
    database_url: str = "postgresql://postgres:password@localhost:5432/restaurant_agent"

    # LLM keys
    openai_api_key: str = ""

    # Vapi (voice calls via browser SDK)
    vapi_public_key: str = ""
    vapi_assistant_id: str = ""
    vapi_server_secret: str = ""

    # LangSmith tracing (optional)
    langchain_tracing_v2: bool = False
    langchain_api_key: str = ""
    langchain_project: str = "restaurant-agent-poc"

    # Restaurant identity
    restaurant_name: str = "The Lamplighter Public House"
    restaurant_timezone: str = "America/Vancouver"
    restaurant_settings_file: str = "restaurant_settings.json"

    # Voice Cloning / TTS
    chatterbox_api_url: str = "http://localhost:8080"
    chatterbox_api_key: str = "local-dev-key"
    # Which cloned voice Vapi should use on live calls (e.g. "custom/sana.wav").
    vapi_voice_key: str = ""

    # Retell (custom-LLM voice orchestration)
    retell_api_key: str = ""
    # The Retell agent to dial for browser web calls (from the Retell dashboard).
    retell_agent_id: str = ""
    retell_chat_agent_id: str = ""
    retell_public_key: str = ""
    # Secret embedded in the custom-LLM WebSocket URL. Retell does not send its
    # REST API key on WebSocket connections.
    retell_ws_token: str = ""
    # Dedicated shared secret for Retell managed-flow custom functions.
    voice_tool_secret: str = ""
    retell_phone_number: str = ""
    staff_transfer_number: str = ""

    # Production voice rollout flags. Writes default off until idempotency and
    # real-carrier acceptance tests have passed.
    voice_live_writes_enabled: bool = False
    enable_legacy_vapi: bool = False
    enable_legacy_retell_custom_llm: bool = True
    enable_public_web_calls: bool = False

    # Widget configuration. These are public browser credentials, never the
    # RETELL_API_KEY.
    widget_enabled: bool = False
    widget_mode: str = "hybrid"
    widget_allowed_domains: str = ""
    widget_title: str = "Talk to our restaurant host"
    widget_logo_url: str = ""
    widget_color: str = ""
    widget_fab_text: str = "How can we help?"
    callback_countries: str = "US,CA,PK"
    caller_phone_country_code: str = ""
    callback_terms_url: str = ""
    recaptcha_site_key: str = ""

    # Agent behavior and data policy.
    ai_agent_name: str = "Clough"
    supported_locales: str = "en-US"
    call_data_retention_days: int = 30
    call_recording_enabled: bool = False
    store_call_transcripts: bool = False
    rag_similarity_threshold: float = 0.72

    # Runtime model settings used by the legacy LangGraph rollback adapter.
    llm_model: str = "gpt-5.6-terra"
    llm_temperature: float = 0.3
    llm_max_tokens: int = 160
    llm_reasoning_effort: str = "none"

    # Dashboard / internal API protection
    # Set a strong random string in .env; required on X-API-Key header for /api/* routes.
    dashboard_api_key: str = ""

    # Single shared login for the browser dashboard (POC)
    login_username: str = "admin"
    login_password: str = "admin"
    # Signs the session cookie — use a long random string in production
    session_secret: str = ""

    # App
    app_env: str = "development"
    allowed_origins: str = "http://localhost:8000,http://127.0.0.1:8000"

    @property
    def is_production(self) -> bool:
        return self.app_env.strip().lower() == "production"

    @staticmethod
    def _csv(value: str) -> list[str]:
        return [part.strip() for part in value.split(",") if part.strip()]

    @property
    def cors_origins(self) -> list[str]:
        return self._csv(self.allowed_origins)

    @property
    def locale_list(self) -> list[str]:
        return self._csv(self.supported_locales) or ["en-US"]

    @property
    def widget_domain_list(self) -> list[str]:
        return self._csv(self.widget_allowed_domains)

    @property
    def default_caller_country_code(self) -> str:
        explicit = self.caller_phone_country_code.strip().lstrip("+")
        if explicit.isdigit():
            return explicit
        iso_to_code = {
            "US": "1",
            "CA": "1",
            "PK": "92",
            "GB": "44",
            "UK": "44",
            "IN": "91",
            "AE": "971",
            "SA": "966",
            "AU": "61",
        }
        for region in self._csv(self.callback_countries):
            code = iso_to_code.get(region.upper())
            if code:
                return code
        return "1"

    def validate_runtime_security(self) -> None:
        """Reject unsafe production configuration before accepting traffic."""
        if not self.is_production:
            return

        failures: list[str] = []
        required = {
            "DASHBOARD_API_KEY": self.dashboard_api_key,
            "SESSION_SECRET": self.session_secret,
            "VOICE_TOOL_SECRET": self.voice_tool_secret,
            "RETELL_API_KEY": self.retell_api_key,
            "RETELL_AGENT_ID": self.retell_agent_id,
        }
        for name, value in required.items():
            minimum = 8 if name in {"RETELL_API_KEY", "RETELL_AGENT_ID"} else 24
            if len(value.strip()) < minimum:
                failures.append(f"{name} must contain at least {minimum} characters")

        if "*" in self.cors_origins:
            failures.append("ALLOWED_ORIGINS must not contain '*' in production")
        for origin in self.cors_origins:
            parsed = urlparse(origin)
            if parsed.scheme != "https":
                failures.append(f"ALLOWED_ORIGINS must use HTTPS in production: {origin}")
        if self.enable_legacy_retell_custom_llm and len(self.retell_ws_token) < 24:
            failures.append(
                "RETELL_WS_TOKEN must contain at least 24 characters while the legacy "
                "custom-LLM adapter is enabled"
            )
        if self.enable_public_web_calls and not self.retell_agent_id:
            failures.append("RETELL_AGENT_ID is required when public web calls are enabled")
        for name, number in (
            ("RETELL_PHONE_NUMBER", self.retell_phone_number),
            ("STAFF_TRANSFER_NUMBER", self.staff_transfer_number),
        ):
            if number and not re.fullmatch(r"\+[1-9]\d{7,14}", number.strip()):
                failures.append(f"{name} must use E.164 format")
        if self.widget_enabled:
            if not self.retell_public_key:
                failures.append("RETELL_PUBLIC_KEY is required when WIDGET_ENABLED=true")
            if not self.widget_domain_list:
                failures.append(
                    "WIDGET_ALLOWED_DOMAINS is required when WIDGET_ENABLED=true"
                )
            if self.widget_mode not in {"hybrid", "callback"}:
                failures.append("WIDGET_MODE must be 'hybrid' or 'callback'")
            if self.widget_mode == "callback":
                if not self.retell_phone_number:
                    failures.append(
                        "RETELL_PHONE_NUMBER is required for callback widget mode"
                    )
                if not self.callback_terms_url.startswith("https://"):
                    failures.append(
                        "CALLBACK_TERMS_URL must use HTTPS for callback widget mode"
                    )

        if failures:
            raise RuntimeError("Unsafe production configuration: " + "; ".join(failures))


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
