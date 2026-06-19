"""Application settings loaded from environment variables."""

from functools import lru_cache
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
    anthropic_api_key: str = ""
    openai_api_key: str = ""

    # Vapi (voice calls via browser SDK)
    vapi_public_key: str = ""
    vapi_assistant_id: str = ""

    # LangSmith tracing (optional)
    langchain_tracing_v2: bool = False
    langchain_api_key: str = ""
    langchain_project: str = "restaurant-agent-poc"

    # Restaurant identity
    restaurant_name: str = "La Casa Restaurant"
    restaurant_timezone: str = "UTC"

    # Voice Cloning / TTS
    chatterbox_api_url: str = "http://localhost:8080"
    chatterbox_api_key: str = "local-dev-key"
    # Which cloned voice Vapi should use on live calls (e.g. "custom/sana.wav").
    vapi_voice_key: str = ""

    # Retell (custom-LLM voice orchestration)
    retell_api_key: str = ""
    # The Retell agent to dial for browser web calls (from the Retell dashboard).
    retell_agent_id: str = ""

    # App
    app_env: str = "development"


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
