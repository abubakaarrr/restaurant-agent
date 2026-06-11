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

    # App
    app_env: str = "development"


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
