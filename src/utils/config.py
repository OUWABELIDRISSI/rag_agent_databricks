"""Centralized configuration using Pydantic Settings."""

from functools import lru_cache

from pydantic import Field, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── LLM ──────────────────────────────────────────────────────────
    anthropic_api_key: str = Field(..., description="Anthropic API key")
    openrouter_api_key: str = Field("", description="OpenRouter API key")
    mistral_api_key: str = Field("", description="Mistral API key for embeddings")
    llm_model: str = Field("anthropic/claude-3-haiku")
    llm_max_tokens: int = Field(2048, ge=256, le=8096)
    llm_temperature: float = Field(0.0, ge=0.0, le=1.0)

    # ── Embeddings ────────────────────────────────────────────────────
    embedding_model: str = Field("BAAI/bge-m3")
    embedding_dimension: int = Field(1024, ge=128, le=4096)

    # ── PostgreSQL ────────────────────────────────────────────────────
    postgres_host: str = Field("localhost")
    postgres_port: int = Field(5432)
    postgres_db: str = Field("rag_databricks")
    postgres_user: str = Field("rag_user")
    postgres_password: str = Field("changeme")

    @computed_field  # type: ignore[prop-decorator]
    @property
    def postgres_url(self) -> str:
        return (
            f"postgresql+psycopg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    # ── Ingestion ─────────────────────────────────────────────────────
    chunk_size: int = Field(512, ge=128, le=2048)
    chunk_overlap: int = Field(64, ge=0, le=256)

    # ── API ───────────────────────────────────────────────────────────
    api_host: str = Field("0.0.0.0")
    api_port: int = Field(8000)
    api_secret_key: str = Field("changeme-secret-key")
    allowed_origins: list[str] = Field(default=["http://localhost:3000"])

    # ── Evaluation ────────────────────────────────────────────────────
    langfuse_public_key: str = Field("")
    langfuse_secret_key: str = Field("")
    langfuse_host: str = Field("https://cloud.langfuse.com")

    # ── Environnement ─────────────────────────────────────────────────
    environment: str = Field("development")
    log_level: str = Field("INFO")

    @property
    def is_production(self) -> bool:
        return self.environment == "production"

    @property
    def langfuse_enabled(self) -> bool:
        return bool(self.langfuse_public_key and self.langfuse_secret_key)


@lru_cache
def get_settings() -> Settings:
    """Return cached settings instance."""
    return Settings()  # type: ignore[call-arg]


settings = get_settings()