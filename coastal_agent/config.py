"""Environment-driven configuration.

Loads from `/opt/coastal-agent.env` on the droplet (via systemd EnvironmentFile)
or from a local `.env` during development. The file `/opt/coastal-agent.env`
itself is never committed to the repo.

Field names are lowercase; pydantic-settings does case-insensitive matching
against env vars (LLM_API_KEY → llm_api_key etc).
"""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- Paths ---
    state_db_path: Path = Path("/var/lib/coastal-agent/state.db")
    corpus_dir: Path = Path("corpus")
    vendor_dir: Path = Path("external/gnn_drl_ews_v003_seed2")

    # --- LLM brief composition (OpenRouter via OpenAI SDK by default) ---
    llm_api_key: str = ""
    llm_model: str = "anthropic/claude-sonnet-4.6"
    llm_base_url: str = "https://openrouter.ai/api/v1"
    llm_enabled: bool = True
    llm_max_briefs_per_day: int = 50
    llm_max_tool_iterations: int = 8
    llm_temperature: float = 0.0

    # --- Email (Resend HTTPS API; SMTP is blocked on DigitalOcean) ---
    email_mode: str = "mock"  # 'real' or 'mock'
    resend_api_key: str = ""
    email_from: str = ""      # 'onboarding@resend.dev' until a domain is verified
    email_to: str = ""        # comma-separated list

    # --- Legacy SMTP fields (retained for env-file compatibility; ignored) ---
    smtp_host: str = ""
    smtp_port: int = 587
    smtp_user: str = ""
    smtp_password: str = ""

    # --- Weather + ISPRA ---
    weather_provider: str = "open-meteo"
    weather_api_key: str = ""
    ispra_gauge_id: str = "lido_diga_sud"

    # --- Dashboard ---
    dashboard_base_url: str = "http://127.0.0.1:8000"

    # --- Scheduler ---
    poll_interval_seconds: int = 3600

    # --- Trigger thresholds (override default §3.5 values) ---
    trigger_activation_surge_cm: float = 110.0
    trigger_standdown_surge_cm: float = 100.0
    trigger_standdown_consecutive_ticks: int = 2


settings = Settings()
