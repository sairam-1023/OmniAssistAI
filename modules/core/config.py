"""
Environment-driven application settings.

All configurable values (paths, API keys, model locations) are defined
here, sourced from environment variables / a .env file. Nothing in this
file should ever contain a real secret — see .env.example for the
template committed to the repo instead.
"""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- General ---
    app_name: str = "OmniAssist AI"
    environment: str = "development"   # development | production

    # --- Storage paths ---
    storage_dir: Path = Path("storage")
    sqlite_path: Path = Path("storage/omniassist.db")

    # --- API auth (Week 7) ---
    api_key: str = "changeme-dev-key"

    # --- Model paths (filled in as we build Weeks 2-6) ---
    analytics_model_path: Path = Path("models/analytics/model.joblib")
    vision_model_path: Path = Path("models/vision/resnet18_finetuned.pt")


@lru_cache
def get_settings() -> Settings:
    """
    Returns a cached Settings instance.

    @lru_cache means Settings() is only constructed once, the first time
    this is called — every subsequent call across the whole app returns
    the same cached object instead of re-reading environment variables
    and re-validating every time. Standard pattern for app-wide config.
    """
    return Settings()