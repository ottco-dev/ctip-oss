"""
backend.config — Application settings via pydantic-settings.

Settings are loaded from environment variables + .env file.
Precedence: env vars > .env file > defaults.

Usage:
    from backend.config import get_settings
    settings = get_settings()
    print(settings.database_url)
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application-wide settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── APP ─────────────────────────────────────────────────────────
    app_name: str = "Trichome Analysis Platform"
    app_version: str = "0.1.0"
    debug: bool = False
    environment: str = "development"

    # ── API ─────────────────────────────────────────────────────────
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_prefix: str = "/api/v1"
    cors_origins: list[str] = Field(
        default=[
            # Local direct access (dev)
            "http://localhost:3001",   # nginx proxy
            "http://localhost:3003",   # Next.js dev
            "http://127.0.0.1:3001",
            "http://127.0.0.1:3003",
            # Public DDNS access via nginx
            "http://your-domain.com:3001",
            "http://your-domain.com",
        ]
    )

    # ── DATABASE ────────────────────────────────────────────────────
    database_url: str = "sqlite:///./trichome.db"
    """
    SQLite for local development:   sqlite:///./trichome.db
    PostgreSQL for production:      postgresql://user:pass@localhost/trichome
    """
    database_echo: bool = False
    """Log all SQL queries (development only)."""

    # ── STORAGE ─────────────────────────────────────────────────────
    data_root: str = "./data"
    """Root directory for all data storage."""

    images_dir: str = "./data/images"
    videos_dir: str = "./data/videos"
    models_dir: str = "./data/models"
    outputs_dir: str = "./data/outputs"
    uploads_dir: str = "./data/uploads"

    max_upload_size_mb: int = 500
    """Maximum file upload size in MB."""

    # ── GPU / HARDWARE ───────────────────────────────────────────────
    cuda_device: str = "cuda:0"
    vram_limit_gb: float = 8.0
    """RTX 4060 VRAM. Used by GPU guard middleware."""

    vram_inference_budget_gb: float = 2.0
    """Reserved VRAM for inference when training is active."""

    max_concurrent_gpu_tasks: int = 1
    """
    GPU semaphore limit. 1 = serialize all GPU tasks.
    RTX 4060 (8GB): no concurrent training + inference.
    """

    gpu_inference_queue_depth: int = 0
    """
    Maximum requests that may queue for the GPU slot before returning HTTP 429.
    0 (default) = no queueing: any request that finds the slot busy gets 429 immediately.
    1 = allow 1 request to wait, subsequent ones get 429.
    Increase for endpoints with acceptable latency tolerance (e.g. batch jobs).
    """

    # ── EXPERIMENT TRACKING ──────────────────────────────────────────
    mlflow_tracking_uri: str = "http://localhost:3004"  # host port; container uses :5000
    mlflow_experiment_name: str = "trichome-detection"

    use_wandb: bool = False
    wandb_api_key: str = ""
    wandb_project: str = "trichome-detection"

    # ── VLM LABELING ─────────────────────────────────────────────────
    default_vlm_backend: str = "moondream"
    vlm_min_confidence: float = 0.40

    # ── ANNOTATION ──────────────────────────────────────────────────
    cvat_url: str = "http://localhost:3006"       # host port 3006 → container 8080
    cvat_username: str = "admin"
    cvat_password: str = "admin"

    label_studio_url: str = "http://localhost:3005"  # host port 3005 → container 8080
    label_studio_api_key: str = ""

    # ── SECURITY ─────────────────────────────────────────────────────
    secret_key: str = "dev-secret-key-change-in-production"
    """Used for session tokens. Change for production."""

    api_token: str = ""
    """
    Single-user API token. Set to enable authentication.
    Empty string (default) = authentication disabled (development mode).
    Set via env var: API_TOKEN=your-secret-token

    All requests must include one of:
      Authorization: Bearer <token>
      X-API-Key: <token>
      ?api_key=<token>
    """

    # ── LOGGING ──────────────────────────────────────────────────────
    log_level: str = "INFO"
    log_file: str | None = "logs/backend.log"

    # ── WEBSOCKET ────────────────────────────────────────────────────
    ws_heartbeat_interval_s: float = 30.0
    training_metrics_interval_s: float = 2.0
    """How often to broadcast training metrics over WebSocket."""

    gpu_poll_interval_s: float = 2.0
    """How often to poll GPU stats for system dashboard."""

    @field_validator("data_root", "images_dir", "videos_dir", "models_dir",
                     "outputs_dir", "uploads_dir", mode="before")
    @classmethod
    def expand_paths(cls, v: str) -> str:
        return str(Path(v).expanduser().resolve())

    @property
    def max_upload_size_bytes(self) -> int:
        return self.max_upload_size_mb * 1024 * 1024

    @property
    def is_production(self) -> bool:
        return self.environment == "production"

    def ensure_dirs(self) -> None:
        """Create required storage directories if they don't exist."""
        for dir_path in [
            self.data_root,
            self.images_dir,
            self.videos_dir,
            self.models_dir,
            self.outputs_dir,
            self.uploads_dir,
        ]:
            Path(dir_path).mkdir(parents=True, exist_ok=True)

        if self.log_file:
            Path(self.log_file).parent.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Get application settings (cached singleton).

    Cache is invalidated by calling get_settings.cache_clear().
    Useful in tests where settings need to be overridden.
    """
    return Settings()
