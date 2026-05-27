"""
alembic/env.py — Alembic environment configuration for Trichome Analysis Platform.

Supports both:
  - Online migrations (direct DB connection for `alembic upgrade head`)
  - Offline migrations (SQL script generation for `alembic upgrade --sql`)

SQLModel table metadata is imported from backend.models.* so Alembic
can auto-detect schema changes via `alembic revision --autogenerate`.

Usage:
    # Apply all pending migrations
    alembic upgrade head

    # Generate new migration from model changes
    alembic revision --autogenerate -m "add column X to runs"

    # Downgrade one step
    alembic downgrade -1

    # Generate SQL script (offline mode, for production)
    alembic upgrade head --sql > migration.sql

Database URL:
    Uses DATABASE_URL env var (same as backend.config.Settings.database_url).
    Default: sqlite:///./trichome.db
    Production: postgresql://trichome:password@localhost:5432/trichome

    Set in .env or as env var:
      DATABASE_URL=postgresql://trichome:pass@localhost/trichome
"""

from __future__ import annotations

import os
from logging.config import fileConfig

from sqlalchemy import engine_from_config, pool
from alembic import context
from sqlmodel import SQLModel

# ── Import all models so their metadata is registered ──────────────────────
# Add any new model modules here so Alembic detects schema changes.
try:
    from backend.models.experiment import Experiment, Run, Metric  # noqa: F401
except ImportError:
    pass

try:
    from backend.models.dataset import Dataset, Sample  # noqa: F401
except ImportError:
    pass

try:
    from backend.models.job import BackgroundJob  # noqa: F401
except ImportError:
    pass

try:
    from backend.models.model_registry import ModelRecord  # noqa: F401
except ImportError:
    pass

try:
    from backend.models.session import AnalysisSession  # noqa: F401
except ImportError:
    pass

# ── Alembic Config ──────────────────────────────────────────────────────────
config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# SQLModel metadata — populated by the model imports above
target_metadata = SQLModel.metadata


# ── Database URL resolution ─────────────────────────────────────────────────

def get_database_url() -> str:
    """
    Resolve database URL from environment.

    Priority:
    1. DATABASE_URL env var (production override)
    2. alembic.ini [alembic] sqlalchemy.url (if not placeholder)
    3. Default SQLite for development
    """
    env_url = os.environ.get("DATABASE_URL", "")
    if env_url:
        return env_url

    ini_url = config.get_main_option("sqlalchemy.url", "")
    if ini_url and ini_url != "driver://user:pass@localhost/dbname":
        return ini_url

    return "sqlite:///./trichome.db"


# ── Offline migrations (SQL script generation) ──────────────────────────────

def run_migrations_offline() -> None:
    """
    Run migrations without a database connection.

    Outputs SQL to stdout. Useful for production deployments
    with restricted DB access or for audit/review.
    """
    url = get_database_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        render_as_batch=True,  # Required for SQLite ALTER TABLE support
    )

    with context.begin_transaction():
        context.run_migrations()


# ── Online migrations (direct DB connection) ────────────────────────────────

def run_migrations_online() -> None:
    """
    Run migrations with a live database connection.

    Standard mode for `alembic upgrade head`.
    """
    url = get_database_url()

    config_section = config.get_section(config.config_ini_section, {})
    config_section["sqlalchemy.url"] = url

    connectable = engine_from_config(
        config_section,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
            render_as_batch=True,
        )

        with context.begin_transaction():
            context.run_migrations()


# ── Entry point ─────────────────────────────────────────────────────────────

if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
