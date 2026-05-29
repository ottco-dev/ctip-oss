"""
backend.database — SQLModel database engine and session management.

DESIGN:
- SQLite for local development (zero setup, no Docker required)
- PostgreSQL for production (set DATABASE_URL env var)
- SQLModel for ORM (Pydantic v2 compatible, less boilerplate than SQLAlchemy)
- Session lifecycle: one session per request, closed on response

MIGRATION:
Uses Alembic for schema migrations:
  alembic init alembic
  alembic revision --autogenerate -m "Initial schema"
  alembic upgrade head

For development: create_all_tables() is sufficient.
"""

from __future__ import annotations

from typing import Generator

from sqlmodel import SQLModel, Session, create_engine

from backend.config import get_settings


def get_engine():
    """Create SQLModel engine from settings."""
    settings = get_settings()

    connect_args = {}
    if settings.database_url.startswith("sqlite"):
        # SQLite requires check_same_thread=False for FastAPI multi-thread usage
        connect_args["check_same_thread"] = False

    return create_engine(
        settings.database_url,
        echo=settings.database_echo,
        connect_args=connect_args,
    )


# Module-level engine (created once)
engine = get_engine()


def create_all_tables() -> None:
    """
    Create all database tables.

    Called on application startup. Idempotent — safe to call multiple times.
    For production, use Alembic migrations instead.
    """
    # Import all models to register with SQLModel metadata
    from backend.models import experiment, dataset, job, model_registry, session  # noqa: F401
    from backend.api.v1 import model_tests  # noqa: F401 — registers ModelTest table
    SQLModel.metadata.create_all(engine)


def get_session() -> Generator[Session, None, None]:
    """
    FastAPI dependency: database session per request.

    Usage in routes:
        @router.get("/items")
        def get_items(db: Session = Depends(get_session)):
            items = db.exec(select(Item)).all()
    """
    with Session(engine) as session:
        yield session
