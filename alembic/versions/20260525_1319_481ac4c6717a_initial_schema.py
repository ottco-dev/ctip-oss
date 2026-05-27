"""initial_schema — Full schema for Trichome Analysis Platform

Revision ID: 481ac4c6717a
Revises:
Create Date: 2026-05-25

Tables created:
  - experiments      (ML experiment containers)
  - runs             (individual training runs)
  - metrics          (per-epoch metric series)
  - datasets         (image dataset collections)
  - samples          (per-image dataset records)
  - jobs             (background job tracking)
  - model_versions   (registered model registry)
  - analysis_sessions (pipeline analysis runs)

Notes:
  - On existing SQLite installs this migration is a no-op (tables already exist).
  - On a fresh PostgreSQL install this creates the full schema.
  - Use `alembic stamp head` on existing installs to mark as applied without running DDL.
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = '481ac4c6717a'
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create initial schema."""
    bind = op.get_bind()
    existing_tables = sa.inspect(bind).get_table_names()

    # ── experiments ──────────────────────────────────────────────────────────
    if "experiments" not in existing_tables:
        op.create_table(
            "experiments",
            sa.Column("id", sa.Integer(), nullable=False),
            sa.Column("name", sa.String(), nullable=False),
            sa.Column("description", sa.String(), nullable=False, server_default=""),
            sa.Column("tags", sa.String(), nullable=False, server_default="[]"),
            sa.Column("config_json", sa.String(), nullable=False, server_default="{}"),
            sa.Column("status", sa.String(), nullable=False, server_default="active"),
            sa.Column("created_at", sa.Float(), nullable=False),
            sa.Column("updated_at", sa.Float(), nullable=False),
            sa.PrimaryKeyConstraint("id"),
        )
        op.create_index("ix_experiments_name", "experiments", ["name"])

    # ── runs ────────────────────────────────────────────────────────────────
    if "runs" not in existing_tables:
        op.create_table(
            "runs",
            sa.Column("id", sa.Integer(), nullable=False),
            sa.Column("run_uuid", sa.String(), nullable=False),
            sa.Column("experiment_id", sa.Integer(), nullable=False),
            sa.Column("model_variant", sa.String(), nullable=False, server_default=""),
            sa.Column("config_json", sa.String(), nullable=False, server_default="{}"),
            sa.Column("status", sa.String(), nullable=False, server_default="pending"),
            sa.Column("started_at", sa.Float(), nullable=True),
            sa.Column("finished_at", sa.Float(), nullable=True),
            sa.Column("duration_s", sa.Float(), nullable=True),
            sa.Column("total_epochs", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("best_epoch", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("best_map50", sa.Float(), nullable=False, server_default="0.0"),
            sa.Column("best_map50_95", sa.Float(), nullable=False, server_default="0.0"),
            sa.Column("best_precision", sa.Float(), nullable=False, server_default="0.0"),
            sa.Column("best_recall", sa.Float(), nullable=False, server_default="0.0"),
            sa.Column("best_model_path", sa.String(), nullable=False, server_default=""),
            sa.Column("mlflow_run_id", sa.String(), nullable=True),
            sa.Column("notes", sa.String(), nullable=False, server_default=""),
            sa.ForeignKeyConstraint(["experiment_id"], ["experiments.id"]),
            sa.PrimaryKeyConstraint("id"),
        )
        op.create_index("ix_runs_run_uuid", "runs", ["run_uuid"])
        op.create_index("ix_runs_experiment_id", "runs", ["experiment_id"])

    # ── metrics ─────────────────────────────────────────────────────────────
    if "metrics" not in existing_tables:
        op.create_table(
            "metrics",
            sa.Column("id", sa.Integer(), nullable=False),
            sa.Column("run_id", sa.Integer(), nullable=False),
            sa.Column("epoch", sa.Integer(), nullable=False),
            sa.Column("key", sa.String(), nullable=False),
            sa.Column("value", sa.Float(), nullable=False),
            sa.ForeignKeyConstraint(["run_id"], ["runs.id"]),
            sa.PrimaryKeyConstraint("id"),
        )
        op.create_index("ix_metrics_run_id", "metrics", ["run_id"])
        op.create_index("ix_metrics_key", "metrics", ["key"])

    # ── datasets ─────────────────────────────────────────────────────────────
    if "datasets" not in existing_tables:
        op.create_table(
            "datasets",
            sa.Column("id", sa.Integer(), nullable=False),
            sa.Column("name", sa.String(), nullable=False),
            sa.Column("description", sa.String(), nullable=False, server_default=""),
            sa.Column("version", sa.String(), nullable=False, server_default="1.0.0"),
            sa.Column("storage_path", sa.String(), nullable=False, server_default=""),
            sa.Column("metadata_json", sa.String(), nullable=False, server_default="{}"),
            sa.Column("split_config_json", sa.String(), nullable=False,
                      server_default='{"train":0.7,"val":0.2,"test":0.1}'),
            sa.Column("class_names_json", sa.String(), nullable=False,
                      server_default='["capitate_stalked","capitate_sessile","bulbous","non_glandular"]'),
            sa.Column("num_samples", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("num_annotated", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("num_reviewed", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("parent_dataset_id", sa.Integer(), nullable=True),
            sa.Column("status", sa.String(), nullable=False, server_default="active"),
            sa.Column("created_at", sa.Float(), nullable=False),
            sa.Column("updated_at", sa.Float(), nullable=False),
            sa.ForeignKeyConstraint(["parent_dataset_id"], ["datasets.id"]),
            sa.PrimaryKeyConstraint("id"),
        )
        op.create_index("ix_datasets_name", "datasets", ["name"])

    # ── samples ──────────────────────────────────────────────────────────────
    if "samples" not in existing_tables:
        op.create_table(
            "samples",
            sa.Column("id", sa.Integer(), nullable=False),
            sa.Column("dataset_id", sa.Integer(), nullable=False),
            sa.Column("file_path", sa.String(), nullable=False),
            sa.Column("file_name", sa.String(), nullable=False, server_default=""),
            sa.Column("width", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("height", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("annotation_path", sa.String(), nullable=False, server_default=""),
            sa.Column("annotation_source", sa.String(), nullable=False, server_default="manual"),
            sa.Column("is_annotated", sa.Boolean(), nullable=False, server_default="0"),
            sa.Column("is_reviewed", sa.Boolean(), nullable=False, server_default="0"),
            sa.Column("quality_score", sa.Float(), nullable=True),
            sa.Column("split", sa.String(), nullable=False, server_default="train"),
            sa.Column("metadata_json", sa.String(), nullable=False, server_default="{}"),
            sa.Column("created_at", sa.Float(), nullable=False),
            sa.ForeignKeyConstraint(["dataset_id"], ["datasets.id"]),
            sa.PrimaryKeyConstraint("id"),
        )
        op.create_index("ix_samples_dataset_id", "samples", ["dataset_id"])

    # ── jobs ─────────────────────────────────────────────────────────────────
    if "jobs" not in existing_tables:
        op.create_table(
            "jobs",
            sa.Column("id", sa.Integer(), nullable=False),
            sa.Column("job_uuid", sa.String(), nullable=False),
            sa.Column("job_type", sa.String(), nullable=False),
            sa.Column("status", sa.String(), nullable=False, server_default="pending"),
            sa.Column("progress", sa.Float(), nullable=False, server_default="0.0"),
            sa.Column("params_json", sa.String(), nullable=False, server_default="{}"),
            sa.Column("result_json", sa.String(), nullable=False, server_default="{}"),
            sa.Column("error_message", sa.String(), nullable=True),
            sa.Column("created_at", sa.Float(), nullable=False),
            sa.Column("started_at", sa.Float(), nullable=True),
            sa.Column("finished_at", sa.Float(), nullable=True),
            sa.Column("experiment_id", sa.Integer(), nullable=True),
            sa.Column("dataset_id", sa.Integer(), nullable=True),
            sa.Column("run_uuid", sa.String(), nullable=True),
            sa.ForeignKeyConstraint(["experiment_id"], ["experiments.id"]),
            sa.ForeignKeyConstraint(["dataset_id"], ["datasets.id"]),
            sa.PrimaryKeyConstraint("id"),
        )
        op.create_index("ix_jobs_job_uuid", "jobs", ["job_uuid"])
        op.create_index("ix_jobs_job_type", "jobs", ["job_type"])
        op.create_index("ix_jobs_status", "jobs", ["status"])

    # ── model_versions ────────────────────────────────────────────────────────
    if "model_versions" not in existing_tables:
        op.create_table(
            "model_versions",
            sa.Column("id", sa.Integer(), nullable=False),
            sa.Column("name", sa.String(), nullable=False),
            sa.Column("model_type", sa.String(), nullable=False, server_default=""),
            sa.Column("version", sa.String(), nullable=False, server_default=""),
            sa.Column("file_path", sa.String(), nullable=False, server_default=""),
            sa.Column("file_size_mb", sa.Float(), nullable=False, server_default="0.0"),
            sa.Column("metrics_json", sa.String(), nullable=False, server_default="{}"),
            sa.Column("vram_required_gb", sa.Float(), nullable=False, server_default="0.0"),
            sa.Column("inference_speed_ms", sa.Float(), nullable=True),
            sa.Column("is_active", sa.Boolean(), nullable=False, server_default="1"),
            sa.Column("created_at", sa.Float(), nullable=False),
            sa.Column("description", sa.String(), nullable=False, server_default=""),
            sa.PrimaryKeyConstraint("id"),
        )
        op.create_index("ix_model_versions_name", "model_versions", ["name"])

    # ── analysis_sessions ────────────────────────────────────────────────────
    if "analysis_sessions" not in existing_tables:
        op.create_table(
            "analysis_sessions",
            sa.Column("id", sa.Integer(), nullable=False),
            sa.Column("session_uuid", sa.String(), nullable=False),
            sa.Column("input_path", sa.String(), nullable=False, server_default=""),
            sa.Column("num_images", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("num_detections", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("maturity_distribution_json", sa.String(), nullable=False, server_default="{}"),
            sa.Column("morphology_distribution_json", sa.String(), nullable=False, server_default="{}"),
            sa.Column("output_dir", sa.String(), nullable=False, server_default=""),
            sa.Column("duration_s", sa.Float(), nullable=True),
            sa.Column("status", sa.String(), nullable=False, server_default="pending"),
            sa.Column("created_at", sa.Float(), nullable=False),
            sa.PrimaryKeyConstraint("id"),
        )
        op.create_index("ix_analysis_sessions_session_uuid", "analysis_sessions", ["session_uuid"])


def downgrade() -> None:
    """Drop all tables (irreversible — use with caution)."""
    for table in [
        "analysis_sessions",
        "model_versions",
        "jobs",
        "samples",
        "datasets",
        "metrics",
        "runs",
        "experiments",
    ]:
        op.drop_table(table)
