"""
backend.models.experiment — Experiment and training run database models.

SCHEMA:
  experiments — top-level container (e.g., "trichome-detection-v1")
  runs        — individual training runs within an experiment
  metrics     — per-epoch metrics for each run
"""

import json
import time
from typing import Any, List, Optional

from sqlmodel import Field, SQLModel, Relationship


class Experiment(SQLModel, table=True):
    """An ML experiment container (grouping of related training runs)."""

    __tablename__ = "experiments"

    id: Optional[int] = Field(default=None, primary_key=True)
    name: str = Field(index=True)
    description: str = ""
    tags: str = Field(default="[]")
    """JSON list of tags for filtering."""

    config_json: str = Field(default="{}")
    """JSON-serialized experiment config."""

    status: str = Field(default="active")
    """active, archived"""

    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)

    # Relationships
    runs: List["Run"] = Relationship(back_populates="experiment")

    def get_config(self) -> dict:
        return json.loads(self.config_json)

    def get_tags(self) -> list:
        return json.loads(self.tags)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "tags": self.get_tags(),
            "status": self.status,
            "created_at": self.created_at,
        }


class Run(SQLModel, table=True):
    """A single training run within an experiment."""

    __tablename__ = "runs"

    id: Optional[int] = Field(default=None, primary_key=True)
    run_uuid: str = Field(index=True)
    """UUID for external reference (not auto-increment)."""

    experiment_id: int = Field(foreign_key="experiments.id", index=True)

    # Config
    model_variant: str = ""
    """e.g. yolo11s"""

    config_json: str = Field(default="{}")
    """Full training config JSON."""

    # Status
    status: str = Field(default="pending")
    """pending, running, completed, failed, stopped"""

    # Timing
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    created_at: float = Field(default_factory=time.time)

    # Best metrics (cached for fast queries)
    best_map50: float = 0.0
    best_map50_95: float = 0.0
    best_precision: float = 0.0
    best_recall: float = 0.0
    best_epoch: int = 0
    total_epochs: int = 0

    # Output paths
    best_model_path: str = ""
    run_dir: str = ""

    # External tracking
    mlflow_run_id: Optional[str] = None
    wandb_run_id: Optional[str] = None

    # Relationships
    experiment: Optional[Experiment] = Relationship(back_populates="runs")
    metrics: List["Metric"] = Relationship(back_populates="run")

    def get_duration_s(self) -> Optional[float]:
        if self.started_at and self.finished_at:
            return self.finished_at - self.started_at
        return None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "run_uuid": self.run_uuid,
            "experiment_id": self.experiment_id,
            "model_variant": self.model_variant,
            "status": self.status,
            "best_map50": self.best_map50,
            "best_map50_95": self.best_map50_95,
            "best_precision": self.best_precision,
            "best_recall": self.best_recall,
            "best_epoch": self.best_epoch,
            "total_epochs": self.total_epochs,
            "best_model_path": self.best_model_path,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_s": self.get_duration_s(),
            "mlflow_run_id": self.mlflow_run_id,
        }


class Metric(SQLModel, table=True):
    """Per-epoch training metric."""

    __tablename__ = "metrics"

    id: Optional[int] = Field(default=None, primary_key=True)
    run_id: int = Field(foreign_key="runs.id", index=True)

    epoch: int
    step: Optional[int] = None
    key: str = Field(index=True)
    """Metric name: train_loss, val_map50, precision, recall, etc."""
    value: float

    created_at: float = Field(default_factory=time.time)

    # Relationships
    run: Optional[Run] = Relationship(back_populates="metrics")
