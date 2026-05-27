"""
backend.models.job — Background job tracking database model.

JOB LIFECYCLE:
  pending → running → completed
                    ↘ failed
                    ↘ cancelled
"""

import json
import time
from enum import Enum
from typing import Any, Optional

from sqlmodel import Field, SQLModel


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class BackgroundJob(SQLModel, table=True):
    """A long-running background task."""

    __tablename__ = "jobs"

    id: Optional[int] = Field(default=None, primary_key=True)
    job_uuid: str = Field(index=True)

    job_type: str = Field(index=True)
    status: str = Field(default="pending", index=True)

    # Progress tracking
    progress: float = 0.0
    total_items: Optional[int] = None
    processed_items: int = 0

    # Configuration & results
    params_json: str = Field(default="{}")
    result_json: str = Field(default="{}")
    error_message: Optional[str] = None

    # Timing
    created_at: float = Field(default_factory=time.time)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None

    # Association
    experiment_id: Optional[int] = Field(default=None, foreign_key="experiments.id")
    dataset_id: Optional[int] = Field(default=None, foreign_key="datasets.id")
    run_uuid: Optional[str] = None

    def get_params(self) -> dict:
        return json.loads(self.params_json)

    def get_result(self) -> dict:
        return json.loads(self.result_json)

    def get_duration_s(self) -> Optional[float]:
        if self.started_at and self.finished_at:
            return self.finished_at - self.started_at
        return None

    def get_progress_pct(self) -> float:
        return self.progress * 100

    def set_result(self, result: dict) -> None:
        self.result_json = json.dumps(result, default=str)

    def set_params(self, params: dict) -> None:
        self.params_json = json.dumps(params, default=str)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "job_uuid": self.job_uuid,
            "job_type": self.job_type,
            "status": self.status,
            "progress": self.progress,
            "progress_pct": self.get_progress_pct(),
            "total_items": self.total_items,
            "processed_items": self.processed_items,
            "error_message": self.error_message,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_s": self.get_duration_s(),
            "run_uuid": self.run_uuid,
        }
