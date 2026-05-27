"""backend.models.session — Analysis session database model."""

import json
import time
from typing import Optional

from sqlmodel import Field, SQLModel


class AnalysisSession(SQLModel, table=True):
    """An analysis session (one run of the detection+maturity pipeline)."""

    __tablename__ = "analysis_sessions"

    id: Optional[int] = Field(default=None, primary_key=True)
    session_uuid: str = Field(index=True)

    # Input
    input_path: str = ""
    input_type: str = "image"
    """image, video, directory"""

    # Models used
    detection_model: str = ""
    segmentation_model: str = ""
    maturity_model: str = ""

    # Results summary
    num_trichomes_detected: int = 0
    mean_confidence: float = 0.0
    maturity_distribution_json: str = Field(default="{}")
    morphology_distribution_json: str = Field(default="{}")

    # Output
    output_dir: str = ""
    report_path: Optional[str] = None

    # Status
    status: str = "completed"
    processing_time_s: float = 0.0
    created_at: float = Field(default_factory=time.time)

    def get_maturity_distribution(self) -> dict:
        return json.loads(self.maturity_distribution_json)

    def get_morphology_distribution(self) -> dict:
        return json.loads(self.morphology_distribution_json)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "session_uuid": self.session_uuid,
            "input_path": self.input_path,
            "input_type": self.input_type,
            "num_trichomes_detected": self.num_trichomes_detected,
            "mean_confidence": self.mean_confidence,
            "maturity_distribution": self.get_maturity_distribution(),
            "morphology_distribution": self.get_morphology_distribution(),
            "output_dir": self.output_dir,
            "processing_time_s": self.processing_time_s,
            "status": self.status,
            "created_at": self.created_at,
        }
