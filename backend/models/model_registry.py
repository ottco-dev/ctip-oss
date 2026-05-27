"""backend.models.model_registry — Model registry database models."""

import json
import time
from typing import Optional

from sqlmodel import Field, SQLModel


class RegisteredModel(SQLModel, table=True):
    """A registered ML model (detection, segmentation, maturity, etc.)."""

    __tablename__ = "model_versions"

    id: Optional[int] = Field(default=None, primary_key=True)
    name: str = Field(index=True)

    model_type: str = ""
    """detection, segmentation, maturity, morphology, vlm"""

    framework: str = "pytorch"
    """pytorch, onnx, tensorrt, ultralytics"""

    variant: str = ""
    """e.g. yolo11s, sam2-tiny, moondream-2b"""

    # File
    file_path: str = ""
    file_size_mb: float = 0.0

    # Performance
    metrics_json: str = Field(default="{}")
    vram_required_gb: float = 0.0
    inference_speed_ms: Optional[float] = None

    # Provenance
    training_run_uuid: Optional[str] = None
    base_model: str = ""

    # Status
    is_active: bool = True
    created_at: float = Field(default_factory=time.time)
    description: str = ""

    def get_metrics(self) -> dict:
        return json.loads(self.metrics_json)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "model_type": self.model_type,
            "framework": self.framework,
            "variant": self.variant,
            "file_path": self.file_path,
            "file_size_mb": self.file_size_mb,
            "metrics": self.get_metrics(),
            "vram_required_gb": self.vram_required_gb,
            "inference_speed_ms": self.inference_speed_ms,
            "training_run_uuid": self.training_run_uuid,
            "is_active": self.is_active,
            "created_at": self.created_at,
        }
