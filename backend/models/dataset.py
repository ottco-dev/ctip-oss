"""
backend.models.dataset — Dataset, Sample, and Annotation database models.

SCHEMA:
  datasets    — dataset containers (YOLO format directories)
  samples     — individual images within a dataset
  annotations — labels for samples (detection boxes, masks, maturity labels)
"""

import json
import time
from typing import Any, List, Optional

from sqlmodel import Field, SQLModel, Relationship


class Dataset(SQLModel, table=True):
    """A dataset container (collection of samples with annotations)."""

    __tablename__ = "datasets"

    id: Optional[int] = Field(default=None, primary_key=True)
    name: str = Field(index=True)
    description: str = ""

    # Storage
    root_path: str = ""
    """Absolute path to dataset directory (YOLO format)."""

    # Metadata
    num_samples: int = 0
    num_annotated: int = 0
    num_reviewed: int = 0

    split_config_json: str = Field(default='{"train":0.7,"val":0.2,"test":0.1}')
    """Train/val/test split ratios."""

    class_names_json: str = Field(
        default='["capitate_stalked","capitate_sessile","bulbous","non_glandular"]'
    )
    """JSON list of class names."""

    # Version control
    version: str = "0.1.0"
    parent_dataset_id: Optional[int] = Field(default=None, foreign_key="datasets.id")

    # Status
    status: str = "active"
    """active, archived, building"""

    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)

    # Relationships
    samples: List["Sample"] = Relationship(back_populates="dataset")

    def get_class_names(self) -> List[str]:
        return json.loads(self.class_names_json)

    def get_split_config(self) -> dict:
        return json.loads(self.split_config_json)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "root_path": self.root_path,
            "num_samples": self.num_samples,
            "num_annotated": self.num_annotated,
            "num_reviewed": self.num_reviewed,
            "class_names": self.get_class_names(),
            "version": self.version,
            "status": self.status,
            "created_at": self.created_at,
        }


class Sample(SQLModel, table=True):
    """A single image sample within a dataset."""

    __tablename__ = "samples"

    id: Optional[int] = Field(default=None, primary_key=True)
    dataset_id: int = Field(foreign_key="datasets.id", index=True)

    # File
    filename: str = ""
    file_path: str = ""
    """Absolute path to the image file."""

    file_hash: str = ""
    """SHA-256 of the image file (for duplicate detection)."""

    # Image dimensions
    width: int = 0
    height: int = 0
    channels: int = 3

    metadata_json: str = Field(default="{}")
    """JSON with microscope settings, strain, session ID, etc."""

    # Quality
    quality_score: float = 0.0
    """Focus + exposure combined quality score [0,1]."""

    focus_score: float = 0.0
    exposure_score: float = 0.0
    is_usable: bool = True

    # Split assignment
    split: str = "train"
    """train, val, test"""

    # Annotation status
    num_annotations: int = 0
    annotation_source: str = ""
    """human, vlm_auto, auto_converted"""

    created_at: float = Field(default_factory=time.time)

    # Relationships
    dataset: Optional[Dataset] = Relationship(back_populates="samples")
    annotations: List["Annotation"] = Relationship(back_populates="sample")

    def get_metadata(self) -> dict:
        return json.loads(self.metadata_json)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "dataset_id": self.dataset_id,
            "filename": self.filename,
            "file_path": self.file_path,
            "width": self.width,
            "height": self.height,
            "quality_score": self.quality_score,
            "focus_score": self.focus_score,
            "split": self.split,
            "num_annotations": self.num_annotations,
            "annotation_source": self.annotation_source,
            "created_at": self.created_at,
        }


class Annotation(SQLModel, table=True):
    """An annotation (label) for a sample."""

    __tablename__ = "annotations"

    id: Optional[int] = Field(default=None, primary_key=True)
    sample_id: int = Field(foreign_key="samples.id", index=True)

    # Label data (YOLO format or structured JSON)
    data_json: str = Field(default="{}")
    """
    JSON with annotation data. Format depends on annotation_type:
    - 'detection': {"class_id": 0, "x_center": 0.5, "y_center": 0.5, "width": 0.1, "height": 0.1}
    - 'segmentation': {"class_id": 0, "polygon": [[x1,y1], [x2,y2], ...]}
    - 'maturity': {"stage": "cloudy", "amber_fraction": 0.1, ...}
    """

    annotation_type: str = "detection"
    """detection, segmentation, maturity, morphology"""

    class_id: Optional[int] = None
    class_name: str = ""

    # Provenance
    source: str = "human"
    """human, vlm_auto, model_prediction, imported"""

    confidence: Optional[float] = None
    """Prediction confidence (for model/VLM annotations)."""

    # Review status
    reviewed: bool = False
    reviewer_id: Optional[str] = None
    review_action: Optional[str] = None
    """approved, corrected, rejected"""

    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)

    # Relationships
    sample: Optional[Sample] = Relationship(back_populates="annotations")

    def get_data(self) -> dict:
        return json.loads(self.data_json)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "sample_id": self.sample_id,
            "annotation_type": self.annotation_type,
            "class_id": self.class_id,
            "class_name": self.class_name,
            "source": self.source,
            "confidence": self.confidence,
            "reviewed": self.reviewed,
            "data": self.get_data(),
            "created_at": self.created_at,
        }
