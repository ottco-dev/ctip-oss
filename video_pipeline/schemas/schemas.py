"""video_pipeline.schemas.schemas — Pydantic schemas for video pipeline API."""

from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class VideoInfoSchema(BaseModel):
    path: str
    total_frames: int
    fps: float
    width: int
    height: int
    duration_s: float
    codec: str = ""
    is_4k: bool = False
    is_hd: bool = False


class FrameQualitySchema(BaseModel):
    composite: float = Field(ge=0, le=1)
    focus: float = Field(ge=0, le=1)
    exposure: float = Field(ge=0, le=1)
    noise: float = Field(ge=0, le=1)
    quality_label: str
    is_usable: bool


class RankedFrameSchema(BaseModel):
    frame_index: int
    timestamp_s: float
    quality: FrameQualitySchema
    output_path: Optional[str] = None


class VideoAnalysisRequest(BaseModel):
    video_path: str
    n_best_frames: int = Field(default=10, ge=1, le=200)
    strategy: str = Field(
        default="adaptive",
        description="Frame selection strategy: 'top_n', 'diverse', 'adaptive'",
    )
    min_focus_score: float = Field(default=0.25, ge=0, le=1)
    every_n_frames: int = Field(default=5, ge=1)
    max_dimension: Optional[int] = Field(default=1920, ge=64)
    output_dir: Optional[str] = None
    compute_motion: bool = False


class VideoAnalysisResponse(BaseModel):
    video_info: VideoInfoSchema
    total_frames_analyzed: int
    n_selected: int
    selected_frames: List[RankedFrameSchema]
    motion_summary: Optional[Dict] = None
    processing_time_s: float = 0.0
