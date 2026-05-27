"""
detection — Trichome Detection Service

Provides multi-model detection of trichomes in microscopy images.
Supports YOLO (v8/v11), RTMDet, and ensemble combinations.

Key challenges addressed:
1. Small objects (bulbous trichomes can be 10-15µm = very few pixels)
2. Transparent/translucent structures with low contrast
3. Dense overlapping fields
4. Reflective surfaces under bright-field illumination
5. Motion blur from manual microscope adjustment
6. Chromatic aberration artifacts

Pipeline:
    Image
    → preprocessing (normalize, denoise, enhance contrast)
    → tiled inference (for high-res images)
    → NMS (standard or soft)
    → confidence calibration
    → uncertainty estimation
    → output: List[Detection]
"""

from detection.domain.detector import TrichomeDetector
from detection.domain.ensemble import DetectionEnsemble
from detection.domain.tiled_inference import TiledInferenceEngine
from detection.infrastructure.yolo_backend import YOLODetector
from detection.application.detect_pipeline import DetectionPipeline

__all__ = [
    "TrichomeDetector",
    "DetectionEnsemble",
    "TiledInferenceEngine",
    "YOLODetector",
    "DetectionPipeline",
]
