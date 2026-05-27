"""
shared.core — Core domain types, entities, and value objects.

This package defines the fundamental domain primitives used across
the entire Trichome Analysis System. All other services depend on
these types; they must remain stable, well-typed, and dependency-free
(no ML framework imports at this level).
"""

from shared.core.entities import (
    Detection,
    Instance,
    MaturityLabel,
    MorphologyType,
    TrichomeRegion,
)
from shared.core.value_objects import (
    BoundingBox,
    Confidence,
    ImageDimensions,
    Mask,
    Micrometer,
    Pixel,
    PolygonPoints,
)
from shared.core.enums import (
    MaturityStage,
    TrichomeType,
    ImageQuality,
    AnnotationSource,
    ModelBackend,
)

__all__ = [
    # Entities
    "Detection",
    "Instance",
    "MaturityLabel",
    "MorphologyType",
    "TrichomeRegion",
    # Value Objects
    "BoundingBox",
    "Confidence",
    "ImageDimensions",
    "Mask",
    "Micrometer",
    "Pixel",
    "PolygonPoints",
    # Enums
    "MaturityStage",
    "TrichomeType",
    "ImageQuality",
    "AnnotationSource",
    "ModelBackend",
]
