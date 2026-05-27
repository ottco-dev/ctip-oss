"""
annotation/sam_assisted/prompter.py — SAM point/box prompts from detections.

Converts bounding box detections into SAM prompts to generate instance masks
for assisted annotation. The annotator reviews and corrects masks rather than
drawing them from scratch — significantly faster workflow.

Pipeline:
  Detection (YOLO) → Box prompts → SAM2 → Instance masks → Review queue
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Prompt types
# ---------------------------------------------------------------------------


@dataclass
class PointPrompt:
    x: float
    y: float
    label: int = 1  # 1=foreground, 0=background


@dataclass
class BoxPrompt:
    x1: float
    y1: float
    x2: float
    y2: float
    class_id: int = 0
    confidence: float = 1.0


@dataclass
class SAMPromptSet:
    """Prompts for one instance (box + optional interior points)."""

    box: BoxPrompt
    interior_points: list[PointPrompt] = field(default_factory=list)
    background_points: list[PointPrompt] = field(default_factory=list)

    def to_sam_box(self) -> np.ndarray:
        """Convert to SAM box format: [x1, y1, x2, y2]."""
        return np.array([self.box.x1, self.box.y1, self.box.x2, self.box.y2])

    def to_sam_points(self) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Convert to SAM point_coords and point_labels arrays."""
        all_points = self.interior_points + self.background_points
        if not all_points:
            return None, None

        coords = np.array([[p.x, p.y] for p in all_points])
        labels = np.array([p.label for p in all_points])
        return coords, labels


# ---------------------------------------------------------------------------
# Prompt generator
# ---------------------------------------------------------------------------


@dataclass
class PrompterConfig:
    """Configuration for SAM-assisted annotation prompter."""

    # Add interior point at box center
    add_center_point: bool = True

    # Add background points at box corners (outside object)
    add_corner_background: bool = False

    # Minimum box size to generate interior point
    min_box_size_px: float = 10.0

    # Confidence threshold: skip low-confidence detections
    min_confidence: float = 0.25

    # Maximum prompts per image (GPU memory limit)
    max_prompts: int = 50


class SAMPrompter:
    """
    Generates SAM prompts from detection results.

    Converts YOLO bounding boxes to SAM box+point prompts for
    instance segmentation. Higher-confidence detections get richer prompts.
    """

    def __init__(self, config: PrompterConfig | None = None) -> None:
        self.config = config or PrompterConfig()

    def from_detections(
        self,
        detections: list[dict],
    ) -> list[SAMPromptSet]:
        """
        Generate SAM prompts from detection results.

        Args:
            detections: List of detection dicts with keys:
                x1, y1, x2, y2, confidence, class_id

        Returns:
            List of SAMPromptSet, one per accepted detection.
        """
        prompts: list[SAMPromptSet] = []

        # Sort by confidence desc, take top-N
        sorted_dets = sorted(
            detections,
            key=lambda d: d.get("confidence", 0.0),
            reverse=True,
        )[: self.config.max_prompts]

        for det in sorted_dets:
            conf = det.get("confidence", 0.0)
            if conf < self.config.min_confidence:
                continue

            x1 = float(det.get("x1", 0))
            y1 = float(det.get("y1", 0))
            x2 = float(det.get("x2", 0))
            y2 = float(det.get("y2", 0))
            class_id = int(det.get("class_id", 0))

            box_w = x2 - x1
            box_h = y2 - y1

            if box_w < 2 or box_h < 2:
                continue

            box = BoxPrompt(x1, y1, x2, y2, class_id=class_id, confidence=conf)
            interior: list[PointPrompt] = []
            background: list[PointPrompt] = []

            # Interior center point
            if self.config.add_center_point and min(box_w, box_h) >= self.config.min_box_size_px:
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                interior.append(PointPrompt(cx, cy, label=1))

            # Background corner points
            if self.config.add_corner_background:
                margin = max(5.0, min(box_w, box_h) * 0.2)
                for bx, by in [
                    (x1 - margin, y1 - margin),
                    (x2 + margin, y1 - margin),
                    (x1 - margin, y2 + margin),
                    (x2 + margin, y2 + margin),
                ]:
                    background.append(PointPrompt(bx, by, label=0))

            prompts.append(
                SAMPromptSet(
                    box=box,
                    interior_points=interior,
                    background_points=background,
                )
            )

        return prompts

    def from_masks(
        self,
        masks: list[np.ndarray],
        image_shape: tuple[int, int],
    ) -> list[SAMPromptSet]:
        """
        Generate refinement prompts from existing rough masks.

        Useful for iterative mask improvement: given a rough mask, find
        interior points and boundary to create better SAM prompts.
        """
        prompts: list[SAMPromptSet] = []
        h, w = image_shape

        for mask in masks:
            mask_u8 = (mask > 0).astype(np.uint8)
            indices = np.where(mask_u8 > 0)
            if len(indices[0]) == 0:
                continue

            # Bounding box from mask
            y_min, y_max = int(indices[0].min()), int(indices[0].max())
            x_min, x_max = int(indices[1].min()), int(indices[1].max())

            box = BoxPrompt(float(x_min), float(y_min), float(x_max), float(y_max))

            # Center of mass as interior point
            cy = float(indices[0].mean())
            cx = float(indices[1].mean())

            prompts.append(
                SAMPromptSet(
                    box=box,
                    interior_points=[PointPrompt(cx, cy, label=1)],
                )
            )

        return prompts


# ---------------------------------------------------------------------------
# Batch prompting (for CVAT/Label Studio integration)
# ---------------------------------------------------------------------------


def convert_to_annotation_format(
    masks: list[np.ndarray],
    classes: list[int],
    image_id: int,
) -> list[dict]:
    """
    Convert SAM masks to annotation format suitable for CVAT/LS import.

    Returns COCO-format annotation dicts.
    """
    from segmentation.domain.polygon_utils import mask_to_coco_segmentation, polygon_area

    annotations = []
    for ann_id, (mask, class_id) in enumerate(zip(masks, classes)):
        segmentation = mask_to_coco_segmentation(mask, simplify_epsilon=2.0)
        if not segmentation:
            continue

        # Bounding box from mask
        rows, cols = np.where(mask > 0)
        if len(rows) == 0:
            continue

        x1, y1 = int(cols.min()), int(rows.min())
        x2, y2 = int(cols.max()), int(rows.max())
        bbox = [x1, y1, x2 - x1, y2 - y1]  # COCO format: [x, y, w, h]
        area = int((mask > 0).sum())

        annotations.append(
            {
                "id": ann_id,
                "image_id": image_id,
                "category_id": class_id,
                "segmentation": segmentation,
                "bbox": bbox,
                "area": area,
                "iscrowd": 0,
            }
        )

    return annotations
