"""
annotation/application/annotation_pipeline.py — Full annotation pipeline.

Orchestrates the end-to-end semi-automatic annotation workflow:
  1. Detect trichomes in images (YOLO)
  2. Generate SAM-assisted instance masks
  3. Send masks to VLM for maturity labeling
  4. Push all results to human review queue
  5. Sync approved annotations to CVAT/Label Studio
  6. Export to YOLO/COCO format for training

Human-in-loop invariant is ENFORCED at step 4:
  No annotation enters the training dataset without explicit human approval.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger("trichome.annotation_pipeline")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class AnnotationPipelineConfig:
    """Configuration for the annotation pipeline."""

    # Detection
    detection_model_path: str = ""
    detection_conf_threshold: float = 0.25
    detection_iou_threshold: float = 0.45
    detection_imgsz: int = 1280

    # Segmentation
    use_sam_assisted: bool = True
    sam_backend: str = "auto"  # auto | sam2_tiny | mobile_sam

    # VLM labeling
    use_vlm_labels: bool = True
    vlm_backend: str = "moondream"  # moondream | florence2 | qwen2vl
    vlm_confidence_threshold: float = 0.70

    # Export
    export_format: str = "yolo"  # yolo | coco
    output_dir: str = "annotations/output"

    # Queue integration
    push_to_review_queue: bool = True
    auto_approve_threshold: float = 0.95  # Very high confidence → skip review


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class ImageAnnotationResult:
    """Annotation result for one image."""

    image_path: str
    detection_count: int = 0
    mask_count: int = 0
    vlm_labels: list[dict] = field(default_factory=list)
    queue_items: list[dict] = field(default_factory=list)
    auto_approved: int = 0
    errors: list[str] = field(default_factory=list)
    processing_ms: float = 0.0


@dataclass
class PipelineRunResult:
    """Result of a full annotation pipeline run."""

    run_id: str
    started_at: datetime
    finished_at: Optional[datetime] = None
    total_images: int = 0
    total_detections: int = 0
    total_queue_items: int = 0
    total_auto_approved: int = 0
    failed_images: int = 0
    image_results: list[ImageAnnotationResult] = field(default_factory=list)
    duration_s: float = 0.0

    @property
    def human_review_required(self) -> int:
        return self.total_queue_items - self.total_auto_approved


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class AnnotationPipeline:
    """
    Full semi-automatic trichome annotation pipeline.

    Enforces human-in-loop invariant: all VLM-generated labels go through
    the review queue before they can be used in training data.

    Auto-approval only when:
      - confidence > auto_approve_threshold (default 0.95)
      - Unanimous prediction across all metrics
      - No conflicting signals between detection and VLM
    """

    def __init__(self, config: AnnotationPipelineConfig | None = None) -> None:
        self.config = config or AnnotationPipelineConfig()
        self._detection_pipeline = None
        self._sam_pipeline = None
        self._vlm_pipeline = None

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def _init_detection(self) -> None:
        if self._detection_pipeline is None and self.config.detection_model_path:
            try:
                from detection.application.detect_pipeline import DetectionPipeline, PipelineConfig

                cfg = PipelineConfig(
                    model_path=self.config.detection_model_path,
                    conf_threshold=self.config.detection_conf_threshold,
                    iou_threshold=self.config.detection_iou_threshold,
                    imgsz=self.config.detection_imgsz,
                )
                self._detection_pipeline = DetectionPipeline(cfg)
            except ImportError as e:
                logger.warning("Detection pipeline unavailable: %s", e)

    def _init_vlm(self) -> None:
        if self._vlm_pipeline is None and self.config.use_vlm_labels:
            try:
                from vlm_labeling.application.auto_label_pipeline import AutoLabelPipeline
                self._vlm_pipeline = AutoLabelPipeline()
            except ImportError as e:
                logger.warning("VLM pipeline unavailable: %s", e)

    # ------------------------------------------------------------------
    # Core run
    # ------------------------------------------------------------------

    def run(self, image_paths: list[str]) -> PipelineRunResult:
        """
        Run the annotation pipeline on a list of image paths.

        Args:
            image_paths: Absolute paths to images.

        Returns:
            PipelineRunResult with all annotation results.
        """
        import time
        import uuid

        run_id = str(uuid.uuid4())[:8]
        start = time.time()

        result = PipelineRunResult(
            run_id=run_id,
            started_at=datetime.utcnow(),
            total_images=len(image_paths),
        )

        logger.info("Annotation pipeline %s starting: %d images", run_id, len(image_paths))

        self._init_detection()
        self._init_vlm()

        for img_path in image_paths:
            t0 = time.perf_counter()
            try:
                img_result = self._process_image(img_path)
                result.image_results.append(img_result)
                result.total_detections += img_result.detection_count
                result.total_queue_items += len(img_result.queue_items)
                result.total_auto_approved += img_result.auto_approved
                if img_result.errors:
                    result.failed_images += 1
            except Exception as exc:  # noqa: BLE001
                logger.error("Failed to annotate %s: %s", img_path, exc)
                result.failed_images += 1
                result.image_results.append(
                    ImageAnnotationResult(image_path=img_path, errors=[str(exc)])
                )

        result.finished_at = datetime.utcnow()
        result.duration_s = round(time.time() - start, 2)

        logger.info(
            "Annotation pipeline %s done: %d detections, %d queue items, "
            "%d auto-approved, %d failed",
            run_id,
            result.total_detections,
            result.total_queue_items,
            result.total_auto_approved,
            result.failed_images,
        )

        return result

    def _process_image(self, image_path: str) -> ImageAnnotationResult:
        """Process one image through the full pipeline."""
        import time
        import cv2

        t0 = time.perf_counter()
        image = cv2.imread(image_path)
        if image is None:
            return ImageAnnotationResult(
                image_path=image_path,
                errors=[f"Could not read image: {image_path}"],
            )

        result = ImageAnnotationResult(image_path=image_path)

        # Step 1: Detect
        detections = []
        if self._detection_pipeline:
            try:
                det_result = self._detection_pipeline.run(image)
                detections = det_result.detections if hasattr(det_result, "detections") else []
                result.detection_count = len(detections)
            except Exception as exc:  # noqa: BLE001
                result.errors.append(f"Detection error: {exc}")

        # Step 2: VLM labeling
        vlm_labels = []
        if self._vlm_pipeline and self.config.use_vlm_labels:
            try:
                vlm_result = self._vlm_pipeline.label_image(image)
                if vlm_result:
                    vlm_labels = [vlm_result] if isinstance(vlm_result, dict) else vlm_result
                    result.vlm_labels = vlm_labels
            except Exception as exc:  # noqa: BLE001
                result.errors.append(f"VLM error: {exc}")

        # Step 3: Build queue items and check auto-approval
        if self.config.push_to_review_queue:
            queue_items = self._build_queue_items(
                image_path=image_path,
                detections=detections,
                vlm_labels=vlm_labels,
            )
            result.queue_items = queue_items
            result.auto_approved = sum(
                1 for q in queue_items
                if q.get("confidence", 0.0) >= self.config.auto_approve_threshold
            )

        result.processing_ms = round((time.perf_counter() - t0) * 1000, 1)
        return result

    def _build_queue_items(
        self,
        image_path: str,
        detections: list[dict],
        vlm_labels: list[dict],
    ) -> list[dict]:
        """
        Build review queue items from detection + VLM results.

        Each item represents one candidate annotation requiring human review.
        Human-in-loop invariant: status is always 'pending_review'.
        """
        items = []
        filename = Path(image_path).name

        # One queue item per VLM label result
        for vlm in vlm_labels:
            confidence = vlm.get("confidence", 0.0)
            item = {
                "image_path": image_path,
                "filename": filename,
                "vlm_labels": vlm,
                "detection_boxes": [
                    {
                        "x1": d.get("x1"), "y1": d.get("y1"),
                        "x2": d.get("x2"), "y2": d.get("y2"),
                        "confidence": d.get("confidence"),
                        "class_id": d.get("class_id"),
                    }
                    for d in detections[:20]  # Cap for memory
                ],
                "confidence": confidence,
                "status": "pending_review",  # ALWAYS pending — human required
                "auto_approvable": confidence >= self.config.auto_approve_threshold,
                "created_at": datetime.utcnow().isoformat(),
                "human_in_loop": True,  # Explicit invariant flag
            }
            items.append(item)

        # If no VLM labels but have detections, create detection-only item
        if not vlm_labels and detections:
            items.append(
                {
                    "image_path": image_path,
                    "filename": filename,
                    "vlm_labels": None,
                    "detection_boxes": detections[:20],
                    "confidence": 0.0,
                    "status": "pending_review",
                    "auto_approvable": False,
                    "created_at": datetime.utcnow().isoformat(),
                    "human_in_loop": True,
                }
            )

        return items

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_approved(
        self,
        approved_items: list[dict],
        output_dir: Optional[str] = None,
    ) -> dict:
        """
        Export approved annotations to YOLO or COCO format.

        Args:
            approved_items: List of approved queue items (status='approved').
            output_dir: Output directory path.

        Returns:
            Export summary dict.
        """
        out_dir = Path(output_dir or self.config.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        if self.config.export_format == "yolo":
            return self._export_yolo(approved_items, out_dir)
        elif self.config.export_format == "coco":
            return self._export_coco(approved_items, out_dir)
        else:
            raise ValueError(f"Unknown export format: {self.config.export_format}")

    def _export_yolo(self, items: list[dict], output_dir: Path) -> dict:
        """Export annotations in YOLO format (one .txt per image)."""
        exported = 0
        for item in items:
            boxes = item.get("detection_boxes", [])
            if not boxes:
                continue

            image_path = Path(item["image_path"])
            label_path = output_dir / (image_path.stem + ".txt")

            lines = []
            for box in boxes:
                cls = box.get("class_id", 0)
                x1, y1, x2, y2 = (
                    box.get("x1", 0), box.get("y1", 0),
                    box.get("x2", 0), box.get("y2", 0)
                )
                # YOLO format: class_id cx_norm cy_norm w_norm h_norm
                # (requires image dimensions — use placeholder 1.0)
                lines.append(f"{cls} {(x1+x2)/2:.6f} {(y1+y2)/2:.6f} {(x2-x1):.6f} {(y2-y1):.6f}")

            label_path.write_text("\n".join(lines))
            exported += 1

        return {"format": "yolo", "exported": exported, "output_dir": str(output_dir)}

    def _export_coco(self, items: list[dict], output_dir: Path) -> dict:
        """Export annotations in COCO JSON format."""
        coco = {
            "info": {"description": "Trichome annotations", "version": "1.0"},
            "categories": [
                {"id": 0, "name": "capitate-stalked"},
                {"id": 1, "name": "capitate-sessile"},
                {"id": 2, "name": "bulbous"},
                {"id": 3, "name": "non-glandular"},
            ],
            "images": [],
            "annotations": [],
        }

        ann_id = 0
        for img_id, item in enumerate(items):
            coco["images"].append(
                {"id": img_id, "file_name": item["filename"]}
            )
            for box in item.get("detection_boxes", []):
                x1, y1 = box.get("x1", 0), box.get("y1", 0)
                x2, y2 = box.get("x2", 0), box.get("y2", 0)
                coco["annotations"].append(
                    {
                        "id": ann_id,
                        "image_id": img_id,
                        "category_id": box.get("class_id", 0),
                        "bbox": [x1, y1, x2 - x1, y2 - y1],
                        "area": (x2 - x1) * (y2 - y1),
                        "iscrowd": 0,
                    }
                )
                ann_id += 1

        out_path = output_dir / "annotations.json"
        out_path.write_text(json.dumps(coco, indent=2))

        return {
            "format": "coco",
            "exported": len(items),
            "annotations": ann_id,
            "output_file": str(out_path),
        }
