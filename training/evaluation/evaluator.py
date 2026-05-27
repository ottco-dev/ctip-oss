"""
training.evaluation.evaluator — Post-training model evaluation pipeline.

Responsibilities:
  1. Run a trained YOLO model against a validation dataset
  2. Collect per-prediction (confidence, is_correct) pairs via IoU matching
  3. Compute ECE / MCE and reliability diagram data
  4. Log calibration artifacts to MLflow:
     - predictions/confidence_scores.npy
     - predictions/is_correct.npy
     - calibration/ece.json (scalar metrics)
     - calibration/reliability_data.json (per-bin data)
  5. Return structured EvaluationResult

This module bridges training (Phase 12) and analytics (Phase 14) by making
calibration artifacts available to the /analytics/calibration/run/{run_id}
endpoint without any manual post-processing.

Scientific basis:
  Guo, C. et al. (2017). On Calibration of Modern Neural Networks.
  ICML 2017. arXiv:1706.04599

  IoU matching strategy follows COCO evaluation protocol:
  Lin, T.-Y. et al. (2014). Microsoft COCO: Common Objects in Context.
  ECCV 2014.
"""

from __future__ import annotations

import json
import logging
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Optional

import numpy as np
from numpy.typing import NDArray

from shared.metrics.calibration_metrics import CalibrationResult, compute_calibration

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# IoU matching
# ---------------------------------------------------------------------------

def _compute_iou(box_a: NDArray, box_b: NDArray) -> float:
    """
    Compute IoU between two [x1, y1, x2, y2] boxes.

    Args:
        box_a: (4,) array — predicted box.
        box_b: (4,) array — ground truth box.

    Returns:
        IoU in [0, 1].
    """
    ix1 = max(box_a[0], box_b[0])
    iy1 = max(box_a[1], box_b[1])
    ix2 = min(box_a[2], box_b[2])
    iy2 = min(box_a[3], box_b[3])

    inter_w = max(0.0, ix2 - ix1)
    inter_h = max(0.0, iy2 - iy1)
    inter = inter_w * inter_h

    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union = area_a + area_b - inter

    return inter / union if union > 0 else 0.0


def _match_detections(
    predictions: list[dict],
    ground_truths: list[dict],
    iou_threshold: float = 0.50,
) -> tuple[list[float], list[bool]]:
    """
    Match predictions to ground truths using greedy IoU matching.

    Each ground truth can only be matched once. Predictions not matched to
    any GT are marked as incorrect (false positives). Unmatched GT boxes
    count as missed detections but do not contribute to calibration pairs
    (we cannot assign a confidence to a missed detection).

    Args:
        predictions: List of dicts {x1, y1, x2, y2, confidence, class_id}.
        ground_truths: List of dicts {x1, y1, x2, y2, class_id}.
        iou_threshold: IoU threshold for a true positive match.

    Returns:
        Tuple (confidences, is_correct) — parallel lists for calibration.
    """
    if not predictions:
        return [], []

    # Sort predictions by confidence descending (greedy best-first match)
    preds_sorted = sorted(predictions, key=lambda p: p["confidence"], reverse=True)
    gt_matched = [False] * len(ground_truths)

    confidences: list[float] = []
    is_correct: list[bool] = []

    for pred in preds_sorted:
        pred_box = np.array([pred["x1"], pred["y1"], pred["x2"], pred["y2"]])
        pred_cls = pred.get("class_id", -1)
        conf = float(pred["confidence"])

        best_iou = 0.0
        best_gt_idx = -1

        for j, gt in enumerate(ground_truths):
            if gt_matched[j]:
                continue
            if gt.get("class_id", -1) != pred_cls:
                continue  # Class mismatch — not a true positive
            gt_box = np.array([gt["x1"], gt["y1"], gt["x2"], gt["y2"]])
            iou = _compute_iou(pred_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = j

        matched = best_iou >= iou_threshold and best_gt_idx >= 0
        if matched:
            gt_matched[best_gt_idx] = True

        confidences.append(conf)
        is_correct.append(matched)

    return confidences, is_correct


# ---------------------------------------------------------------------------
# Evaluation configuration
# ---------------------------------------------------------------------------

@dataclass
class EvaluationConfig:
    """Configuration for the evaluation pipeline."""

    model_path: str
    """Path to trained .pt model or ONNX file."""

    data_yaml: str
    """Path to YOLO dataset YAML with validation split."""

    iou_threshold: float = 0.50
    """IoU threshold for true-positive matching (COCO default)."""

    conf_threshold: float = 0.001
    """
    Confidence threshold during evaluation — intentionally low to capture
    all predictions including low-confidence ones for calibration analysis.
    Note: calibration is NOT threshold-dependent at compute time.
    """

    imgsz: int = 1280
    """Inference image size."""

    device: str = "cuda"
    """'cuda', 'cuda:0', 'cpu'."""

    num_bins: int = 15
    """ECE bins (Guo et al. recommend 15 for calibration analysis)."""

    max_images: Optional[int] = None
    """Cap evaluation at N images. None = full validation set."""

    mlflow_run_id: Optional[str] = None
    """If set, log calibration artifacts to this existing MLflow run."""

    mlflow_tracking_uri: str = "http://localhost:5000"
    """MLflow tracking server URI."""

    save_artifacts_locally: bool = True
    """Save calibration artifacts to disk even if MLflow is unavailable."""

    artifact_output_dir: str = "runs/eval"
    """Local output directory for calibration artifacts."""


# ---------------------------------------------------------------------------
# Evaluation result
# ---------------------------------------------------------------------------

@dataclass
class EvaluationResult:
    """Full evaluation result including detection metrics and calibration."""

    run_id: str
    model_path: str
    data_yaml: str

    # ── Calibration ─────────────────────────────────────────────────────
    calibration: Optional[CalibrationResult] = None
    """ECE / MCE / reliability diagram data."""

    total_predictions: int = 0
    true_positives: int = 0
    false_positives: int = 0
    total_ground_truths: int = 0
    false_negatives: int = 0  # missed detections

    # ── Standard YOLO metrics ────────────────────────────────────────────
    map50: float = 0.0
    map50_95: float = 0.0
    precision: float = 0.0
    recall: float = 0.0

    # ── Timing ──────────────────────────────────────────────────────────
    eval_time_s: float = 0.0
    images_evaluated: int = 0

    # ── Artifact paths ───────────────────────────────────────────────────
    mlflow_run_id: Optional[str] = None
    confidence_scores_path: Optional[str] = None
    is_correct_path: Optional[str] = None
    calibration_json_path: Optional[str] = None

    @property
    def ece(self) -> float:
        return self.calibration.ece if self.calibration else 0.0

    @property
    def mce(self) -> float:
        return self.calibration.mce if self.calibration else 0.0

    @property
    def mean_confidence(self) -> float:
        """Mean predicted confidence (calibration bias indicator)."""
        if self.calibration is None:
            return 0.0
        non_empty = self.calibration.bin_counts > 0
        if not non_empty.any():
            return 0.0
        return float(np.average(
            self.calibration.bin_confidences[non_empty],
            weights=self.calibration.bin_counts[non_empty],
        ))

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "model_path": self.model_path,
            "data_yaml": self.data_yaml,
            "images_evaluated": self.images_evaluated,
            "total_predictions": self.total_predictions,
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "total_ground_truths": self.total_ground_truths,
            "false_negatives": self.false_negatives,
            "map50": round(self.map50, 6),
            "map50_95": round(self.map50_95, 6),
            "precision": round(self.precision, 6),
            "recall": round(self.recall, 6),
            "ece": round(self.ece, 6),
            "mce": round(self.mce, 6),
            "mean_confidence": round(self.mean_confidence, 6),
            "eval_time_s": round(self.eval_time_s, 3),
            "mlflow_run_id": self.mlflow_run_id,
            "calibration": self.calibration.to_dict() if self.calibration else None,
        }


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class ModelEvaluator:
    """
    Post-training model evaluator.

    Runs a trained YOLO model against its validation split, collects
    per-prediction calibration data, computes ECE, and logs artifacts
    to MLflow.

    Usage::

        config = EvaluationConfig(
            model_path="runs/detect/exp/weights/best.pt",
            data_yaml="/data/datasets/trichome/data.yaml",
            mlflow_run_id="a1b2c3d4...",   # from training run
        )
        evaluator = ModelEvaluator(config)
        result = evaluator.evaluate()

        print(f"ECE: {result.ece:.4f}")
        print(f"mAP50: {result.map50:.4f}")
    """

    def __init__(self, config: EvaluationConfig) -> None:
        self.config = config
        self._run_id = config.mlflow_run_id or f"eval_{int(time.time())}"

    def evaluate(self) -> EvaluationResult:
        """
        Run the full evaluation pipeline.

        1. Load model
        2. Run YOLO validation → get YOLO mAP metrics + raw predictions
        3. Match predictions to ground truths via IoU
        4. Compute ECE / MCE
        5. Log artifacts to MLflow

        Returns:
            EvaluationResult with all metrics and artifact paths.

        Raises:
            ImportError: ultralytics not installed.
            FileNotFoundError: model_path or data_yaml not found.
        """
        t0 = time.monotonic()
        result = EvaluationResult(
            run_id=self._run_id,
            model_path=self.config.model_path,
            data_yaml=self.config.data_yaml,
            mlflow_run_id=self.config.mlflow_run_id,
        )

        # ── Step 1: YOLO validation (standard COCO metrics) ──────────────
        logger.info("Starting evaluation: model=%s", self.config.model_path)
        yolo_metrics, predictions_by_image, gts_by_image = self._run_yolo_validation()

        result.map50     = yolo_metrics.get("metrics/mAP50(B)", 0.0)
        result.map50_95  = yolo_metrics.get("metrics/mAP50-95(B)", 0.0)
        result.precision = yolo_metrics.get("metrics/precision(B)", 0.0)
        result.recall    = yolo_metrics.get("metrics/recall(B)", 0.0)

        # ── Step 2: IoU matching → calibration pairs ─────────────────────
        all_confidences: list[float] = []
        all_is_correct: list[bool] = []
        total_gts = 0
        total_preds = 0
        total_tp = 0

        for image_id in predictions_by_image:
            preds = predictions_by_image[image_id]
            gts = gts_by_image.get(image_id, [])

            confs, correct = _match_detections(
                preds, gts, iou_threshold=self.config.iou_threshold
            )
            all_confidences.extend(confs)
            all_is_correct.extend(correct)
            total_gts += len(gts)
            total_preds += len(preds)
            total_tp += sum(correct)

        result.total_predictions = total_preds
        result.true_positives = total_tp
        result.false_positives = total_preds - total_tp
        result.total_ground_truths = total_gts
        result.false_negatives = total_gts - total_tp
        result.images_evaluated = len(predictions_by_image)

        # ── Step 3: ECE / reliability diagram ─────────────────────────────
        if len(all_confidences) >= 2:
            try:
                result.calibration = compute_calibration(
                    all_confidences, all_is_correct, num_bins=self.config.num_bins
                )
                logger.info(
                    "Calibration: ECE=%.4f, MCE=%.4f, n=%d",
                    result.ece,
                    result.mce,
                    len(all_confidences),
                )
            except Exception as exc:
                logger.warning("Calibration computation failed: %s", exc)
        else:
            logger.warning(
                "Too few predictions (%d) for calibration. Need ≥ 2.",
                len(all_confidences),
            )

        # ── Step 4: Save artifacts ────────────────────────────────────────
        conf_arr = np.array(all_confidences, dtype=np.float32)
        correct_arr = np.array(all_is_correct, dtype=bool)

        artifact_paths = self._save_artifacts(result, conf_arr, correct_arr)
        result.confidence_scores_path = artifact_paths.get("confidence_scores")
        result.is_correct_path = artifact_paths.get("is_correct")
        result.calibration_json_path = artifact_paths.get("calibration_json")

        # ── Step 5: Log to MLflow ─────────────────────────────────────────
        self._log_to_mlflow(result, artifact_paths)

        result.eval_time_s = time.monotonic() - t0
        logger.info(
            "Evaluation complete in %.1fs — mAP50=%.4f ECE=%.4f",
            result.eval_time_s,
            result.map50,
            result.ece,
        )
        return result

    # ------------------------------------------------------------------
    # YOLO validation
    # ------------------------------------------------------------------

    def _run_yolo_validation(
        self,
    ) -> tuple[dict[str, float], dict[str, list[dict]], dict[str, list[dict]]]:
        """
        Run YOLO .val() and collect per-image predictions + ground truths.

        Returns:
            (yolo_metrics, predictions_by_image, gts_by_image)
        """
        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise ImportError(
                "ultralytics not installed. Install: pip install ultralytics"
            ) from e

        if not Path(self.config.model_path).exists():
            raise FileNotFoundError(
                f"Model not found: {self.config.model_path}"
            )
        if not Path(self.config.data_yaml).exists():
            raise FileNotFoundError(
                f"Dataset YAML not found: {self.config.data_yaml}"
            )

        model = YOLO(self.config.model_path)

        # Run validation — this gives mAP/precision/recall metrics
        val_results = model.val(
            data=self.config.data_yaml,
            imgsz=self.config.imgsz,
            conf=self.config.conf_threshold,
            iou=self.config.iou_threshold,
            device=self.config.device,
            verbose=False,
            plots=False,
            save=False,
        )

        # Extract scalar metrics from results object
        yolo_metrics = {}
        try:
            box = val_results.box
            yolo_metrics["metrics/mAP50(B)"]    = float(box.map50)
            yolo_metrics["metrics/mAP50-95(B)"] = float(box.map)
            yolo_metrics["metrics/precision(B)"] = float(box.mp)
            yolo_metrics["metrics/recall(B)"]    = float(box.mr)
        except Exception as exc:
            logger.warning("Could not extract YOLO scalar metrics: %s", exc)

        # ── Collect per-image predictions + GT for calibration ────────────
        predictions_by_image: dict[str, list[dict]] = {}
        gts_by_image: dict[str, list[dict]] = {}

        try:
            # Re-run in predict mode to get raw boxes (val() doesn't expose per-image)
            import yaml as _yaml

            with open(self.config.data_yaml) as f:
                ds_cfg = _yaml.safe_load(f)

            val_split = ds_cfg.get("val", "val")
            dataset_root = Path(self.config.data_yaml).parent

            # Handle relative paths in YAML
            if not Path(val_split).is_absolute():
                val_split_path = dataset_root / val_split
            else:
                val_split_path = Path(val_split)

            images = self._collect_images(val_split_path)
            if self.config.max_images:
                images = images[: self.config.max_images]

            logger.info("Collecting per-image predictions for %d images", len(images))

            for img_path in images:
                img_id = str(img_path)
                # Predict
                pred_results = model.predict(
                    str(img_path),
                    imgsz=self.config.imgsz,
                    conf=self.config.conf_threshold,
                    iou=self.config.iou_threshold,
                    device=self.config.device,
                    verbose=False,
                    save=False,
                )
                preds = self._parse_predictions(pred_results[0] if pred_results else None)
                predictions_by_image[img_id] = preds

                # Load ground truth labels
                gts = self._load_ground_truth(img_path)
                gts_by_image[img_id] = gts

        except Exception as exc:
            logger.warning(
                "Per-image collection failed (calibration data unavailable): %s", exc
            )
            # Return empty per-image data — will result in no calibration artifacts

        return yolo_metrics, predictions_by_image, gts_by_image

    @staticmethod
    def _collect_images(path: Path) -> list[Path]:
        """Collect all image files from a directory or file list."""
        extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
        if path.is_dir():
            return sorted(p for p in path.rglob("*") if p.suffix.lower() in extensions)
        elif path.is_file() and path.suffix == ".txt":
            return [Path(line.strip()) for line in path.read_text().splitlines() if line.strip()]
        return []

    @staticmethod
    def _parse_predictions(result: Any) -> list[dict]:
        """Parse Ultralytics result → list of prediction dicts."""
        if result is None or result.boxes is None:
            return []
        preds = []
        boxes = result.boxes
        xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, "cpu") else boxes.xyxy
        confs = boxes.conf.cpu().numpy() if hasattr(boxes.conf, "cpu") else boxes.conf
        cls   = boxes.cls.cpu().numpy()  if hasattr(boxes.cls, "cpu")  else boxes.cls
        for i in range(len(xyxy)):
            preds.append({
                "x1": float(xyxy[i][0]),
                "y1": float(xyxy[i][1]),
                "x2": float(xyxy[i][2]),
                "y2": float(xyxy[i][3]),
                "confidence": float(confs[i]),
                "class_id": int(cls[i]),
            })
        return preds

    @staticmethod
    def _load_ground_truth(img_path: Path) -> list[dict]:
        """
        Load YOLO-format ground truth labels.

        Expects <img_stem>.txt in labels/ directory parallel to images/.
        YOLO format: <class_id> <cx_norm> <cy_norm> <w_norm> <h_norm>
        """
        # labels/ is typically a sibling of the images/ directory
        label_path = (
            img_path.parent.parent / "labels" / img_path.stem
        ).with_suffix(".txt")

        if not label_path.exists():
            # Try direct sibling
            label_path = img_path.with_suffix(".txt")

        if not label_path.exists():
            return []

        gts = []
        try:
            # We need image shape to de-normalise boxes
            import cv2
            img = cv2.imread(str(img_path))
            if img is None:
                return []
            h, w = img.shape[:2]

            for line in label_path.read_text().strip().splitlines():
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cls_id = int(parts[0])
                cx, cy, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                x1 = (cx - bw / 2) * w
                y1 = (cy - bh / 2) * h
                x2 = (cx + bw / 2) * w
                y2 = (cy + bh / 2) * h
                gts.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2, "class_id": cls_id})
        except Exception as exc:
            logger.debug("Failed to parse GT for %s: %s", img_path, exc)

        return gts

    # ------------------------------------------------------------------
    # Artifact saving
    # ------------------------------------------------------------------

    def _save_artifacts(
        self,
        result: EvaluationResult,
        conf_arr: NDArray[np.float32],
        correct_arr: NDArray[np.bool_],
    ) -> dict[str, str]:
        """Save calibration artifacts to a temporary directory."""
        artifact_paths: dict[str, str] = {}

        if not self.config.save_artifacts_locally:
            return artifact_paths

        output_dir = Path(self.config.artifact_output_dir) / self._run_id
        output_dir.mkdir(parents=True, exist_ok=True)

        # confidence_scores.npy
        conf_path = output_dir / "confidence_scores.npy"
        np.save(conf_path, conf_arr)
        artifact_paths["confidence_scores"] = str(conf_path)

        # is_correct.npy
        correct_path = output_dir / "is_correct.npy"
        np.save(correct_path, correct_arr)
        artifact_paths["is_correct"] = str(correct_path)

        # calibration.json (scalar metrics + reliability data)
        if result.calibration is not None:
            cal_data = {
                "ece": round(result.ece, 6),
                "mce": round(result.mce, 6),
                "num_bins": result.calibration.num_bins,
                "is_overconfident": result.calibration.is_overconfident,
                "total_predictions": len(conf_arr),
                "bin_confidences": result.calibration.bin_confidences.tolist(),
                "bin_accuracies": result.calibration.bin_accuracies.tolist(),
                "bin_counts": result.calibration.bin_counts.tolist(),
                "mean_confidence": round(result.mean_confidence, 6),
                "map50": round(result.map50, 6),
                "model_path": result.model_path,
            }
            cal_path = output_dir / "calibration.json"
            cal_path.write_text(json.dumps(cal_data, indent=2))
            artifact_paths["calibration_json"] = str(cal_path)

        logger.info("Artifacts saved to: %s", output_dir)
        return artifact_paths

    # ------------------------------------------------------------------
    # MLflow logging
    # ------------------------------------------------------------------

    def _log_to_mlflow(
        self,
        result: EvaluationResult,
        artifact_paths: dict[str, str],
    ) -> None:
        """Log calibration metrics and artifacts to MLflow."""
        try:
            import mlflow  # type: ignore[import]
        except ImportError:
            logger.debug("MLflow not installed — skipping artifact logging")
            return

        if not self.config.mlflow_run_id:
            logger.debug("No mlflow_run_id — skipping artifact logging")
            return

        try:
            mlflow.set_tracking_uri(self.config.mlflow_tracking_uri)
            client = mlflow.tracking.MlflowClient()

            # Log scalar calibration metrics to the existing training run
            client.log_metric(self.config.mlflow_run_id, "eval/ece", result.ece)
            client.log_metric(self.config.mlflow_run_id, "eval/mce", result.mce)
            client.log_metric(self.config.mlflow_run_id, "eval/map50_val", result.map50)
            client.log_metric(self.config.mlflow_run_id, "eval/map50_95_val", result.map50_95)
            client.log_metric(self.config.mlflow_run_id, "eval/precision_val", result.precision)
            client.log_metric(self.config.mlflow_run_id, "eval/recall_val", result.recall)
            client.log_metric(
                self.config.mlflow_run_id,
                "eval/mean_confidence",
                result.mean_confidence,
            )

            # Log numpy artifacts under predictions/ folder
            if "confidence_scores" in artifact_paths:
                client.log_artifact(
                    self.config.mlflow_run_id,
                    artifact_paths["confidence_scores"],
                    artifact_path="predictions",
                )
            if "is_correct" in artifact_paths:
                client.log_artifact(
                    self.config.mlflow_run_id,
                    artifact_paths["is_correct"],
                    artifact_path="predictions",
                )
            if "calibration_json" in artifact_paths:
                client.log_artifact(
                    self.config.mlflow_run_id,
                    artifact_paths["calibration_json"],
                    artifact_path="calibration",
                )

            logger.info(
                "Logged calibration artifacts to MLflow run %s (ECE=%.4f)",
                self.config.mlflow_run_id,
                result.ece,
            )
        except Exception as exc:
            logger.warning("MLflow artifact logging failed: %s", exc)
