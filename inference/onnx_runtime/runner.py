"""
inference/onnx_runtime/runner.py — ONNX Runtime inference runner.

Runs exported YOLO ONNX models via onnxruntime-gpu / onnxruntime-cpu.
Preferred over PyTorch for production inference when GPU is not available
or for cross-platform deployment without CUDA dependencies.

ONNX export from YOLO:
    from ultralytics import YOLO
    model = YOLO("best.pt")
    model.export(format="onnx", imgsz=1280, half=True, simplify=True)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class ONNXRunnerConfig:
    """Configuration for ONNX Runtime inference."""

    model_path: str = ""
    imgsz: int = 1280
    conf_threshold: float = 0.25
    iou_threshold: float = 0.45
    # Execution provider preference order
    providers: list[str] = field(
        default_factory=lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"]
    )
    # ONNX session options
    intra_op_num_threads: int = 0  # 0 = use all cores
    inter_op_num_threads: int = 0
    graph_optimization_level: str = "ORT_ENABLE_ALL"

    # Output format
    input_name: str = "images"  # YOLO default
    output_names: list[str] = field(default_factory=lambda: ["output0"])


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class ONNXDetection:
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int
    class_name: str


@dataclass
class ONNXResult:
    detections: list[ONNXDetection]
    inference_ms: float
    preprocess_ms: float
    postprocess_ms: float
    model_path: str
    input_shape: tuple[int, ...]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

TRICHOME_CLASSES = {0: "capitate-stalked", 1: "capitate-sessile", 2: "bulbous", 3: "non-glandular"}


class ONNXRuntimeRunner:
    """
    ONNX Runtime inference runner for YOLO trichome detection.

    Handles preprocessing (letterbox resize), inference, and
    NMS postprocessing from YOLO's raw output format.

    YOLO ONNX output shape: (1, num_boxes, 4 + num_classes) or
    transposed (1, 4 + num_classes, num_boxes) depending on export version.
    """

    def __init__(self, config: ONNXRunnerConfig) -> None:
        self.config = config
        self._session = None
        self._input_shape: Optional[tuple] = None

    def load(self) -> None:
        """Load ONNX model and create inference session."""
        try:
            import onnxruntime as ort
        except ImportError as e:
            raise ImportError(
                "onnxruntime not installed. Run: pip install onnxruntime-gpu"
            ) from e

        model_path = self.config.model_path
        if not Path(model_path).exists():
            raise FileNotFoundError(f"ONNX model not found: {model_path}")

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = self.config.intra_op_num_threads
        opts.inter_op_num_threads = self.config.inter_op_num_threads

        opt_level_map = {
            "ORT_DISABLE_ALL": ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
            "ORT_ENABLE_BASIC": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
            "ORT_ENABLE_EXTENDED": ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
            "ORT_ENABLE_ALL": ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
        }
        opts.graph_optimization_level = opt_level_map.get(
            self.config.graph_optimization_level,
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
        )

        self._session = ort.InferenceSession(
            model_path,
            sess_options=opts,
            providers=self.config.providers,
        )

        # Get model input shape
        input_meta = self._session.get_inputs()[0]
        self._input_shape = tuple(input_meta.shape)

        active_provider = self._session.get_providers()[0]
        print(f"ONNX Runtime loaded: {Path(model_path).name} | Provider: {active_provider}")

    def unload(self) -> None:
        self._session = None

    def infer(self, image: np.ndarray) -> ONNXResult:
        """
        Run inference on a single BGR image (as returned by cv2.imread).

        Args:
            image: numpy array, shape (H, W, 3), dtype uint8, BGR.

        Returns:
            ONNXResult with detections in original image coordinates.
        """
        if self._session is None:
            self.load()

        orig_h, orig_w = image.shape[:2]
        target_size = self.config.imgsz

        # 1. Preprocess
        t0 = time.perf_counter()
        blob, (pad_x, pad_y, scale) = self._letterbox(image, target_size)
        t1 = time.perf_counter()
        preprocess_ms = (t1 - t0) * 1000

        # 2. Inference
        outputs = self._session.run(
            self.config.output_names,
            {self.config.input_name: blob},
        )
        t2 = time.perf_counter()
        inference_ms = (t2 - t1) * 1000

        # 3. Postprocess
        raw = outputs[0]
        detections = self._postprocess(
            raw, scale, pad_x, pad_y, orig_w, orig_h
        )
        t3 = time.perf_counter()
        postprocess_ms = (t3 - t2) * 1000

        return ONNXResult(
            detections=detections,
            inference_ms=round(inference_ms, 2),
            preprocess_ms=round(preprocess_ms, 2),
            postprocess_ms=round(postprocess_ms, 2),
            model_path=self.config.model_path,
            input_shape=self._input_shape or (),
        )

    def infer_batch(self, images: list[np.ndarray]) -> list[ONNXResult]:
        """Run inference on a list of images."""
        return [self.infer(img) for img in images]

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    def _letterbox(
        self,
        image: np.ndarray,
        target: int = 1280,
        color: tuple = (114, 114, 114),
    ) -> tuple[np.ndarray, tuple[float, float, float]]:
        """
        Resize image with letterboxing to maintain aspect ratio.

        Returns:
            (blob, (pad_x, pad_y, scale)) where blob is (1, 3, H, W) float32.
        """
        h, w = image.shape[:2]
        scale = min(target / h, target / w)
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))

        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Pad to square
        pad_x = (target - new_w) / 2
        pad_y = (target - new_h) / 2

        top = int(round(pad_y - 0.1))
        bottom = int(round(pad_y + 0.1))
        left = int(round(pad_x - 0.1))
        right = int(round(pad_x + 0.1))
        bottom = target - new_h - top
        right = target - new_w - left

        padded = cv2.copyMakeBorder(
            resized, top, bottom, left, right,
            cv2.BORDER_CONSTANT, value=color
        )

        # BGR → RGB → HWC → CHW → float32 → batch
        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        blob = rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
        blob = np.expand_dims(blob, axis=0)

        return blob, (float(left), float(top), scale)

    # ------------------------------------------------------------------
    # Postprocessing
    # ------------------------------------------------------------------

    def _postprocess(
        self,
        output: np.ndarray,
        scale: float,
        pad_x: float,
        pad_y: float,
        orig_w: int,
        orig_h: int,
    ) -> list[ONNXDetection]:
        """
        Parse YOLO ONNX raw output → filtered detections in original coords.

        YOLO ONNX output (v8/v11):
          Shape (1, 4+nc, N) or (1, N, 4+nc)
        """
        raw = output[0]  # Remove batch dim

        # Transpose if needed: ensure shape (N, 4+nc)
        if raw.ndim == 2 and raw.shape[0] < raw.shape[1]:
            raw = raw.T  # (4+nc, N) → (N, 4+nc)

        num_classes = raw.shape[1] - 4
        boxes_xywh = raw[:, :4]
        class_scores = raw[:, 4:]

        # Max class confidence and class id
        class_ids = class_scores.argmax(axis=1)
        confidences = class_scores[np.arange(len(class_scores)), class_ids]

        # Filter by confidence
        mask = confidences >= self.config.conf_threshold
        if not mask.any():
            return []

        boxes_xywh = boxes_xywh[mask]
        confidences = confidences[mask]
        class_ids = class_ids[mask]

        # cx,cy,w,h → x1,y1,x2,y2 (in letterbox space)
        cx, cy, bw, bh = boxes_xywh[:, 0], boxes_xywh[:, 1], boxes_xywh[:, 2], boxes_xywh[:, 3]
        x1 = (cx - bw / 2 - pad_x) / scale
        y1 = (cy - bh / 2 - pad_y) / scale
        x2 = (cx + bw / 2 - pad_x) / scale
        y2 = (cy + bh / 2 - pad_y) / scale

        # Clip to image bounds
        x1 = np.clip(x1, 0, orig_w)
        y1 = np.clip(y1, 0, orig_h)
        x2 = np.clip(x2, 0, orig_w)
        y2 = np.clip(y2, 0, orig_h)

        # NMS
        boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)
        indices = cv2.dnn.NMSBoxes(
            boxes_xyxy.tolist(),
            confidences.tolist(),
            self.config.conf_threshold,
            self.config.iou_threshold,
        )
        if len(indices) == 0:
            return []

        indices = indices.flatten()
        detections = []
        for i in indices:
            cls_id = int(class_ids[i])
            detections.append(
                ONNXDetection(
                    x1=round(float(x1[i]), 2),
                    y1=round(float(y1[i]), 2),
                    x2=round(float(x2[i]), 2),
                    y2=round(float(y2[i]), 2),
                    confidence=round(float(confidences[i]), 4),
                    class_id=cls_id,
                    class_name=TRICHOME_CLASSES.get(cls_id, f"class_{cls_id}"),
                )
            )

        return detections
