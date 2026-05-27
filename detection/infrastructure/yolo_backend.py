"""
detection.infrastructure.yolo_backend — YOLO detector implementation.

Wraps Ultralytics YOLO (v8/v11) for trichome detection.

Model selection guidance:
- YOLOv11n: Fastest, for real-time video analysis (RTX 3070)
- YOLOv11s: Good balance for interactive annotation assistance
- YOLOv11m: Recommended for dataset inference
- YOLOv11l: Training with full dataset
- YOLOv11x: Maximum accuracy, final evaluation and benchmarks

Small object detection adaptations:
YOLO's standard P3/P4/P5 feature pyramid handles objects down to ~8×8px.
For trichomes smaller than this at target resolution, we use:
1. Higher input resolution (1280 instead of 640)
2. Tiled inference (see TiledInferenceEngine)
3. Additional P2 detection head (if fine-tuning architecture)
4. Smaller anchor sizes in training config

Reference:
  Wang, C.Y. et al. (2024). YOLOv9: Learning What You Want to Learn
  Using Programmable Gradient Information. arXiv:2402.13616.

  Jocher, G. et al. (2024). Ultralytics YOLO11.
  https://github.com/ultralytics/ultralytics
"""

from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray

from detection.domain.detector import BaseDetector, DetectionConfig, DetectionResult
from shared.core.entities import Detection
from shared.core.enums import ModelBackend, TrichomeType
from shared.core.value_objects import BoundingBox, Confidence


# Map YOLO class indices to TrichomeType
# This mapping MUST match the class order in your training data.yaml
YOLO_CLASS_MAP: dict[int, TrichomeType] = {
    0: TrichomeType.CAPITATE_STALKED,
    1: TrichomeType.CAPITATE_SESSILE,
    2: TrichomeType.BULBOUS,
    3: TrichomeType.NON_GLANDULAR,
}


class YOLODetector(BaseDetector):
    """
    YOLO-based trichome detector.

    Supports:
    - PyTorch inference (full feature set)
    - ONNX inference (deployment, slightly faster on CPU)
    - TensorRT inference (fastest GPU inference, requires TRT build)

    Usage:
        detector = YOLODetector(
            model_id="yolo11x-trichome-v2",
            weights_path=Path("/mnt/models/yolo11x_trichome_v2.pt"),
            device="cuda:0",
        )
        detector.load()
        result = detector.detect(image_array)
    """

    def __init__(
        self,
        model_id: str,
        weights_path: Path,
        device: str = "cuda:0",
        backend: ModelBackend = ModelBackend.PYTORCH,
        class_map: dict[int, TrichomeType] | None = None,
    ) -> None:
        super().__init__(model_id, weights_path, device)
        self._backend = backend
        self._class_map = class_map or YOLO_CLASS_MAP
        self._model: Any | None = None  # ultralytics.YOLO instance

    def load(self) -> None:
        """
        Load YOLO model from weights file.

        Memory requirements:
        - YOLOv11n: ~14MB VRAM
        - YOLOv11s: ~40MB VRAM
        - YOLOv11m: ~100MB VRAM
        - YOLOv11l: ~200MB VRAM
        - YOLOv11x: ~360MB VRAM
        """
        import logging

        logger = logging.getLogger(__name__)

        if self._is_loaded:
            logger.warning(f"Model {self._model_id} already loaded, skipping.")
            return

        if not self._weights_path.exists():
            raise FileNotFoundError(
                f"YOLO weights not found: {self._weights_path}\n"
                f"Download pretrained weights or train your own. "
                f"See training/ for training pipeline."
            )

        logger.info(f"Loading YOLO model: {self._model_id} from {self._weights_path}")

        try:
            from ultralytics import YOLO

            self._model = YOLO(str(self._weights_path))
            self._model.to(self._device)

            if self._backend == ModelBackend.PYTORCH and self._device.startswith("cuda"):
                # Compile with torch.compile for ~20% speedup (PyTorch 2.0+)
                # Note: First inference after compile is slow (compilation step)
                # Disable if you get CUDA errors with complex images
                # self._model.model = torch.compile(self._model.model, mode="reduce-overhead")
                pass

            self._is_loaded = True
            logger.info(
                f"✓ YOLO model loaded: {self._model_id} "
                f"(device={self._device}, backend={self._backend.value})"
            )

        except ImportError as e:
            raise ImportError(
                f"Ultralytics not installed. Run: uv pip install ultralytics>=8.2.0"
            ) from e
        except Exception as e:
            raise RuntimeError(
                f"Failed to load YOLO model from {self._weights_path}: {e}"
            ) from e

    def unload(self) -> None:
        """Release model from GPU memory."""
        if self._model is not None:
            del self._model
            self._model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        self._is_loaded = False

    def _run_inference(
        self,
        image: NDArray[np.uint8],
        config: DetectionConfig,
    ) -> tuple[list[Detection], int]:
        """
        Run YOLO inference on preprocessed image.

        Returns:
            Tuple of (post-NMS detections, num pre-NMS detections)
        """
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        # YOLO inference
        results = self._model(
            image,
            imgsz=config.input_size[0],
            conf=config.confidence_threshold,
            iou=config.iou_threshold,
            max_det=config.max_detections,
            half=config.use_fp16 and self._device.startswith("cuda"),
            augment=config.augment,
            verbose=False,
        )

        detections: list[Detection] = []
        num_raw = 0

        for result in results:
            if result.boxes is None:
                continue

            boxes = result.boxes
            num_raw += len(boxes)

            for i in range(len(boxes)):
                # Extract box coordinates (XYXY format)
                xyxy = boxes.xyxy[i].cpu().numpy()
                conf_val = float(boxes.conf[i].cpu().numpy())
                cls_id = int(boxes.cls[i].cpu().numpy())

                # Map class to TrichomeType
                trichome_type = self._class_map.get(cls_id, TrichomeType.UNKNOWN)

                try:
                    bbox = BoundingBox(
                        x_min=float(xyxy[0]),
                        y_min=float(xyxy[1]),
                        x_max=float(xyxy[2]),
                        y_max=float(xyxy[3]),
                    )
                except ValueError:
                    # Skip invalid boxes (edge cases from tiled inference)
                    continue

                detection = Detection(
                    id=str(uuid.uuid4()),
                    bounding_box=bbox,
                    confidence=Confidence(min(conf_val, 1.0)),
                    trichome_type=trichome_type,
                    model_id=self._model_id,
                    class_id=cls_id,
                )
                detections.append(detection)

        return detections, num_raw

    def detect_batch(
        self,
        images: list[NDArray[np.uint8]],
        config: DetectionConfig | None = None,
    ) -> list[DetectionResult]:
        """
        Batch detection — significantly faster than sequential.

        YOLO handles batching internally, padding images to same size.
        Optimal batch size depends on GPU VRAM:
        - RTX 3070 (8GB): batch_size 4-8 at 1280px
        - RTX 4090 (24GB): batch_size 16-32 at 1280px
        """
        if not self._is_loaded:
            self.load()

        cfg = config or self._default_config
        results = []

        # Preprocess all images
        preprocessed = [self._preprocess(img) for img in images]

        # Single YOLO call for all images (true batching)
        t_start = time.perf_counter()
        raw_results = self._model(
            preprocessed,
            imgsz=cfg.input_size[0],
            conf=cfg.confidence_threshold,
            iou=cfg.iou_threshold,
            max_det=cfg.max_detections,
            half=cfg.use_fp16 and self._device.startswith("cuda"),
            verbose=False,
        )
        t_end = time.perf_counter()
        total_ms = (t_end - t_start) * 1000

        for i, (raw_result, image) in enumerate(zip(raw_results, images)):
            detections: list[Detection] = []
            num_raw = 0

            if raw_result.boxes is not None:
                boxes = raw_result.boxes
                num_raw = len(boxes)

                for j in range(len(boxes)):
                    xyxy = boxes.xyxy[j].cpu().numpy()
                    conf_val = float(boxes.conf[j].cpu().numpy())
                    cls_id = int(boxes.cls[j].cpu().numpy())
                    trichome_type = self._class_map.get(cls_id, TrichomeType.UNKNOWN)

                    try:
                        bbox = BoundingBox(
                            x_min=float(xyxy[0]),
                            y_min=float(xyxy[1]),
                            x_max=float(xyxy[2]),
                            y_max=float(xyxy[3]),
                        )
                    except ValueError:
                        continue

                    detection = Detection(
                        id=str(uuid.uuid4()),
                        bounding_box=bbox,
                        confidence=Confidence(min(conf_val, 1.0)),
                        trichome_type=trichome_type,
                        model_id=self._model_id,
                        class_id=cls_id,
                    )
                    detections.append(detection)

            results.append(
                DetectionResult(
                    detections=detections,
                    image_id=f"batch_{i}",
                    model_id=self._model_id,
                    inference_time_ms=total_ms / len(images),
                    image_shape=image.shape,
                    num_raw_detections=num_raw,
                    confidence_threshold_used=cfg.confidence_threshold,
                    iou_threshold_used=cfg.iou_threshold,
                )
            )

        return results

    def export_onnx(self, output_path: Path, input_size: int = 1280) -> Path:
        """
        Export model to ONNX format for deployment.

        ONNX allows:
        - Inference without PyTorch dependency
        - Deployment on ONNX Runtime (CPU, CUDA, TensorRT EP)
        - Cross-platform compatibility
        """
        if not self._is_loaded:
            self.load()

        output_path.parent.mkdir(parents=True, exist_ok=True)
        self._model.export(
            format="onnx",
            imgsz=input_size,
            half=True,
            dynamic=True,  # Dynamic batch size
            simplify=True,  # ONNX simplification for TRT compatibility
        )
        return output_path

    def export_tensorrt(
        self,
        output_path: Path,
        input_size: int = 1280,
        int8: bool = False,
    ) -> Path:
        """
        Export to TensorRT engine for maximum GPU inference speed.

        TRT provides 2-5× speedup over ONNX Runtime on NVIDIA GPUs.
        The .engine file is GPU-specific (cannot be shared between GPU models).

        Requires:
        - TensorRT installed (nvidia-tensorrt package)
        - CUDA toolkit matching TRT version
        """
        if not self._is_loaded:
            self.load()

        output_path.parent.mkdir(parents=True, exist_ok=True)
        self._model.export(
            format="engine",
            imgsz=input_size,
            half=not int8,
            int8=int8,
            device=0,
        )
        return output_path

    def get_model_info(self) -> dict[str, Any]:
        """Return model architecture information."""
        if not self._is_loaded:
            return {"loaded": False, "model_id": self._model_id}

        info: dict[str, Any] = {
            "model_id": self._model_id,
            "loaded": True,
            "device": self._device,
            "backend": self._backend.value,
            "weights_path": str(self._weights_path),
        }

        if self._model is not None:
            try:
                info["num_parameters"] = sum(
                    p.numel() for p in self._model.model.parameters()
                )
                info["num_classes"] = len(self._class_map)
                info["class_map"] = {k: v.value for k, v in self._class_map.items()}
            except Exception:
                pass

        return info
