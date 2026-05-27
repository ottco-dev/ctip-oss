"""
inference/tensorrt_engine/runner.py — TensorRT 10.x inference runner.

Runs YOLO models exported to TensorRT .engine format for maximum throughput
on NVIDIA GPUs. Fully compatible with TensorRT 10.x API (get_tensor_name /
get_tensor_shape / get_tensor_dtype / execute_async_v3).

TRT export from YOLO:
    from ultralytics import YOLO
    model = YOLO("best.pt")
    model.export(format="engine", imgsz=1280, half=True, device=0)

Or use the bundled engine builder:
    from inference.tensorrt_engine.builder import build_engine_from_onnx
    build_engine_from_onnx("model.onnx", "model.engine", fp16=True)

This module is OPTIONAL — the system falls back to ONNX Runtime if TensorRT
is not available. Check availability via tensorrt_available().

Performance on RTX 4060 (expected):
  YOLO11s, 1280px, FP16: ~4-6 ms/image (≈2× speedup over ONNX-CPU)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Availability guard
# ---------------------------------------------------------------------------

def tensorrt_available() -> bool:
    """Return True if TensorRT + pycuda Python bindings are importable."""
    try:
        import tensorrt  # noqa: F401
        import pycuda.driver  # noqa: F401
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class TRTRunnerConfig:
    """Configuration for TensorRT engine inference."""
    engine_path: str = ""
    imgsz: int = 1280
    conf_threshold: float = 0.25
    iou_threshold: float = 0.45
    device_index: int = 0
    warmup_runs: int = 3
    # FP16 mode (engine must have been built with fp16=True)
    fp16: bool = True


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class TRTDetection:
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int
    class_name: str


@dataclass
class TRTResult:
    detections: list[TRTDetection]
    inference_ms: float
    preprocess_ms: float
    postprocess_ms: float
    engine_path: str
    image_hw: tuple[int, int]


# ---------------------------------------------------------------------------
# Class map
# ---------------------------------------------------------------------------

TRICHOME_CLASSES: dict[int, str] = {
    0: "capitate-stalked",
    1: "capitate-sessile",
    2: "bulbous",
    3: "non-glandular",
}


# ---------------------------------------------------------------------------
# TensorRT runner (TRT 10.x API)
# ---------------------------------------------------------------------------

class TensorRTRunner:
    """
    TensorRT 10.x inference runner for YOLO trichome detection.

    Lifecycle:
        runner = TensorRTRunner(TRTRunnerConfig(engine_path="model.engine"))
        runner.load()
        result = runner.infer(bgr_image)
        runner.unload()

    Memory ownership:
        - Page-locked host buffers + device memory allocated at load()
        - Freed at unload() / __del__
        - CUDA context created per-runner with pycuda.autoinit OR shared ctx

    TRT 10 API notes:
        - ICudaEngine.num_io_tensors replaces num_bindings
        - ICudaEngine.get_tensor_name(i) replaces get_binding_name(i)
        - ICudaEngine.get_tensor_mode(name) → TensorIOMode.INPUT/OUTPUT
        - ICudaEngine.get_tensor_shape(name) → tuple
        - IExecutionContext.set_tensor_address(name, ptr) replaces bindings[]
        - IExecutionContext.execute_async_v3(stream) replaces execute_async_v2
    """

    def __init__(self, config: TRTRunnerConfig) -> None:
        self.config = config
        self._engine = None
        self._context = None
        self._stream = None
        self._host_inputs: list = []
        self._host_outputs: list = []
        self._device_inputs: list = []
        self._device_outputs: list = []
        self._input_names: list[str] = []
        self._output_names: list[str] = []
        self._output_shapes: list[tuple] = []
        self._loaded = False

    # ------------------------------------------------------------------ load

    def load(self) -> None:
        """Deserialise engine file, allocate CUDA buffers, run warmup."""
        if self._loaded:
            return

        if not tensorrt_available():
            raise ImportError(
                "TensorRT or pycuda not available. "
                "Install via: pip install pycuda tensorrt\n"
                "Or use ONNXRuntimeRunner as fallback."
            )

        import tensorrt as trt
        import pycuda.autoinit  # noqa: F401 — creates default CUDA context
        import pycuda.driver as cuda

        engine_path = Path(self.config.engine_path)
        if not engine_path.exists():
            raise FileNotFoundError(
                f"TensorRT engine not found: {engine_path}\n"
                "Build it first:\n"
                "  from inference.tensorrt_engine.builder import build_engine_from_onnx\n"
                "  build_engine_from_onnx('model.onnx', 'model.engine')"
            )

        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)

        with open(engine_path, "rb") as fh:
            engine_bytes = fh.read()

        self._engine = runtime.deserialize_cuda_engine(engine_bytes)
        if self._engine is None:
            raise RuntimeError("Failed to deserialise TensorRT engine.")

        self._context = self._engine.create_execution_context()
        self._stream = cuda.Stream()

        self._allocate_buffers_trt10(cuda)

        # Warmup — eliminates first-inference JIT overhead
        dummy = np.zeros(
            (1, 3, self.config.imgsz, self.config.imgsz), dtype=np.float16 if self.config.fp16 else np.float32
        )
        for _ in range(self.config.warmup_runs):
            self._run_engine(dummy)

        self._loaded = True

    # -------------------------------------------------------- unload / __del__

    def unload(self) -> None:
        """Free device memory and destroy execution context."""
        self._loaded = False
        # pycuda device memory freed via GC when references drop
        self._device_inputs.clear()
        self._device_outputs.clear()
        self._host_inputs.clear()
        self._host_outputs.clear()
        self._context = None
        self._engine = None

    def __del__(self):
        try:
            self.unload()
        except Exception:
            pass

    # ----------------------------------------------- buffer allocation (TRT10)

    def _allocate_buffers_trt10(self, cuda) -> None:
        """
        TRT 10.x buffer allocation using get_tensor_name / get_tensor_mode.
        Replaces the deprecated num_bindings / binding_is_input pattern.
        """
        import tensorrt as trt

        n = self._engine.num_io_tensors
        dtype_map = {
            trt.DataType.FLOAT: np.float32,
            trt.DataType.HALF:  np.float16,
            trt.DataType.INT32: np.int32,
            trt.DataType.INT8:  np.int8,
            trt.DataType.BOOL:  np.bool_,
        }

        for i in range(n):
            name = self._engine.get_tensor_name(i)
            mode = self._engine.get_tensor_mode(name)     # INPUT or OUTPUT
            shape = tuple(self._engine.get_tensor_shape(name))
            trt_dtype = self._engine.get_tensor_dtype(name)
            np_dtype = dtype_map.get(trt_dtype, np.float32)

            # Replace dynamic dims (-1) with concrete size
            shape = tuple(
                self.config.imgsz if d == -1 else d for d in shape
            )
            # Batch dim
            shape = tuple(1 if d == 0 else d for d in shape)
            size = int(np.prod(shape))

            host_mem = cuda.pagelocked_empty(size, np_dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)

            if mode == trt.TensorIOMode.INPUT:
                self._input_names.append(name)
                self._host_inputs.append(host_mem)
                self._device_inputs.append(device_mem)
                # Bind tensor address to context
                self._context.set_tensor_address(name, int(device_mem))
            else:
                self._output_names.append(name)
                self._host_outputs.append(host_mem)
                self._device_outputs.append(device_mem)
                self._output_shapes.append(shape)
                self._context.set_tensor_address(name, int(device_mem))

    # ---------------------------------------------------------------- engine run

    def _run_engine(self, blob: np.ndarray) -> list[np.ndarray]:
        """
        Execute TRT engine using TRT 10.x execute_async_v3.
        blob: float32 or float16 array, shape (1, 3, H, W).
        """
        import pycuda.driver as cuda

        # Copy input to page-locked host → device
        flat = blob.ravel()
        if flat.dtype != self._host_inputs[0].dtype:
            flat = flat.astype(self._host_inputs[0].dtype)
        np.copyto(self._host_inputs[0], flat)
        cuda.memcpy_htod_async(self._device_inputs[0], self._host_inputs[0], self._stream)

        # TRT 10.x: execute_async_v3 takes only stream handle
        self._context.execute_async_v3(self._stream.handle)

        # Copy outputs device → host
        outputs = []
        for host_out, dev_out, shape in zip(
            self._host_outputs, self._device_outputs, self._output_shapes
        ):
            cuda.memcpy_dtoh_async(host_out, dev_out, self._stream)
        self._stream.synchronize()

        for host_out, shape in zip(self._host_outputs, self._output_shapes):
            outputs.append(host_out.reshape(shape).copy())

        return outputs

    # ------------------------------------------------------------------ infer

    def infer(self, image: np.ndarray) -> TRTResult:
        """
        Run inference on a single BGR image (HWC).

        Args:
            image: uint8 numpy array, shape (H, W, 3), BGR colour order.

        Returns:
            TRTResult with detections in original image coordinates.
        """
        if not self._loaded:
            self.load()

        import cv2

        orig_h, orig_w = image.shape[:2]
        target = self.config.imgsz

        # -- Preprocess -------------------------------------------------
        t_pre = time.perf_counter()
        scale = min(target / orig_h, target / orig_w)
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)
        pad_x = (target - new_w) // 2
        pad_y = (target - new_h) // 2

        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((target, target, 3), 114, dtype=np.uint8)
        canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized

        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        blob = rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
        blob = np.ascontiguousarray(np.expand_dims(blob, axis=0))
        pre_ms = (time.perf_counter() - t_pre) * 1000

        # -- Engine inference -------------------------------------------
        t_inf = time.perf_counter()
        outputs = self._run_engine(blob)
        inf_ms = (time.perf_counter() - t_inf) * 1000

        # -- Postprocess ------------------------------------------------
        t_post = time.perf_counter()
        detections = self._postprocess(outputs[0], scale, pad_x, pad_y, orig_w, orig_h)
        post_ms = (time.perf_counter() - t_post) * 1000

        return TRTResult(
            detections=detections,
            inference_ms=round(inf_ms, 2),
            preprocess_ms=round(pre_ms, 2),
            postprocess_ms=round(post_ms, 2),
            engine_path=self.config.engine_path,
            image_hw=(orig_h, orig_w),
        )

    # -------------------------------------------------------------- postprocess

    def _postprocess(
        self,
        output: np.ndarray,
        scale: float,
        pad_x: int,
        pad_y: int,
        orig_w: int,
        orig_h: int,
    ) -> list[TRTDetection]:
        """
        Convert raw TRT output (same format as ONNX YOLO export) to TRTDetection list.
        Expected shape: (1, 4+num_classes, num_anchors) → transposed to (num_anchors, 4+C).
        """
        import cv2

        # Handle batch dim
        if output.ndim == 3:
            raw = output[0]
        else:
            raw = output

        # Guard: empty output
        if raw.size == 0 or raw.ndim < 2:
            return []

        # YOLO11 raw output is (4+C, N) where N >> 4+C (e.g. 8 vs 8400).
        # Transpose only when dim-0 is clearly the feature dim (much smaller than dim-1).
        # Threshold 100: if shape[0] < 100 AND shape[0] < shape[1], it's likely (4+C, N).
        if raw.ndim == 2 and raw.shape[0] < raw.shape[1] and raw.shape[0] < 100:
            raw = raw.T

        if raw.shape[0] == 0 or raw.shape[1] < 5:
            return []

        class_scores = raw[:, 4:].astype(np.float32)
        class_ids = class_scores.argmax(axis=1)
        confidences = class_scores[np.arange(len(class_scores)), class_ids]

        mask = confidences >= self.config.conf_threshold
        if not mask.any():
            return []

        raw_f = raw[mask]
        confidences = confidences[mask]
        class_ids = class_ids[mask]

        # cx, cy, bw, bh → x1, y1, x2, y2 in original coords
        cx = raw_f[:, 0].astype(np.float32)
        cy = raw_f[:, 1].astype(np.float32)
        bw = raw_f[:, 2].astype(np.float32)
        bh = raw_f[:, 3].astype(np.float32)

        x1 = np.clip((cx - bw / 2 - pad_x) / scale, 0, orig_w)
        y1 = np.clip((cy - bh / 2 - pad_y) / scale, 0, orig_h)
        x2 = np.clip((cx + bw / 2 - pad_x) / scale, 0, orig_w)
        y2 = np.clip((cy + bh / 2 - pad_y) / scale, 0, orig_h)

        # NMS
        boxes_xywh = np.stack([x1, y1, x2 - x1, y2 - y1], axis=1).tolist()
        indices = cv2.dnn.NMSBoxes(
            boxes_xywh,
            confidences.tolist(),
            self.config.conf_threshold,
            self.config.iou_threshold,
        )
        if len(indices) == 0:
            return []

        indices = np.asarray(indices).flatten()
        return [
            TRTDetection(
                x1=round(float(x1[i]), 2),
                y1=round(float(y1[i]), 2),
                x2=round(float(x2[i]), 2),
                y2=round(float(y2[i]), 2),
                confidence=round(float(confidences[i]), 4),
                class_id=int(class_ids[i]),
                class_name=TRICHOME_CLASSES.get(
                    int(class_ids[i]), f"class_{int(class_ids[i])}"
                ),
            )
            for i in indices
        ]

    # ---------------------------------------------------------------- context

    def __enter__(self) -> "TensorRTRunner":
        self.load()
        return self

    def __exit__(self, *_) -> None:
        self.unload()

    # ---------------------------------------------------------------- repr

    def __repr__(self) -> str:
        status = "loaded" if self._loaded else "not loaded"
        return (
            f"TensorRTRunner(engine={self.config.engine_path!r}, "
            f"imgsz={self.config.imgsz}, fp16={self.config.fp16}, "
            f"status={status})"
        )
