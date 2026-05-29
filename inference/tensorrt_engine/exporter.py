"""
inference/tensorrt_engine/exporter.py — YOLO .pt → ONNX → TensorRT export pipeline.

Full two-stage export:
  1. ultralytics YOLO .export(format="onnx", …) — produces an ONNX file
  2. Optional onnx-simplifier pass (onnxsim)
  3. build_engine_from_onnx() from builder.py — produces a TensorRT .engine

Graceful degradation:
  - ultralytics absent  → ExportError with install instructions
  - TensorRT absent     → stops after ONNX step, logs WARNING
  - onnxsim absent      → skips simplification, continues
  - model .pt not found → clean FileNotFoundError

Designed for RTX 4060 (8 GB VRAM) with FP16 inference at imgsz=1280.

Usage:
    from inference.tensorrt_engine.exporter import YOLOToTensorRT, YOLOExportConfig

    cfg = YOLOExportConfig(
        model_path="models/best.pt",
        output_dir="models/exported/",
        imgsz=1280,
        fp16=True,
        simplify=True,
    )
    exporter = YOLOToTensorRT(cfg)
    result = exporter.export()
    # {"onnx_path": "…/best.onnx", "engine_path": "…/best.engine", "export_time_s": 42.1}
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional dependency handles — imported lazily but exposed at module level
# so unit tests can patch them via `patch("inference.tensorrt_engine.exporter.YOLO")`
# and `patch("inference.tensorrt_engine.exporter.ort")`.
# ---------------------------------------------------------------------------

try:
    from ultralytics import YOLO  # type: ignore[import]
except ImportError:
    YOLO = None  # type: ignore[assignment, misc]

try:
    import onnxruntime as ort  # type: ignore[import]
except ImportError:
    ort = None  # type: ignore[assignment]

try:
    import onnx  # type: ignore[import]
    import onnxsim  # type: ignore[import]
except ImportError:
    onnx = None  # type: ignore[assignment]
    onnxsim = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------

class ExportError(RuntimeError):
    """Raised when the export pipeline cannot proceed."""


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class YOLOExportConfig:
    """
    Parameters controlling the full YOLO → ONNX → TensorRT export.

    Attributes:
        model_path:    Path to the YOLO .pt weights file.
        output_dir:    Directory where .onnx and .engine will be written.
        imgsz:         Square input resolution for export (pixels).  Default 1280
                       matches RTX 4060 tiled-inference target.
        fp16:          Enable FP16 mode in both ONNX export and TRT build.
        opset:         ONNX opset version (17 recommended for TRT 10).
        simplify:      Run onnx-simplifier on the generated ONNX file.
        workspace_gb:  TensorRT builder GPU workspace budget in GB.
        dynamic_batch: If True, enable dynamic batch axis in ONNX export.
                       Keep False for RTX 4060 VRAM safety (static batch=1).
    """
    model_path: str = ""
    output_dir: str = ""
    imgsz: int = 1280
    fp16: bool = True
    opset: int = 17
    simplify: bool = True
    workspace_gb: float = 4.0
    dynamic_batch: bool = False

    def __post_init__(self) -> None:
        if not self.model_path:
            raise ValueError("YOLOExportConfig.model_path must not be empty")
        if not self.output_dir:
            raise ValueError("YOLOExportConfig.output_dir must not be empty")
        if self.imgsz <= 0:
            raise ValueError(f"imgsz must be a positive integer, got {self.imgsz}")
        if not (1 <= self.opset <= 20):
            raise ValueError(f"opset must be between 1 and 20, got {self.opset}")
        if self.workspace_gb <= 0:
            raise ValueError(f"workspace_gb must be positive, got {self.workspace_gb}")


# ---------------------------------------------------------------------------
# Main exporter class
# ---------------------------------------------------------------------------

class YOLOToTensorRT:
    """
    Full export pipeline: YOLO .pt → ONNX → TensorRT .engine.

    Steps
    -----
    1. Validate inputs (model file exists, output dir writable).
    2. Run ``model.export(format="onnx", …)`` via ultralytics.
    3. Optionally simplify the ONNX graph with onnx-simplifier.
    4. Build a TensorRT .engine via
       ``inference.tensorrt_engine.builder.build_engine_from_onnx``.

    Graceful degradation
    --------------------
    - If *ultralytics* is not installed, raises ``ExportError`` immediately
      with pip install instructions.
    - If *TensorRT* is not installed, ``export()`` stops after ONNX and logs
      a WARNING; ``engine_path`` in the result dict will be ``None``.
    - If *onnx-simplifier* is not installed, the simplification step is
      silently skipped and a DEBUG message is logged.
    """

    def __init__(self, config: YOLOExportConfig) -> None:
        self.config = config
        self._output_dir = Path(config.output_dir)

    # ------------------------------------------------------------------ public

    def export(self) -> dict:
        """
        Run the full export pipeline.

        Returns
        -------
        dict with keys:
            - ``onnx_path``     : str — absolute path to the exported ONNX file
            - ``engine_path``   : str | None — path to the TensorRT engine,
                                  or None if TensorRT is not available
            - ``export_time_s`` : float — wall-clock seconds for the whole run
        """
        t_start = time.perf_counter()

        # Step 1 — ONNX export
        onnx_path = self.export_onnx_only()

        # Step 2 — TensorRT build
        engine_path: Optional[str] = None
        try:
            from inference.tensorrt_engine.builder import build_engine_from_onnx, TRTBuildConfig
        except ImportError:
            logger.warning(
                "inference.tensorrt_engine.builder not importable. "
                "Skipping TensorRT build step."
            )
            return {
                "onnx_path": onnx_path,
                "engine_path": None,
                "export_time_s": round(time.perf_counter() - t_start, 2),
            }

        # Check TensorRT availability before attempting build
        try:
            import tensorrt  # noqa: F401
        except ImportError:
            logger.warning(
                "TensorRT Python bindings not available. "
                "Install with: pip install tensorrt  "
                "ONNX export is complete; skipping engine build."
            )
            return {
                "onnx_path": onnx_path,
                "engine_path": None,
                "export_time_s": round(time.perf_counter() - t_start, 2),
            }

        stem = Path(self.config.model_path).stem
        engine_file = self._output_dir / f"{stem}.engine"

        trt_cfg = TRTBuildConfig(
            onnx_path=onnx_path,
            engine_path=str(engine_file),
            imgsz=self.config.imgsz,
            fp16=self.config.fp16,
            workspace_gb=self.config.workspace_gb,
        )

        logger.info("Building TensorRT engine from %s", onnx_path)
        built_path = build_engine_from_onnx(trt_cfg, overwrite=True)
        engine_path = str(built_path)

        elapsed = round(time.perf_counter() - t_start, 2)
        logger.info("Full export complete in %.1f s", elapsed)

        return {
            "onnx_path": onnx_path,
            "engine_path": engine_path,
            "export_time_s": elapsed,
        }

    def export_onnx_only(self) -> str:
        """
        Export the YOLO .pt model to ONNX only (no TensorRT required).

        Returns
        -------
        str — absolute path to the generated ONNX file.

        Raises
        ------
        ExportError  — ultralytics is not installed.
        FileNotFoundError — model .pt file does not exist.
        """
        model_path = Path(self.config.model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"YOLO model weights not found: {model_path}. "
                "Provide the correct path to a .pt file."
            )

        # Use the module-level YOLO reference (patchable in tests)
        _YOLO = YOLO
        if _YOLO is None:
            raise ExportError(
                "ultralytics is not installed. "
                "Install it with: pip install ultralytics\n"
                "Then retry the export."
            )

        self._output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            "Exporting %s → ONNX  (imgsz=%d, opset=%d, fp16=%s, simplify=%s)",
            model_path,
            self.config.imgsz,
            self.config.opset,
            self.config.fp16,
            self.config.simplify,
        )

        model = _YOLO(str(model_path))

        # ultralytics writes the ONNX next to the .pt file by default;
        # we redirect it to output_dir via the 'project' + 'name' params
        # or by using the returned path and moving if necessary.
        export_kwargs: dict = {
            "format": "onnx",
            "imgsz": self.config.imgsz,
            "opset": self.config.opset,
            "simplify": False,          # we handle simplification ourselves
            "half": self.config.fp16,
            "dynamic": self.config.dynamic_batch,
        }

        exported = model.export(**export_kwargs)

        # ultralytics returns the path as a string or Path-like
        exported_path = Path(str(exported))

        # Move to output_dir if ultralytics placed it elsewhere
        stem = model_path.stem
        target_onnx = self._output_dir / f"{stem}.onnx"
        if exported_path.resolve() != target_onnx.resolve():
            if exported_path.exists():
                target_onnx.parent.mkdir(parents=True, exist_ok=True)
                exported_path.replace(target_onnx)
                logger.debug("Moved %s → %s", exported_path, target_onnx)
            else:
                # ultralytics may have placed it next to model_path
                candidate = model_path.parent / f"{stem}.onnx"
                if candidate.exists():
                    candidate.replace(target_onnx)
                    logger.debug("Moved %s → %s", candidate, target_onnx)
                else:
                    raise ExportError(
                        f"ONNX export reported success but file not found at "
                        f"{exported_path} or {candidate}"
                    )

        onnx_path = str(target_onnx)

        # Optional: onnx-simplifier pass
        if self.config.simplify:
            onnx_path = self._simplify_onnx(onnx_path)

        return onnx_path

    def validate_onnx(self, onnx_path: str) -> dict:
        """
        Validate an ONNX model by running a random test inference with
        ONNXRuntime and checking the output shapes make sense.

        Parameters
        ----------
        onnx_path : str
            Path to the .onnx file to validate.

        Returns
        -------
        dict with keys:
            - ``ok``            : bool — True if inference succeeded
            - ``output_shapes`` : list[tuple] — shapes of each output tensor
            - ``input_shape``   : tuple — shape fed to the model
            - ``error``         : str | None — error message if ok=False
        """
        import numpy as np

        _ort = ort  # use module-level reference (patchable in tests)
        if _ort is None:
            return {
                "ok": False,
                "output_shapes": [],
                "input_shape": (),
                "error": "onnxruntime not installed. pip install onnxruntime",
            }

        onnx_file = Path(onnx_path)
        if not onnx_file.exists():
            return {
                "ok": False,
                "output_shapes": [],
                "input_shape": (),
                "error": f"ONNX file not found: {onnx_file}",
            }

        try:
            # Use CPU provider — validation doesn't require GPU
            sess_options = _ort.SessionOptions()
            sess_options.log_severity_level = 3  # ERROR only
            session = _ort.InferenceSession(
                str(onnx_file),
                sess_options=sess_options,
                providers=["CPUExecutionProvider"],
            )

            input_meta = session.get_inputs()[0]
            # Build a concrete input shape (replace dynamic dims -1/None with 1)
            raw_shape = input_meta.shape
            input_shape = tuple(
                (d if isinstance(d, int) and d > 0 else 1)
                for d in raw_shape
            )

            dummy = np.random.rand(*input_shape).astype(np.float32)
            outputs = session.run(None, {input_meta.name: dummy})
            output_shapes = [tuple(o.shape) for o in outputs]

            logger.info(
                "ONNX validation OK: input=%s, outputs=%s",
                input_shape,
                output_shapes,
            )

            return {
                "ok": True,
                "output_shapes": output_shapes,
                "input_shape": input_shape,
                "error": None,
            }

        except Exception as exc:
            logger.error("ONNX validation failed: %s", exc)
            return {
                "ok": False,
                "output_shapes": [],
                "input_shape": (),
                "error": str(exc),
            }

    # ----------------------------------------------------------------- private

    def _simplify_onnx(self, onnx_path: str) -> str:
        """
        Run onnx-simplifier on *onnx_path* in-place.

        If onnxsim is not installed, logs a DEBUG message and returns the
        original path unchanged.

        Returns the (possibly simplified) ONNX path.
        """
        _onnx = onnx      # module-level reference (patchable in tests)
        _onnxsim = onnxsim

        if _onnx is None or _onnxsim is None:
            logger.debug(
                "onnx-simplifier not installed — skipping simplification. "
                "Install with: pip install onnx-simplifier"
            )
            return onnx_path

        try:
            logger.info("Simplifying ONNX model: %s", onnx_path)
            model = _onnx.load(onnx_path)
            simplified, check_ok = _onnxsim.simplify(model)
            if check_ok:
                _onnx.save(simplified, onnx_path)
                logger.info("ONNX simplification succeeded")
            else:
                logger.warning(
                    "onnxsim.simplify() returned check_ok=False — "
                    "keeping original unsimplified graph"
                )
        except Exception as exc:
            logger.warning("ONNX simplification failed (%s) — using original graph", exc)

        return onnx_path
