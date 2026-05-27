"""
inference/tensorrt_engine/builder.py — TensorRT engine builder from ONNX.

Converts an ONNX model to an optimised TensorRT .engine file using TRT 10.x
builder API. Supports FP16 mode, profile-based dynamic shapes, and GPU memory
budget enforcement for RTX 4060 (8 GB VRAM).

Usage:
    from inference.tensorrt_engine.builder import build_engine_from_onnx, TRTBuildConfig

    cfg = TRTBuildConfig(
        onnx_path="models/yolo11s_trichome.onnx",
        engine_path="models/yolo11s_trichome_fp16.engine",
        imgsz=1280,
        fp16=True,
        workspace_gb=4.0,
    )
    build_engine_from_onnx(cfg)

The engine file can then be loaded by TensorRTRunner for inference.

TRT 10.x API notes:
  - IBuilderConfig.set_memory_pool_limit(MemoryPoolType.WORKSPACE, bytes)
    replaces the deprecated max_workspace_size attribute.
  - Dynamic shape profiles via IOptimizationProfile.
  - NetworkDefinitionCreationFlag.STRONGLY_TYPED enables strict type checking.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Build configuration
# ---------------------------------------------------------------------------

@dataclass
class TRTBuildConfig:
    """Parameters controlling the TensorRT engine build."""
    onnx_path: str = ""
    engine_path: str = ""
    imgsz: int = 1280
    fp16: bool = True
    int8: bool = False                  # requires calibration dataset
    workspace_gb: float = 4.0          # max GPU workspace during optimisation
    min_batch: int = 1
    opt_batch: int = 1
    max_batch: int = 1                  # RTX 4060: batch=1 for VRAM safety
    verbosity: str = "WARNING"          # VERBOSE / INFO / WARNING / ERROR


# ---------------------------------------------------------------------------
# Build function
# ---------------------------------------------------------------------------

def build_engine_from_onnx(
    config: TRTBuildConfig,
    *,
    overwrite: bool = False,
) -> Path:
    """
    Build a TensorRT engine from an ONNX file.

    Args:
        config: TRTBuildConfig with paths and build options.
        overwrite: If True, rebuild even if engine_path already exists.

    Returns:
        Path to the serialised .engine file.

    Raises:
        FileNotFoundError: ONNX model not found.
        ImportError: TensorRT not installed.
        RuntimeError: Build failure.
    """
    try:
        import tensorrt as trt
    except ImportError as exc:
        raise ImportError(
            "TensorRT not available. "
            "Install: pip install tensorrt  (or apt-get install python3-libnvinfer)"
        ) from exc

    onnx_path = Path(config.onnx_path)
    engine_path = Path(config.engine_path)

    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

    if engine_path.exists() and not overwrite:
        logger.info("Engine already exists (use overwrite=True to rebuild): %s", engine_path)
        return engine_path

    engine_path.parent.mkdir(parents=True, exist_ok=True)

    # -- Logger -----------------------------------------------------------
    verbosity_map = {
        "VERBOSE": trt.Logger.VERBOSE,
        "INFO": trt.Logger.INFO,
        "WARNING": trt.Logger.WARNING,
        "ERROR": trt.Logger.ERROR,
    }
    trt_logger = trt.Logger(verbosity_map.get(config.verbosity.upper(), trt.Logger.WARNING))

    # -- Builder + config -------------------------------------------------
    builder = trt.Builder(trt_logger)
    build_config = builder.create_builder_config()

    # Workspace limit — TRT 10.x API
    workspace_bytes = int(config.workspace_gb * (1024 ** 3))
    build_config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_bytes)

    # Precision flags
    if config.fp16 and builder.platform_has_fast_fp16:
        build_config.set_flag(trt.BuilderFlag.FP16)
        logger.info("FP16 mode enabled")
    elif config.fp16:
        logger.warning("FP16 requested but platform_has_fast_fp16=False — building in FP32")

    if config.int8:
        build_config.set_flag(trt.BuilderFlag.INT8)
        logger.warning("INT8 enabled without calibrator — accuracy may degrade")

    # -- Network + ONNX parser --------------------------------------------
    network_flags = 0  # explicit batch (default in TRT 10)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, trt_logger)

    logger.info("Parsing ONNX: %s", onnx_path)
    with open(onnx_path, "rb") as fh:
        onnx_bytes = fh.read()

    if not parser.parse(onnx_bytes):
        errors = "\n".join(
            str(parser.get_error(i)) for i in range(parser.num_errors)
        )
        raise RuntimeError(f"ONNX parse failed:\n{errors}")

    # -- Optimisation profile (dynamic shapes) ----------------------------
    profile = builder.create_optimization_profile()
    input_tensor = network.get_input(0)
    input_name = input_tensor.name

    # Static imgsz — batch is always 1 for VRAM safety on RTX 4060
    profile.set_shape(
        input_name,
        (config.min_batch, 3, config.imgsz, config.imgsz),   # min
        (config.opt_batch, 3, config.imgsz, config.imgsz),   # opt
        (config.max_batch, 3, config.imgsz, config.imgsz),   # max
    )
    build_config.add_optimization_profile(profile)

    logger.info(
        "Building engine: imgsz=%d, fp16=%s, workspace=%.1f GB",
        config.imgsz,
        config.fp16,
        config.workspace_gb,
    )

    # -- Build ------------------------------------------------------------
    serialised = builder.build_serialized_network(network, build_config)
    if serialised is None:
        raise RuntimeError(
            "TensorRT engine build returned None. "
            "Check VRAM availability and ONNX model compatibility."
        )

    with open(engine_path, "wb") as fh:
        fh.write(serialised)

    size_mb = engine_path.stat().st_size / (1024 ** 2)
    logger.info("Engine saved: %s (%.1f MB)", engine_path, size_mb)
    return engine_path


# ---------------------------------------------------------------------------
# Convenience: inspect engine tensor names + shapes
# ---------------------------------------------------------------------------

def inspect_engine(engine_path: str | Path) -> dict:
    """
    Return a summary dict of an engine's I/O tensors.

    Returns:
        {
          "inputs":  [{"name": str, "shape": tuple, "dtype": str}],
          "outputs": [{"name": str, "shape": tuple, "dtype": str}],
          "trt_version": str,
        }
    """
    try:
        import tensorrt as trt
    except ImportError as exc:
        raise ImportError("TensorRT not available.") from exc

    engine_path = Path(engine_path)
    if not engine_path.exists():
        raise FileNotFoundError(engine_path)

    trt_logger = trt.Logger(trt.Logger.ERROR)
    runtime = trt.Runtime(trt_logger)

    with open(engine_path, "rb") as fh:
        engine = runtime.deserialize_cuda_engine(fh.read())

    if engine is None:
        raise RuntimeError(f"Failed to load engine: {engine_path}")

    inputs, outputs = [], []
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        shape = tuple(engine.get_tensor_shape(name))
        dtype = str(engine.get_tensor_dtype(name))
        mode = engine.get_tensor_mode(name)
        entry = {"name": name, "shape": shape, "dtype": dtype}
        (inputs if str(mode) == "TensorIOMode.INPUT" else outputs).append(entry)

    return {
        "inputs": inputs,
        "outputs": outputs,
        "trt_version": trt.__version__,
        "engine_path": str(engine_path),
        "size_mb": round(engine_path.stat().st_size / (1024 ** 2), 1),
    }
