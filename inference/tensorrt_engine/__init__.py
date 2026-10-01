"""inference.tensorrt_engine — TensorRT 10.x engine builder and runner (requires NVIDIA TRT SDK + pycuda)."""

from inference.tensorrt_engine.builder import (
    TRTBuildConfig,
    build_engine_from_onnx,
    inspect_engine,
)
from inference.tensorrt_engine.runner import (
    TRICHOME_CLASSES,
    TensorRTRunner,
    TRTDetection,
    TRTResult,
    TRTRunnerConfig,
    tensorrt_available,
)

__all__ = [
    "TRICHOME_CLASSES",
    "TRTBuildConfig",
    "TRTDetection",
    "TRTResult",
    "TRTRunnerConfig",
    "TensorRTRunner",
    "build_engine_from_onnx",
    "inspect_engine",
    "tensorrt_available",
]
