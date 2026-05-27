"""inference.tensorrt_engine — TensorRT 10.x engine builder and runner (requires NVIDIA TRT SDK + pycuda)."""

from inference.tensorrt_engine.runner import (
    TensorRTRunner,
    TRTRunnerConfig,
    TRTDetection,
    TRTResult,
    tensorrt_available,
    TRICHOME_CLASSES,
)
from inference.tensorrt_engine.builder import (
    build_engine_from_onnx,
    inspect_engine,
    TRTBuildConfig,
)

__all__ = [
    "TensorRTRunner",
    "TRTRunnerConfig",
    "TRTDetection",
    "TRTResult",
    "TRTBuildConfig",
    "tensorrt_available",
    "TRICHOME_CLASSES",
    "build_engine_from_onnx",
    "inspect_engine",
]
