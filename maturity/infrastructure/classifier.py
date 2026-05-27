"""
maturity.infrastructure.classifier — Lightweight CNN classifier for maturity staging.

ARCHITECTURE CHOICE:
EfficientNet-Lite0 as backbone (MobileNetV2-based, optimized for edge inference).

Why EfficientNet-Lite0 over alternatives:
- ResNet18: Good but not optimized for mobile/edge (no depthwise convolutions)
- MobileNetV3: Harder squeeze-excite blocks increase latency
- EfficientNet-B0: Works but has squeezy layers incompatible with some ONNX exports
- EfficientNet-Lite0: No squeeze-excite, no swish activation → ONNX-friendly ✓
  Size: ~4.7M params, ~20MB. FP16 VRAM: <0.5GB. Inference: <5ms per crop.

TRAINING TARGETS:
- 4-class classification: clear, cloudy, amber, degraded
- Optional 5th class: "mixed" for crops containing multiple stages

INPUT:
- Trichome head crops: 64×64 or 96×96 pixels, RGB
- Crops extracted from detections after NMS

OUTPUT:
- Class probabilities (softmax)
- Temperature-scaled calibrated probabilities
- Predicted class + confidence

INTEGRATION:
- Loaded lazily (only when use_trained_model=True in config)
- Inference via ONNX Runtime (no PyTorch required at inference time)
- Falls back to rule-based classifier if model file not found
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

MATURITY_CLASSES = ["clear", "cloudy", "amber", "degraded"]
DEFAULT_INPUT_SIZE = (96, 96)
DEFAULT_TEMPERATURE = 1.5  # Temperature scaling for calibration


class ClassifierOutput(NamedTuple):
    """Output from the CNN maturity classifier."""
    probabilities: NDArray[np.float32]
    """Softmax probabilities for each class, shape (n_classes,)"""
    predicted_class: str
    """Most likely class name"""
    confidence: float
    """Max probability (confidence in prediction)"""
    is_calibrated: bool
    """Whether temperature scaling was applied"""


def _softmax(x: NDArray[np.float32]) -> NDArray[np.float32]:
    """Numerically stable softmax."""
    e = np.exp(x - x.max())
    return (e / e.sum()).astype(np.float32)


def _temperature_scale(
    logits: NDArray[np.float32],
    temperature: float,
) -> NDArray[np.float32]:
    """Apply temperature scaling for calibration."""
    return _softmax(logits / temperature)


class MaturityClassifier:
    """
    Lightweight CNN-based trichome maturity classifier.

    Wraps an ONNX Runtime session for fast CPU/GPU inference.
    Designed for integration into the maturity analysis pipeline.

    Usage:
        classifier = MaturityClassifier.from_path("models/maturity_classifier.onnx")
        output = classifier.predict(crop_rgb)
    """

    def __init__(
        self,
        session,  # onnxruntime.InferenceSession
        input_name: str,
        temperature: float = DEFAULT_TEMPERATURE,
        classes: list[str] | None = None,
    ):
        self._session = session
        self._input_name = input_name
        self._temperature = temperature
        self._classes = classes or MATURITY_CLASSES
        self._input_size = DEFAULT_INPUT_SIZE

    @classmethod
    def from_path(
        cls,
        model_path: str | Path,
        temperature: float = DEFAULT_TEMPERATURE,
        use_gpu: bool = True,
    ) -> "MaturityClassifier":
        """
        Load classifier from ONNX model file.

        Args:
            model_path: Path to .onnx model file
            temperature: Temperature scaling factor (calibration)
            use_gpu: Whether to use CUDA execution provider

        Returns:
            Initialized MaturityClassifier

        Raises:
            FileNotFoundError: If model file does not exist
            ImportError: If onnxruntime is not installed
        """
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Maturity classifier model not found: {model_path}\n"
                "Train a model first with: trichome train maturity\n"
                "Or download pretrained weights from the model registry."
            )

        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError(
                "onnxruntime is required for trained classifier inference. "
                "Install with: pip install onnxruntime-gpu  (or onnxruntime for CPU-only)"
            )

        providers = []
        if use_gpu:
            try:
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            except Exception:
                providers = ["CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]

        session = ort.InferenceSession(str(model_path), providers=providers)
        input_name = session.get_inputs()[0].name

        # Check output shape to determine classes
        output_info = session.get_outputs()[0]
        n_classes = output_info.shape[-1] if output_info.shape else len(MATURITY_CLASSES)
        classes = MATURITY_CLASSES[:n_classes]

        logger.info(
            f"Maturity classifier loaded: {model_path.name} "
            f"({n_classes} classes, providers={providers})"
        )

        return cls(session, input_name, temperature, classes)

    def preprocess(self, image: NDArray[np.uint8]) -> NDArray[np.float32]:
        """
        Preprocess trichome crop for classifier input.

        Resizes to input size, normalizes to [0,1], transposes to NCHW.

        Args:
            image: RGB uint8 image (H, W, 3)

        Returns:
            NCHW float32 tensor (1, 3, H, W) normalized to [-1, 1]
        """
        import cv2

        # Resize
        resized = cv2.resize(image, self._input_size)
        # Normalize to [-1, 1] (EfficientNet standard)
        norm = (resized.astype(np.float32) / 127.5) - 1.0
        # HWC → NCHW
        return norm.transpose(2, 0, 1)[np.newaxis]

    def predict(
        self,
        image: NDArray[np.uint8],
        return_raw_logits: bool = False,
    ) -> ClassifierOutput:
        """
        Run inference on a single trichome crop.

        Args:
            image: RGB uint8 trichome head crop
            return_raw_logits: If True, skip temperature scaling

        Returns:
            ClassifierOutput with probabilities and predicted class
        """
        tensor = self.preprocess(image)
        outputs = self._session.run(None, {self._input_name: tensor})
        logits = outputs[0][0].astype(np.float32)

        if return_raw_logits:
            probs = _softmax(logits)
            calibrated = False
        else:
            probs = _temperature_scale(logits, self._temperature)
            calibrated = True

        pred_idx = int(np.argmax(probs))
        predicted_class = self._classes[pred_idx] if pred_idx < len(self._classes) else "unknown"
        confidence = float(probs[pred_idx])

        return ClassifierOutput(
            probabilities=probs,
            predicted_class=predicted_class,
            confidence=confidence,
            is_calibrated=calibrated,
        )

    def predict_batch(
        self,
        images: list[NDArray[np.uint8]],
        batch_size: int = 32,
    ) -> list[ClassifierOutput]:
        """
        Run inference on multiple crops efficiently.

        Args:
            images: List of RGB uint8 trichome crops
            batch_size: Batch size for inference

        Returns:
            List of ClassifierOutput (same length as input)
        """
        results: list[ClassifierOutput] = []

        for i in range(0, len(images), batch_size):
            batch_imgs = images[i:i + batch_size]
            tensors = np.concatenate(
                [self.preprocess(img) for img in batch_imgs], axis=0
            )

            outputs = self._session.run(None, {self._input_name: tensors})
            logits_batch = outputs[0].astype(np.float32)

            for logits in logits_batch:
                probs = _temperature_scale(logits, self._temperature)
                pred_idx = int(np.argmax(probs))
                cls = self._classes[pred_idx] if pred_idx < len(self._classes) else "unknown"
                results.append(ClassifierOutput(
                    probabilities=probs,
                    predicted_class=cls,
                    confidence=float(probs[pred_idx]),
                    is_calibrated=True,
                ))

        return results


class RuleBasedFallbackClassifier:
    """
    Rule-based fallback classifier using color and texture features.

    Used when no trained ONNX model is available.
    Less accurate than CNN but requires no GPU and no training data.

    Decision tree based on HSV color analysis:
    1. Very dark pixels → degraded
    2. Brown hue → amber/degraded
    3. Low saturation + high value → clear
    4. Medium saturation + medium value → cloudy
    5. Golden/amber hue → amber
    """

    def predict(self, image: NDArray[np.uint8]) -> ClassifierOutput:
        """
        Classify trichome crop using color rules only.

        Args:
            image: RGB uint8 trichome head crop

        Returns:
            ClassifierOutput (probabilities are approximate, not calibrated)
        """
        import cv2

        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
        h = hsv[:, :, 0]   # 0-180
        s = hsv[:, :, 1] / 255.0
        v = hsv[:, :, 2] / 255.0

        # Feature extraction
        mean_s = float(s.mean())
        mean_v = float(v.mean())
        brown_frac = float(((h >= 8) & (h <= 25) & (s > 0.20) & (v < 0.60)).mean())
        dark_frac = float((v < 0.12).mean())
        amber_frac = float(((h >= 20) & (h <= 35) & (s > 0.30)).mean())
        low_sat_frac = float((s < 0.20).mean())

        # Rule-based classification
        probs = np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float32)  # Prior

        if dark_frac > 0.20 or brown_frac > 0.30:
            # Degraded
            probs = np.array([0.05, 0.10, 0.25, 0.60], dtype=np.float32)
        elif amber_frac > 0.25 or brown_frac > 0.15:
            # Amber
            probs = np.array([0.05, 0.15, 0.70, 0.10], dtype=np.float32)
        elif low_sat_frac > 0.50 and mean_v > 0.65:
            # Clear (high value, low saturation = transparent)
            probs = np.array([0.70, 0.20, 0.05, 0.05], dtype=np.float32)
        elif mean_s > 0.15 and 0.40 < mean_v < 0.90:
            # Cloudy (moderate saturation and value)
            probs = np.array([0.10, 0.70, 0.15, 0.05], dtype=np.float32)
        else:
            # Unclear
            probs = np.array([0.25, 0.40, 0.25, 0.10], dtype=np.float32)

        pred_idx = int(np.argmax(probs))
        return ClassifierOutput(
            probabilities=probs,
            predicted_class=MATURITY_CLASSES[pred_idx],
            confidence=float(probs[pred_idx]),
            is_calibrated=False,
        )


def load_classifier(
    model_path: str | Path | None,
    temperature: float = DEFAULT_TEMPERATURE,
    use_gpu: bool = True,
) -> MaturityClassifier | RuleBasedFallbackClassifier:
    """
    Load trained classifier or fall back to rule-based.

    Args:
        model_path: Path to ONNX model. None → rule-based fallback.
        temperature: Temperature scaling factor
        use_gpu: Whether to use CUDA

    Returns:
        MaturityClassifier or RuleBasedFallbackClassifier
    """
    if model_path is None:
        logger.info("No model path provided — using rule-based fallback classifier")
        return RuleBasedFallbackClassifier()

    try:
        return MaturityClassifier.from_path(model_path, temperature, use_gpu)
    except (FileNotFoundError, ImportError) as e:
        logger.warning(f"Could not load trained classifier ({e}). Using rule-based fallback.")
        return RuleBasedFallbackClassifier()
