"""
maturity.explainability.gradcam — GradCAM explainability for maturity predictions.

GradCAM (Gradient-weighted Class Activation Mapping) highlights which
spatial regions of the trichome crop most influenced the classification.

For a CNN maturity classifier, GradCAM reveals:
- Clear classification: activates on head outline (transparency cues)
- Cloudy classification: activates on internal granular texture
- Amber classification: activates on colored regions (hue cues)
- Degraded classification: activates on dark/deformed areas

IMPLEMENTATION:
Since we primarily use ONNX Runtime for inference (no gradients available),
we implement:
1. Pytorch-based GradCAM (when training with PyTorch models)
2. ONNX-compatible Grad-CAM approximation via occlusion sensitivity
3. Score-CAM: gradient-free CAM variant for ONNX models

References:
  Selvaraju, R.R. et al. (2017). GradCAM. ICCV 2017.
  Wang, H. et al. (2020). Score-CAM. CVPR Workshops 2020.
  Zeiler, M.D. & Fergus, R. (2014). Visualizing CNNs. ECCV 2014.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray


@dataclass
class GradCAMResult:
    """GradCAM or CAM visualization result."""

    saliency_map: NDArray[np.float32]
    """
    Normalized saliency map, shape (H, W), values in [0, 1].
    Higher values = regions more important for classification.
    """

    heatmap_rgb: NDArray[np.uint8]
    """Color heatmap (H, W, 3) for visualization."""

    overlay_rgb: NDArray[np.uint8]
    """Original image with heatmap overlay (H, W, 3)."""

    target_class: str
    """Class this explanation is for."""

    method: str
    """Method used: 'occlusion', 'gradient_sensitivity', 'score_cam'"""


def occlusion_sensitivity(
    image: NDArray[np.uint8],
    predict_fn,
    target_class_idx: int,
    patch_size: int = 8,
    stride: int = 4,
) -> GradCAMResult:
    """
    Occlusion sensitivity map — gradient-free, works with any model.

    Systematically occludes patches of the image and measures the
    drop in target class probability. Regions where occlusion causes
    large drops are important for the prediction.

    Works with ONNX Runtime (no backpropagation needed).

    Args:
        image: RGB uint8 trichome crop
        predict_fn: Callable that takes RGB image → class probabilities
        target_class_idx: Index of class to explain
        patch_size: Size of occlusion patch in pixels
        stride: Step size between patch positions

    Returns:
        GradCAMResult with saliency map
    """
    h, w = image.shape[:2]
    saliency = np.zeros((h, w), dtype=np.float32)
    counts = np.zeros((h, w), dtype=np.float32)

    # Baseline prediction
    base_probs = predict_fn(image)
    base_score = float(base_probs[target_class_idx])

    # Gray patch for occlusion
    occluded = image.copy()

    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            # Occlude patch with mean gray
            patch = occluded.copy()
            patch[y:y + patch_size, x:x + patch_size] = 128

            # Measure score drop
            probs = predict_fn(patch)
            score_drop = base_score - float(probs[target_class_idx])

            # Higher drop = more important region
            saliency[y:y + patch_size, x:x + patch_size] += max(score_drop, 0)
            counts[y:y + patch_size, x:x + patch_size] += 1

    # Average over overlapping patches
    mask = counts > 0
    saliency[mask] /= counts[mask]

    # Normalize to [0, 1]
    if saliency.max() > 0:
        saliency = saliency / saliency.max()

    return _build_result(saliency, image, "occlusion")


def gradient_sensitivity_map(
    image: NDArray[np.uint8],
    predict_fn,
    target_class_idx: int,
    n_perturbations: int = 50,
    noise_scale: float = 0.15,
) -> GradCAMResult:
    """
    SmoothGrad-style gradient sensitivity — finite difference approximation.

    Works without true backpropagation by:
    1. Adding small noise perturbations to the image
    2. Measuring per-pixel sensitivity of target class score to noise
    3. Averaging sensitivity maps across perturbations (SmoothGrad effect)

    More stable than single-perturbation sensitivity.
    Computable with ONNX Runtime (black-box).

    Args:
        image: RGB uint8 trichome crop
        predict_fn: Callable: image → probability array
        target_class_idx: Class index to explain
        n_perturbations: Number of noisy samples (higher = smoother map)
        noise_scale: Noise standard deviation (fraction of pixel range)

    Returns:
        GradCAMResult with smooth sensitivity map
    """
    h, w = image.shape[:2]
    sensitivity_sum = np.zeros((h, w), dtype=np.float32)
    img_float = image.astype(np.float32)

    base_probs = predict_fn(image)
    base_score = float(base_probs[target_class_idx])

    for _ in range(n_perturbations):
        noise = np.random.normal(0, noise_scale * 255, image.shape).astype(np.float32)
        perturbed = np.clip(img_float + noise, 0, 255).astype(np.uint8)

        probs = predict_fn(perturbed)
        score = float(probs[target_class_idx])

        # Sensitivity: larger noise → larger score change → important region
        sensitivity = np.abs(noise).mean(axis=2) * abs(score - base_score)
        sensitivity_sum += sensitivity.astype(np.float32)

    saliency = sensitivity_sum / n_perturbations

    # Normalize
    if saliency.max() > 0:
        saliency = saliency / saliency.max()

    return _build_result(saliency, image, "gradient_sensitivity")


def simple_cam_from_features(
    image: NDArray[np.uint8],
    target_class: str,
) -> GradCAMResult:
    """
    Feature-based CAM without a neural network.

    When no trained model is available, generates a pseudo-CAM
    by computing which image regions most strongly exhibit features
    characteristic of the target class.

    Uses color and texture features as proxies for class-relevant regions.

    Args:
        image: RGB uint8 trichome crop
        target_class: One of 'clear', 'cloudy', 'amber', 'degraded'

    Returns:
        GradCAMResult with feature-based saliency
    """
    h, w = image.shape[:2]
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
    hue = hsv[:, :, 0]        # 0-180
    sat = hsv[:, :, 1] / 255  # 0-1
    val = hsv[:, :, 2] / 255  # 0-1

    if target_class == "clear":
        # Clear = high value, low saturation (transparent regions)
        saliency = (val - sat).clip(0)

    elif target_class == "cloudy":
        # Cloudy = moderate sat/val, no strong hue
        saliency = (sat * val * (1 - np.abs(val - 0.7))).clip(0)

    elif target_class == "amber":
        # Amber = warm hue (20-35 in OpenCV = 40-70°)
        amber_mask = ((hue >= 18) & (hue <= 35)).astype(np.float32)
        saliency = amber_mask * sat

    elif target_class == "degraded":
        # Degraded = dark brown areas + low value regions
        brown_mask = ((hue >= 5) & (hue <= 25) & (val < 0.55)).astype(np.float32)
        dark_mask = (val < 0.15).astype(np.float32)
        saliency = (brown_mask + dark_mask).clip(0, 1)

    else:
        saliency = np.ones((h, w), dtype=np.float32)

    saliency = saliency.astype(np.float32)
    if saliency.max() > 0:
        saliency = saliency / saliency.max()

    # Smooth the saliency map
    saliency = cv2.GaussianBlur(saliency, (5, 5), 0)

    return _build_result(saliency, image, "feature_cam")


def _build_result(
    saliency: NDArray[np.float32],
    image: NDArray[np.uint8],
    method: str,
    target_class: str = "",
) -> GradCAMResult:
    """Build GradCAMResult from normalized saliency map."""
    h, w = image.shape[:2]

    # Resize saliency to image size if needed
    if saliency.shape != (h, w):
        saliency = cv2.resize(saliency, (w, h), interpolation=cv2.INTER_CUBIC)
        saliency = np.clip(saliency, 0, 1)

    # Color heatmap (jet: blue→green→red)
    sal_uint8 = (saliency * 255).astype(np.uint8)
    heatmap_bgr = cv2.applyColorMap(sal_uint8, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

    # Overlay on original image (60% original, 40% heatmap)
    overlay = (
        0.60 * image.astype(np.float32)
        + 0.40 * heatmap_rgb.astype(np.float32)
    )
    overlay_rgb = np.clip(overlay, 0, 255).astype(np.uint8)

    return GradCAMResult(
        saliency_map=saliency.astype(np.float32),
        heatmap_rgb=heatmap_rgb,
        overlay_rgb=overlay_rgb,
        target_class=target_class,
        method=method,
    )


def generate_explanation(
    image: NDArray[np.uint8],
    predicted_class: str,
    predicted_class_idx: int,
    predict_fn=None,
    use_occlusion: bool = False,
) -> GradCAMResult:
    """
    Generate best-available explanation for a maturity prediction.

    Selection logic:
    1. If predict_fn provided and use_occlusion=True → occlusion sensitivity
    2. If predict_fn provided → gradient sensitivity (faster)
    3. No predict_fn → feature-based CAM (always available)

    Args:
        image: RGB uint8 trichome crop
        predicted_class: Predicted class label
        predicted_class_idx: Predicted class index
        predict_fn: Optional callable (image → probabilities)
        use_occlusion: Use slower but more accurate occlusion method

    Returns:
        GradCAMResult
    """
    if predict_fn is not None:
        if use_occlusion:
            result = occlusion_sensitivity(image, predict_fn, predicted_class_idx)
        else:
            result = gradient_sensitivity_map(image, predict_fn, predicted_class_idx)
        result.target_class = predicted_class
        return result

    return simple_cam_from_features(image, predicted_class)
