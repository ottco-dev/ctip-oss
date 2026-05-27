"""
training.augmentation.microscopy_aug — Domain-specific augmentations for trichome microscopy.

WHY CUSTOM AUGMENTATIONS:
Standard ImageNet augmentations are designed for natural photos.
Microscopy images have different properties:
1. No perspective/projective distortion (flat sample plane)
2. No orientation bias (trichomes appear at all angles equally)
3. Color is semantically meaningful (amber/cloudy/clear maturity)
4. Blur IS the signal (focus quality is a feature, not noise)
5. Spatial scale matters (head diameter is diagnostic)

DOMAIN-SAFE AUGMENTATIONS (preserve semantic content):
✅ Rotation (any angle — microscopy has no canonical orientation)
✅ Horizontal/vertical flip (symmetric task)
✅ Mild brightness/contrast adjustment (lighting variation)
✅ Mild HSV shift (white balance variation)
✅ Gaussian noise (sensor noise)
✅ Grid distortion (mild, simulates field curvature)
✅ Elastic transform (very mild, simulates slight focus variation)

DOMAIN-UNSAFE AUGMENTATIONS (do NOT use):
❌ Strong hue shift (would change amber↔cloudy classification)
❌ Strong saturation shift (affects translucency appearance)
❌ Scale changes (would alter head diameter — diagnostic feature)
❌ Artificial blur (confounds focus quality as a feature)
❌ Heavy compression artifacts (not in microscopy data)
❌ CutMix/GridMask (removes semantic head regions)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray


@dataclass
class MicroscopyAugConfig:
    """
    Configuration for microscopy domain augmentations.

    Conservative defaults preserve maturity-related color information.
    """

    # Geometric
    enable_rotation: bool = True
    rotation_limit_deg: float = 180.0
    """Full rotation — trichomes have no orientation bias."""

    enable_flip: bool = True
    flip_horizontal_prob: float = 0.5
    flip_vertical_prob: float = 0.5

    enable_grid_distortion: bool = True
    grid_distortion_limit: float = 0.10
    """Very mild grid distortion for field curvature simulation."""

    # Color (CONSERVATIVE — maturity stage is color-coded)
    enable_brightness_contrast: bool = True
    brightness_limit: float = 0.15
    """±15% brightness — simulates illumination variation."""

    contrast_limit: float = 0.15

    enable_hsv_shift: bool = True
    hue_shift_limit: int = 8
    """±8° hue shift — enough for white balance variation, not maturity change."""

    saturation_shift_limit: int = 15
    value_shift_limit: int = 20

    # Noise
    enable_gaussian_noise: bool = True
    noise_var_limit: tuple[float, float] = (5.0, 30.0)
    """Sensor noise range (variance in pixel values)."""

    # Focus-related (be very conservative)
    enable_mild_blur: bool = False
    """
    Only enable if training data includes both sharp and slightly blurry frames.
    If your dataset is all sharp frames, do NOT enable blur augmentation.
    """
    blur_limit: int = 3
    """Max kernel size for any blur operation (if enabled)."""

    # Probability gates
    geometric_prob: float = 0.8
    color_prob: float = 0.5
    noise_prob: float = 0.3


def build_albumentations_pipeline(
    config: MicroscopyAugConfig | None = None,
    is_training: bool = True,
) -> Any:
    """
    Build Albumentations augmentation pipeline for trichome microscopy.

    Args:
        config: Augmentation configuration.
        is_training: If False, returns minimal validation transform (only normalize).

    Returns:
        Albumentations Compose transform.

    Requires: pip install albumentations>=1.3.0
    """
    try:
        import albumentations as A
        from albumentations.pytorch import ToTensorV2
    except ImportError:
        raise ImportError(
            "Albumentations not installed. "
            "Install with: pip install albumentations>=1.3.0"
        )

    cfg = config or MicroscopyAugConfig()

    if not is_training:
        return A.Compose(
            [
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ],
            bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]),
        )

    transforms = []

    # ── GEOMETRIC AUGMENTATIONS ─────────────────────────────────────
    geometric_transforms = []

    if cfg.enable_rotation:
        geometric_transforms.append(
            A.Rotate(
                limit=cfg.rotation_limit_deg,
                interpolation=cv2.INTER_LANCZOS4,
                border_mode=cv2.BORDER_REFLECT_101,
                p=1.0,
            )
        )

    if cfg.enable_flip:
        if cfg.flip_horizontal_prob > 0:
            geometric_transforms.append(
                A.HorizontalFlip(p=cfg.flip_horizontal_prob)
            )
        if cfg.flip_vertical_prob > 0:
            geometric_transforms.append(
                A.VerticalFlip(p=cfg.flip_vertical_prob)
            )

    if cfg.enable_grid_distortion:
        geometric_transforms.append(
            A.GridDistortion(
                num_steps=5,
                distort_limit=cfg.grid_distortion_limit,
                interpolation=cv2.INTER_LANCZOS4,
                border_mode=cv2.BORDER_REFLECT_101,
                p=0.4,
            )
        )

    if geometric_transforms:
        transforms.append(
            A.SomeOf(geometric_transforms, n=len(geometric_transforms), replace=False, p=cfg.geometric_prob)
        )

    # ── COLOR AUGMENTATIONS ─────────────────────────────────────────
    color_transforms = []

    if cfg.enable_brightness_contrast:
        color_transforms.append(
            A.RandomBrightnessContrast(
                brightness_limit=cfg.brightness_limit,
                contrast_limit=cfg.contrast_limit,
                p=0.7,
            )
        )

    if cfg.enable_hsv_shift:
        color_transforms.append(
            A.HueSaturationValue(
                hue_shift_limit=cfg.hue_shift_limit,
                sat_shift_limit=cfg.saturation_shift_limit,
                val_shift_limit=cfg.value_shift_limit,
                p=0.5,
            )
        )

    # CLAHE for local contrast enhancement (simulates different microscope settings)
    color_transforms.append(
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.2)
    )

    if color_transforms:
        transforms.append(
            A.OneOf(color_transforms, p=cfg.color_prob)
        )

    # ── NOISE AUGMENTATIONS ─────────────────────────────────────────
    noise_transforms = []

    if cfg.enable_gaussian_noise:
        noise_transforms.append(
            A.GaussNoise(
                var_limit=cfg.noise_var_limit,
                mean=0,
                per_channel=True,
                p=1.0,
            )
        )

    # ISO noise (simulates camera sensor noise at different settings)
    noise_transforms.append(
        A.ISONoise(
            color_shift=(0.01, 0.05),
            intensity=(0.1, 0.5),
            p=1.0,
        )
    )

    if noise_transforms and cfg.enable_gaussian_noise:
        transforms.append(
            A.OneOf(noise_transforms, p=cfg.noise_prob)
        )

    # ── MILD BLUR (OPTIONAL) ────────────────────────────────────────
    if cfg.enable_mild_blur:
        transforms.append(
            A.OneOf([
                A.MotionBlur(blur_limit=cfg.blur_limit, p=1.0),
                A.MedianBlur(blur_limit=cfg.blur_limit, p=1.0),
            ], p=0.2)
        )

    # ── NORMALIZATION ───────────────────────────────────────────────
    transforms.append(
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
    )
    transforms.append(ToTensorV2())

    return A.Compose(
        transforms,
        bbox_params=A.BboxParams(
            format="yolo",
            label_fields=["class_labels"],
            min_visibility=0.3,  # Discard boxes with <30% visibility after augmentation
        ),
    )


def augment_image_only(
    image: NDArray[np.uint8],
    config: MicroscopyAugConfig | None = None,
) -> NDArray[np.uint8]:
    """
    Apply augmentations to an image without bounding box handling.

    Useful for maturity classification training (no bounding boxes).

    Args:
        image: RGB uint8 array (H, W, 3).
        config: Augmentation configuration.

    Returns:
        Augmented RGB uint8 array.
    """
    cfg = config or MicroscopyAugConfig()

    try:
        import albumentations as A
        aug_list = []

        if cfg.enable_rotation:
            aug_list.append(A.Rotate(limit=cfg.rotation_limit_deg, p=0.8))
        if cfg.enable_flip:
            aug_list.extend([A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5)])
        if cfg.enable_brightness_contrast:
            aug_list.append(A.RandomBrightnessContrast(
                brightness_limit=cfg.brightness_limit,
                contrast_limit=cfg.contrast_limit,
                p=0.5,
            ))
        if cfg.enable_hsv_shift:
            aug_list.append(A.HueSaturationValue(
                hue_shift_limit=cfg.hue_shift_limit,
                sat_shift_limit=cfg.saturation_shift_limit,
                val_shift_limit=cfg.value_shift_limit,
                p=0.3,
            ))
        if cfg.enable_gaussian_noise:
            aug_list.append(A.GaussNoise(var_limit=cfg.noise_var_limit, p=0.3))

        pipeline = A.Compose(aug_list)
        result = pipeline(image=image)
        return result["image"]

    except ImportError:
        # Fallback: basic augmentation without albumentations
        return _basic_augment_numpy(image, cfg)


def _basic_augment_numpy(
    image: NDArray[np.uint8],
    cfg: MicroscopyAugConfig,
) -> NDArray[np.uint8]:
    """
    Basic augmentation using only numpy and OpenCV (no albumentations).
    Used as fallback when albumentations is not installed.
    """
    result = image.copy()

    # Random horizontal flip
    if cfg.enable_flip and np.random.random() < 0.5:
        result = np.fliplr(result)

    # Random vertical flip
    if cfg.enable_flip and np.random.random() < 0.5:
        result = np.flipud(result)

    # Random rotation
    if cfg.enable_rotation and np.random.random() < 0.8:
        angle = np.random.uniform(-cfg.rotation_limit_deg, cfg.rotation_limit_deg)
        h, w = result.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        result = cv2.warpAffine(
            result, M, (w, h),
            flags=cv2.INTER_LANCZOS4,
            borderMode=cv2.BORDER_REFLECT_101,
        )

    # Brightness/contrast
    if cfg.enable_brightness_contrast and np.random.random() < 0.5:
        alpha = 1.0 + np.random.uniform(-cfg.contrast_limit, cfg.contrast_limit)
        beta = np.random.uniform(-cfg.brightness_limit * 255, cfg.brightness_limit * 255)
        result = np.clip(result.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

    # Gaussian noise
    if cfg.enable_gaussian_noise and np.random.random() < 0.3:
        var = np.random.uniform(*cfg.noise_var_limit)
        noise = np.random.normal(0, np.sqrt(var), result.shape).astype(np.float32)
        result = np.clip(result.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    return result
