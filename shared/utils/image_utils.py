"""
shared.utils.image_utils — Image I/O and manipulation utilities.

Centralized image loading/saving for consistent behavior across the codebase.
All images are loaded as NumPy uint8 RGB arrays (H, W, 3) internally.
OpenCV BGR ↔ RGB conversion is handled here — callers should never deal with
channel order.

Performance notes for RTX 4060 / i5-13400F:
- Use imageio for TIFF (handles 16-bit scientific images correctly)
- Use cv2 for resizing (faster than PIL for large images)
- Avoid loading full 4K images into RAM if only thumbnails are needed
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Literal

import cv2
import imageio.v3 as iio
import numpy as np
from numpy.typing import NDArray


# Supported image formats
SUPPORTED_EXTENSIONS = {
    ".tif", ".tiff",   # Preferred for scientific microscopy
    ".png",
    ".jpg", ".jpeg",
    ".bmp",
    ".webp",
}


def load_image(
    path: Path | str,
    color_mode: Literal["rgb", "bgr", "gray"] = "rgb",
    normalize: bool = False,
) -> NDArray[np.uint8]:
    """
    Load an image from disk as a NumPy array.

    Handles:
    - 8-bit and 16-bit TIFF files (16-bit is downsampled to 8-bit)
    - Grayscale → RGB conversion
    - RGBA → RGB conversion (alpha channel discarded)
    - Corrupt file detection

    Args:
        path: Image file path
        color_mode: "rgb" (default), "bgr" (for OpenCV direct use), "gray"
        normalize: If True, return float32 in [0,1]. Default uint8 [0,255].

    Returns:
        Image array of shape (H, W, 3) for rgb/bgr, (H, W) for gray.
        dtype: uint8 if normalize=False, float32 if normalize=True.

    Raises:
        FileNotFoundError: If file doesn't exist.
        ValueError: If file format is unsupported or image is corrupt.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported image format: {path.suffix}. "
            f"Supported: {SUPPORTED_EXTENSIONS}"
        )

    try:
        # imageio handles TIFF including multi-page and 16-bit correctly
        image = iio.imread(str(path))
    except Exception as e:
        raise ValueError(f"Failed to load image {path}: {e}") from e

    if image is None or image.size == 0:
        raise ValueError(f"Image is empty or corrupt: {path}")

    # Handle 16-bit → 8-bit conversion (common in scientific microscopy)
    if image.dtype == np.uint16:
        # Scale to 8-bit using percentile normalization for better contrast
        p2, p98 = np.percentile(image, [2, 98])
        if p98 > p2:
            image = np.clip((image.astype(np.float32) - p2) / (p98 - p2) * 255, 0, 255)
        else:
            image = image.astype(np.float32) / 256.0
        image = image.astype(np.uint8)

    # Handle float images
    if image.dtype in (np.float32, np.float64):
        image = (np.clip(image, 0, 1) * 255).astype(np.uint8)

    # Ensure 3D array
    if image.ndim == 2:
        # Grayscale
        if color_mode == "gray":
            return image.astype(np.float32) / 255.0 if normalize else image
        image = np.stack([image, image, image], axis=-1)
    elif image.ndim == 3 and image.shape[2] == 4:
        # RGBA → RGB (discard alpha)
        image = image[:, :, :3]
    elif image.ndim == 3 and image.shape[2] == 1:
        image = np.squeeze(image, axis=-1)
        image = np.stack([image, image, image], axis=-1)

    # imageio loads as RGB by default
    if color_mode == "bgr":
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    elif color_mode == "gray":
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        return image.astype(np.float32) / 255.0 if normalize else image

    if normalize:
        return image.astype(np.float32) / 255.0

    return image.astype(np.uint8)


def save_image(
    image: NDArray,
    path: Path | str,
    quality: int = 95,
) -> None:
    """
    Save a NumPy image array to disk.

    Args:
        image: Image array (H, W, 3) RGB or (H, W) grayscale.
        path: Output file path. Format inferred from extension.
        quality: JPEG quality (1-100). Only applies to JPEG output.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if image.dtype == np.float32 or image.dtype == np.float64:
        image = (np.clip(image, 0, 1) * 255).astype(np.uint8)

    if path.suffix.lower() in (".jpg", ".jpeg"):
        iio.imwrite(str(path), image, quality=quality)
    else:
        iio.imwrite(str(path), image)


def resize_image(
    image: NDArray[np.uint8],
    target_size: int | tuple[int, int],
    maintain_aspect: bool = True,
    interpolation: int = cv2.INTER_LANCZOS4,
) -> NDArray[np.uint8]:
    """
    Resize an image to target size.

    Args:
        image: Input image (H, W, C)
        target_size: If int, resize longest side to this value.
                     If tuple (W, H), resize to exact dimensions.
        maintain_aspect: If True (default), maintain aspect ratio.
        interpolation: OpenCV interpolation method.
            INTER_LANCZOS4: Best quality for downsampling microscopy images.
            INTER_LINEAR: Faster, acceptable for real-time use.

    Returns:
        Resized image.
    """
    h, w = image.shape[:2]

    if isinstance(target_size, int):
        if maintain_aspect:
            scale = target_size / max(h, w)
            new_h = int(h * scale)
            new_w = int(w * scale)
        else:
            new_h = new_w = target_size
    else:
        new_w, new_h = target_size

    if new_w == w and new_h == h:
        return image

    return cv2.resize(image, (new_w, new_h), interpolation=interpolation)


def pad_to_square(
    image: NDArray[np.uint8],
    fill_value: int = 114,
) -> tuple[NDArray[np.uint8], tuple[int, int, int, int]]:
    """
    Pad image to square by adding letterbox borders.

    Used before feeding non-square images to models that expect square input.
    Using fill_value=114 (gray) matches YOLO's default letterbox behavior.

    Returns:
        Padded image and (top, bottom, left, right) padding amounts.
    """
    h, w = image.shape[:2]
    max_side = max(h, w)

    pad_h = max_side - h
    pad_w = max_side - w
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left

    padded = cv2.copyMakeBorder(
        image, top, bottom, left, right,
        cv2.BORDER_CONSTANT, value=(fill_value, fill_value, fill_value)
    )
    return padded, (top, bottom, left, right)


def crop_region(
    image: NDArray[np.uint8],
    x_min: int,
    y_min: int,
    x_max: int,
    y_max: int,
    margin_px: int = 0,
) -> NDArray[np.uint8]:
    """
    Crop a rectangular region from an image with optional margin.

    Args:
        image: Source image (H, W, C)
        x_min, y_min, x_max, y_max: Bounding box coordinates
        margin_px: Expand crop by this many pixels in each direction.

    Returns:
        Cropped region. May be smaller than requested if at image boundary.
    """
    h, w = image.shape[:2]
    x1 = max(0, x_min - margin_px)
    y1 = max(0, y_min - margin_px)
    x2 = min(w, x_max + margin_px)
    y2 = min(h, y_max + margin_px)
    return image[y1:y2, x1:x2].copy()


def compute_image_hash(image: NDArray) -> str:
    """
    Compute perceptual hash for duplicate detection.

    Uses average hash (aHash) — simple but effective for exact and near-duplicate
    detection. For more robust deduplication under transformations, use pHash.

    Returns:
        Hex string of 64-bit average hash.
    """
    # Resize to 8×8 and convert to grayscale
    small = cv2.resize(image, (8, 8))
    if small.ndim == 3:
        small = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
    mean = small.mean()
    bits = (small > mean).flatten()
    # Pack bits into integer
    hash_val = 0
    for bit in bits:
        hash_val = (hash_val << 1) | int(bit)
    return format(hash_val, "016x")


def compute_file_hash(path: Path | str, chunk_size: int = 65536) -> str:
    """SHA-256 hash of file contents — for exact duplicate detection."""
    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            sha256.update(chunk)
    return sha256.hexdigest()


def apply_clahe(
    image: NDArray[np.uint8],
    clip_limit: float = 2.0,
    tile_grid_size: tuple[int, int] = (8, 8),
) -> NDArray[np.uint8]:
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization).

    Applied to the L channel in LAB color space to improve contrast
    without affecting color balance. Essential for low-contrast microscopy images.

    Parameters tuned for trichome microscopy:
    - clip_limit=2.0: Moderate clipping (prevents over-amplification of noise)
    - tile_grid_size=(8,8): Good for 1920×1080 images

    Reference:
        Zuiderveld, K. (1994). "Contrast limited adaptive histogram equalization."
        Graphics Gems IV, Academic Press.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_enhanced = clahe.apply(l_ch)
    enhanced_lab = cv2.merge([l_enhanced, a_ch, b_ch])
    return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)


def compute_image_stats(image: NDArray[np.uint8]) -> dict[str, float]:
    """
    Compute basic image statistics for quality assessment.

    Returns:
        Dict with: mean, std, min, max, brightness, contrast, saturation_mean
    """
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        saturation_mean = float(hsv[:, :, 1].mean())
    else:
        gray = image
        saturation_mean = 0.0

    return {
        "mean": float(gray.mean()),
        "std": float(gray.std()),
        "min": float(gray.min()),
        "max": float(gray.max()),
        "brightness": float(gray.mean()) / 255.0,
        "contrast": float(gray.std()) / 128.0,
        "saturation_mean": saturation_mean / 255.0,
        "dynamic_range": float(gray.max() - gray.min()) / 255.0,
    }
