"""
shared.visualization.annotator — Draw annotations on images.

All rendering uses OpenCV for performance. Returned images are RGB uint8.
Supports: bounding boxes, masks, polygons, labels, confidence bars, uncertainty halos.
"""
from __future__ import annotations
from typing import Any
import cv2
import numpy as np
from numpy.typing import NDArray

# Color palette per trichome type (BGR for OpenCV, converted to RGB on output)
TRICHOME_COLORS: dict[str, tuple[int, int, int]] = {
    "capitate_stalked": (163, 113, 247),   # purple
    "capitate_sessile": (56, 139, 253),    # blue
    "bulbous":          (139, 148, 158),   # gray
    "non_glandular":    (100, 100, 100),   # dark gray
    "unknown":          (200, 200, 200),   # light gray
}

MATURITY_COLORS: dict[str, tuple[int, int, int]] = {
    "clear":            (200, 230, 255),   # pale blue
    "cloudy":           (56, 139, 253),    # blue
    "cloudy_amber_mix": (210, 153, 34),    # orange
    "amber":            (210, 153, 34),    # amber
    "degraded":         (218, 54, 51),     # red
    "unknown":          (150, 150, 150),
}


def draw_detections(
    image: NDArray[np.uint8],
    detections: list[dict[str, Any]],
    show_confidence: bool = True,
    show_uncertainty: bool = True,
    line_thickness: int = 2,
    font_scale: float = 0.45,
) -> NDArray[np.uint8]:
    """
    Draw detection bounding boxes on an image.

    Args:
        image: RGB image (H, W, 3)
        detections: List of dicts: {bbox:[x1,y1,x2,y2], label:str, confidence:float, uncertainty:float}
        show_confidence: Draw confidence score
        show_uncertainty: Draw uncertainty halo for uncertain predictions
        line_thickness: Box border thickness in pixels
        font_scale: Label text size

    Returns:
        Annotated RGB image.
    """
    out = image.copy()
    h, w = out.shape[:2]

    for det in detections:
        bbox = det.get("bbox", [0, 0, 10, 10])
        label = det.get("label", "unknown")
        conf = det.get("confidence", 0.0)
        uncertainty = det.get("uncertainty", None)

        x1, y1, x2, y2 = [int(v) for v in bbox]
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(w - 1, x2); y2 = min(h - 1, y2)

        color = TRICHOME_COLORS.get(label, (200, 200, 200))

        # Uncertainty halo: yellow pulsing border for uncertain predictions
        if show_uncertainty and uncertainty is not None and uncertainty > 0.15:
            cv2.rectangle(out, (x1 - 3, y1 - 3), (x2 + 3, y2 + 3),
                          (210, 153, 34), thickness=2)

        # Main bounding box
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness=line_thickness)

        # Label background
        label_text = f"{label.replace('_', ' ')}"
        if show_confidence:
            label_text += f" {conf:.2f}"

        (text_w, text_h), baseline = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
        )
        label_y = max(y1 - 4, text_h + 4)
        cv2.rectangle(
            out,
            (x1, label_y - text_h - baseline - 2),
            (x1 + text_w + 4, label_y + 2),
            color, cv2.FILLED
        )
        # Text color: white on dark, black on light
        brightness = sum(color) / 3
        text_color = (0, 0, 0) if brightness > 150 else (255, 255, 255)
        cv2.putText(
            out, label_text, (x1 + 2, label_y - baseline),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, 1, cv2.LINE_AA
        )

    return out


def draw_masks(
    image: NDArray[np.uint8],
    masks: list[NDArray[np.bool_]],
    labels: list[str] | None = None,
    alpha: float = 0.35,
) -> NDArray[np.uint8]:
    """
    Overlay instance segmentation masks on image.

    Args:
        image: RGB image (H, W, 3)
        masks: List of binary masks (H, W) each
        labels: Optional label per mask for color assignment
        alpha: Mask opacity [0=transparent, 1=opaque]

    Returns:
        Image with colored mask overlays.
    """
    out = image.astype(np.float32)

    for i, mask in enumerate(masks):
        if mask.shape[:2] != image.shape[:2]:
            continue
        label = (labels[i] if labels else "unknown")
        color = np.array(TRICHOME_COLORS.get(label, (100, 200, 100)), dtype=np.float32)

        mask_3ch = np.stack([mask, mask, mask], axis=-1)
        out[mask_3ch] = out[mask_3ch] * (1 - alpha) + color * alpha

        # Draw mask contour
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(out.astype(np.uint8), contours, -1,
                         tuple(int(c) for c in color), 1)

    return np.clip(out, 0, 255).astype(np.uint8)


def create_confidence_heatmap(
    shape: tuple[int, int],
    detections: list[dict[str, Any]],
    sigma: float = 30.0,
) -> NDArray[np.uint8]:
    """
    Create a spatial confidence heatmap from detections.

    Gaussian kernel centered at each detection's center.
    Intensity proportional to confidence.

    Useful for:
    - Visualizing detector confidence distribution across image
    - Identifying high-confidence vs uncertain regions
    - Quality control of detection density
    """
    h, w = shape
    heatmap = np.zeros((h, w), dtype=np.float32)

    for det in detections:
        bbox = det.get("bbox", [0, 0, 0, 0])
        conf = float(det.get("confidence", 0.5))
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2

        # Create Gaussian kernel
        y_grid, x_grid = np.ogrid[:h, :w]
        gaussian = conf * np.exp(
            -((x_grid - cx) ** 2 + (y_grid - cy) ** 2) / (2 * sigma ** 2)
        )
        heatmap += gaussian

    # Normalize and apply colormap
    if heatmap.max() > 0:
        heatmap = (heatmap / heatmap.max() * 255).astype(np.uint8)
    colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_TURBO)
    return cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
