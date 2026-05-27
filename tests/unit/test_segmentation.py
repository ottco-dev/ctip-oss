"""
tests/unit/test_segmentation.py — Unit tests for segmentation utilities.

No GPU or SAM2 model required. Tests mask utilities, polygon conversions,
and refinement logic on synthetic masks.
"""

import pytest
import numpy as np


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def square_mask():
    """100×100 mask with a 60×60 square in the center."""
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[20:80, 20:80] = 255
    return mask


@pytest.fixture
def noisy_mask():
    """Mask with small noise components."""
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[20:80, 20:80] = 255  # Main object
    mask[5, 5] = 255          # Noise pixel
    mask[95, 95] = 255        # Noise pixel
    return mask


@pytest.fixture
def circular_mask():
    """Approximately circular mask."""
    mask = np.zeros((100, 100), dtype=np.uint8)
    center = (50, 50)
    for y in range(100):
        for x in range(100):
            if (x - center[0]) ** 2 + (y - center[1]) ** 2 <= 900:  # r=30
                mask[y, x] = 255
    return mask


# ---------------------------------------------------------------------------
# Mask refinement tests
# ---------------------------------------------------------------------------


def test_remove_small_components_cleans_noise(noisy_mask):
    """Noise pixels should be removed by small component removal."""
    from segmentation.domain.mask_refinement import remove_small_components

    cleaned = remove_small_components(noisy_mask, min_area_px=50)
    # Noise pixels should be gone
    assert cleaned[5, 5] == 0
    assert cleaned[95, 95] == 0
    # Main object preserved
    assert cleaned[50, 50] == 255


def test_fill_holes_fills_interior(square_mask):
    """Interior holes should be filled."""
    from segmentation.domain.mask_refinement import fill_holes

    # Create a mask with a hole
    holed = square_mask.copy()
    holed[40:60, 40:60] = 0  # Interior hole

    filled = fill_holes(holed)
    # Hole should be filled
    assert filled[50, 50] == 255


def test_morphological_clean_preserves_main_object(square_mask):
    """Morphological cleaning should preserve the main object."""
    from segmentation.domain.mask_refinement import morphological_clean

    cleaned = morphological_clean(square_mask, close_kernel=3, open_kernel=2)
    # Center should still be set
    assert cleaned[50, 50] == 255
    # Overall shape should be largely preserved
    original_area = (square_mask > 0).sum()
    cleaned_area = (cleaned > 0).sum()
    # Allow ±10% area change from morphology
    assert abs(original_area - cleaned_area) / original_area < 0.10


def test_refine_mask_pipeline(noisy_mask):
    """Full refinement pipeline should produce valid mask."""
    from segmentation.domain.mask_refinement import refine_mask, RefinementConfig

    config = RefinementConfig(
        close_kernel_size=3,
        open_kernel_size=2,
        min_component_area_px=50,
        smooth_epsilon_fraction=0.01,
    )
    refined = refine_mask(noisy_mask, config=config)

    assert refined.dtype == np.uint8
    assert refined.shape == noisy_mask.shape
    assert set(np.unique(refined)).issubset({0, 255})
    # Main object preserved
    assert refined[50, 50] == 255


# ---------------------------------------------------------------------------
# Polygon tests
# ---------------------------------------------------------------------------


def test_mask_to_polygon_circle(circular_mask):
    """Circular mask should produce a polygon with ~4+ vertices."""
    from segmentation.domain.polygon_utils import mask_to_polygon

    polygons = mask_to_polygon(circular_mask, simplify_epsilon=2.0)
    assert len(polygons) >= 1
    assert len(polygons[0]) >= 4


def test_polygon_to_mask_and_back():
    """Polygon → mask → polygon should be consistent."""
    from segmentation.domain.polygon_utils import polygon_to_mask, mask_to_polygon, polygon_area

    polygon = [[30, 30], [70, 30], [70, 70], [30, 70]]
    h, w = 100, 100

    mask = polygon_to_mask(polygon, h, w)
    assert mask.shape == (h, w)
    assert mask[50, 50] == 255

    recovered = mask_to_polygon(mask, simplify_epsilon=1.0)
    assert len(recovered) >= 1
    orig_area = polygon_area(polygon)
    rec_area = polygon_area(recovered[0])
    assert abs(orig_area - rec_area) / orig_area < 0.05


def test_rle_encoding_roundtrip():
    """mask → RLE → mask roundtrip should be lossless."""
    from segmentation.domain.polygon_utils import mask_to_rle, rle_to_mask

    mask = np.zeros((80, 100), dtype=np.uint8)
    mask[10:60, 15:75] = 255

    rle = mask_to_rle(mask)
    assert "counts" in rle
    assert "size" in rle

    recovered = rle_to_mask(rle)
    assert recovered.shape == mask.shape
    np.testing.assert_array_equal(mask > 0, recovered > 0)


def test_polygon_circularity_circle(circular_mask):
    """Circularity of a circular mask should be close to 1.0."""
    from segmentation.domain.polygon_utils import mask_to_polygon, polygon_circularity

    polygons = mask_to_polygon(circular_mask, simplify_epsilon=1.0)
    assert len(polygons) >= 1

    circ = polygon_circularity(polygons[0])
    # A circle should have circularity close to 1.0
    assert circ > 0.7, f"Expected circularity > 0.7, got {circ:.4f}"


def test_polygon_circularity_square():
    """Circularity of a square should be ~0.785 (π/4)."""
    from segmentation.domain.polygon_utils import polygon_circularity
    import math

    square = [[0, 0], [10, 0], [10, 10], [0, 10]]
    circ = polygon_circularity(square)
    # π/4 ≈ 0.785 for a perfect square
    assert 0.6 < circ < 0.9


def test_bbox_from_polygon():
    """Bounding box should enclose polygon."""
    from segmentation.domain.polygon_utils import bbox_from_polygon

    polygon = [[10, 20], [50, 20], [50, 80], [10, 80]]
    x1, y1, x2, y2 = bbox_from_polygon(polygon)
    assert x1 == 10
    assert y1 == 20
    assert x2 == 50
    assert y2 == 80


def test_polygon_iou_identical():
    """IoU of identical polygons should be 1.0."""
    from segmentation.domain.polygon_utils import polygon_iou

    polygon = [[20, 20], [60, 20], [60, 60], [20, 60]]
    iou = polygon_iou(polygon, polygon, h=100, w=100)
    assert iou == pytest.approx(1.0, abs=0.01)


def test_polygon_iou_no_overlap():
    """IoU of non-overlapping polygons should be ~0.0."""
    from segmentation.domain.polygon_utils import polygon_iou

    poly_a = [[0, 0], [20, 0], [20, 20], [0, 20]]
    poly_b = [[50, 50], [70, 50], [70, 70], [50, 70]]
    iou = polygon_iou(poly_a, poly_b, h=100, w=100)
    assert iou < 0.01
