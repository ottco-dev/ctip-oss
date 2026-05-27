"""
tests/unit/test_detection_metrics.py — Unit tests for detection metrics.

No GPU required — all tests use synthetic data.
"""

import pytest
import numpy as np


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def perfect_predictions():
    """Predictions that exactly match ground truth."""
    gt = [
        {"x1": 10, "y1": 10, "x2": 50, "y2": 50, "class_id": 0},
        {"x1": 100, "y1": 100, "x2": 200, "y2": 200, "class_id": 1},
    ]
    preds = [
        {"x1": 10, "y1": 10, "x2": 50, "y2": 50, "confidence": 0.99, "class_id": 0},
        {"x1": 100, "y1": 100, "x2": 200, "y2": 200, "confidence": 0.95, "class_id": 1},
    ]
    return gt, preds


@pytest.fixture
def empty_predictions():
    """No predictions."""
    gt = [{"x1": 10, "y1": 10, "x2": 50, "y2": 50, "class_id": 0}]
    preds = []
    return gt, preds


# ---------------------------------------------------------------------------
# IoU tests
# ---------------------------------------------------------------------------


def test_iou_perfect_overlap():
    """IoU of identical boxes should be 1.0."""
    from shared.utils.geometry import compute_iou

    box = [10.0, 10.0, 50.0, 50.0]
    assert compute_iou(box, box) == pytest.approx(1.0, abs=1e-6)


def test_iou_no_overlap():
    """IoU of non-overlapping boxes should be 0.0."""
    from shared.utils.geometry import compute_iou

    box_a = [0.0, 0.0, 10.0, 10.0]
    box_b = [20.0, 20.0, 30.0, 30.0]
    assert compute_iou(box_a, box_b) == pytest.approx(0.0, abs=1e-6)


def test_iou_half_overlap():
    """IoU of boxes with 50% overlap."""
    from shared.utils.geometry import compute_iou

    # box_a: [0, 0, 20, 10] area=200; box_b: [10, 0, 30, 10] area=200
    # Intersection: [10, 0, 20, 10] area=100; Union=300
    box_a = [0.0, 0.0, 20.0, 10.0]
    box_b = [10.0, 0.0, 30.0, 10.0]
    iou = compute_iou(box_a, box_b)
    assert iou == pytest.approx(100.0 / 300.0, abs=1e-6)


def test_iou_symmetry():
    """IoU should be symmetric."""
    from shared.utils.geometry import compute_iou

    a = [5.0, 5.0, 25.0, 25.0]
    b = [10.0, 10.0, 35.0, 35.0]
    assert compute_iou(a, b) == pytest.approx(compute_iou(b, a), abs=1e-9)


# ---------------------------------------------------------------------------
# Entropy tests
# ---------------------------------------------------------------------------


def test_entropy_uniform():
    """Uniform distribution should have maximum entropy."""
    from active_learning.sampling.entropy import compute_entropy
    import math

    probs = [0.25, 0.25, 0.25, 0.25]
    h = compute_entropy(probs)
    expected = math.log(4)
    assert h == pytest.approx(expected, abs=1e-6)


def test_entropy_certain():
    """Certain prediction (one-hot) should have near-zero entropy."""
    from active_learning.sampling.entropy import compute_entropy

    probs = [1.0, 0.0, 0.0, 0.0]
    h = compute_entropy(probs)
    assert h < 1e-5


def test_entropy_nonnegative():
    """Entropy is always non-negative."""
    from active_learning.sampling.entropy import compute_entropy

    probs = [0.6, 0.3, 0.07, 0.03]
    assert compute_entropy(probs) >= 0.0


def test_normalized_entropy_bounds():
    """Normalized entropy should be in [0, 1]."""
    from active_learning.sampling.entropy import compute_normalized_entropy
    import math

    for probs in [
        [0.25, 0.25, 0.25, 0.25],  # max entropy
        [1.0, 0.0, 0.0, 0.0],       # min entropy
        [0.7, 0.2, 0.05, 0.05],     # intermediate
    ]:
        ne = compute_normalized_entropy(probs, num_classes=4)
        assert 0.0 <= ne <= 1.0 + 1e-9


# ---------------------------------------------------------------------------
# Disagreement tests
# ---------------------------------------------------------------------------


def test_disagreement_identical_predictions():
    """No disagreement when all ensemble members agree."""
    from active_learning.sampling.disagreement import (
        EnsemblePrediction,
        compute_disagreement,
    )

    probs = [0.9, 0.05, 0.03, 0.02]
    predictions = [
        EnsemblePrediction("s1", probs, 0, 0.9),
        EnsemblePrediction("s1", probs, 0, 0.9),
        EnsemblePrediction("s1", probs, 0, 0.9),
    ]
    result = compute_disagreement(predictions)
    assert result.bald_score < 0.01  # Near-zero BALD
    assert result.vote_entropy < 0.01  # All vote for class 0


def test_disagreement_maximum():
    """Maximum disagreement when ensemble evenly split."""
    from active_learning.sampling.disagreement import (
        EnsemblePrediction,
        compute_disagreement,
    )

    preds = [
        EnsemblePrediction("s1", [1.0, 0.0, 0.0, 0.0], 0, 1.0),
        EnsemblePrediction("s1", [0.0, 1.0, 0.0, 0.0], 1, 1.0),
        EnsemblePrediction("s1", [0.0, 0.0, 1.0, 0.0], 2, 1.0),
        EnsemblePrediction("s1", [0.0, 0.0, 0.0, 1.0], 3, 1.0),
    ]
    result = compute_disagreement(preds)
    assert result.vote_entropy > 1.0  # High vote entropy


def test_disagreement_composite_nonnegative():
    """Composite score should always be non-negative."""
    from active_learning.sampling.disagreement import (
        EnsemblePrediction,
        compute_disagreement,
    )

    for _ in range(20):
        probs = np.random.dirichlet([1, 1, 1, 1]).tolist()
        preds = [
            EnsemblePrediction("s1", np.random.dirichlet([1, 1, 1, 1]).tolist(), 0, 0.5)
            for _ in range(3)
        ]
        result = compute_disagreement(preds)
        assert result.composite_score >= 0.0


# ---------------------------------------------------------------------------
# Priority queue tests
# ---------------------------------------------------------------------------


def test_priority_queue_order():
    """Higher priority items should be returned first."""
    from active_learning.queuing.priority_queue import AnnotationPriorityQueue

    q = AnnotationPriorityQueue()
    q.push("s1", "d1", "/img/s1.jpg", uncertainty_score=0.1)  # Low
    q.push("s2", "d1", "/img/s2.jpg", uncertainty_score=0.9)  # High
    q.push("s3", "d1", "/img/s3.jpg", uncertainty_score=0.5)  # Medium

    first = q.pop()
    assert first is not None
    assert first.sample_id == "s2"  # Highest uncertainty first


def test_priority_queue_complete():
    """Completed items should not be returned again."""
    from active_learning.queuing.priority_queue import AnnotationPriorityQueue

    q = AnnotationPriorityQueue()
    entry = q.push("s1", "d1", "/img/s1.jpg", uncertainty_score=0.9)
    popped = q.pop()
    assert popped is not None
    assert popped.status == "assigned"

    q.complete(entry.item_id)
    assert q.stats().completed == 1


def test_priority_queue_len():
    """Queue length should reflect pending items only."""
    from active_learning.queuing.priority_queue import AnnotationPriorityQueue

    q = AnnotationPriorityQueue()
    for i in range(5):
        q.push(f"s{i}", "d1", f"/img/s{i}.jpg")

    assert len(q) == 5
    q.pop()
    assert len(q) == 4


# ---------------------------------------------------------------------------
# Trigger tests
# ---------------------------------------------------------------------------


def test_trigger_cooldown():
    """Should not trigger during cooldown period."""
    from active_learning.retraining.trigger import RetrainingTrigger, TriggerConfig
    from datetime import datetime, timedelta

    config = TriggerConfig(min_retraining_interval_hours=2.0)
    trigger = RetrainingTrigger(config=config)
    trigger.state.last_retrain_at = datetime.utcnow() - timedelta(minutes=30)
    trigger.state.total_annotations = 200
    trigger.on_annotation_approved(500)

    decision = trigger.evaluate()
    assert not decision.should_retrain


def test_trigger_annotation_threshold():
    """Should trigger when annotation count threshold is met."""
    from active_learning.retraining.trigger import RetrainingTrigger, TriggerConfig
    from datetime import datetime, timedelta

    config = TriggerConfig(
        annotation_count_threshold=10,
        min_retraining_interval_hours=0.0,
        min_annotations_for_trigger=5,
    )
    trigger = RetrainingTrigger(config=config)
    trigger.state.total_annotations = 50
    trigger.on_annotation_approved(15)  # Above threshold

    decision = trigger.evaluate()
    assert decision.should_retrain
    assert any("annotation" in r.lower() for r in decision.reasons)


# ---------------------------------------------------------------------------
# Polygon utilities tests
# ---------------------------------------------------------------------------


def test_polygon_area_square():
    """Area of a 10×10 square should be 100."""
    from segmentation.domain.polygon_utils import polygon_area

    square = [[0, 0], [10, 0], [10, 10], [0, 10]]
    assert polygon_area(square) == pytest.approx(100.0, abs=1e-6)


def test_polygon_to_mask_roundtrip():
    """Converting polygon → mask → polygon should approximately preserve shape."""
    from segmentation.domain.polygon_utils import polygon_to_mask, mask_to_polygon, polygon_area

    polygon = [[50, 50], [150, 50], [150, 150], [50, 150]]
    mask = polygon_to_mask(polygon, height=200, width=200)
    recovered = mask_to_polygon(mask, simplify_epsilon=2.0)

    assert len(recovered) > 0
    area = polygon_area(polygon)
    recovered_area = polygon_area(recovered[0])
    # Should preserve area within 5%
    assert abs(area - recovered_area) / area < 0.05


def test_mask_to_rle_roundtrip():
    """RLE encode → decode should produce identical mask."""
    from segmentation.domain.polygon_utils import mask_to_rle, rle_to_mask
    import numpy as np

    mask = np.zeros((50, 60), dtype=np.uint8)
    mask[10:40, 15:50] = 255

    rle = mask_to_rle(mask)
    recovered = rle_to_mask(rle)

    np.testing.assert_array_equal(mask > 0, recovered > 0)


# ---------------------------------------------------------------------------
# Scientific caveat tests
# ---------------------------------------------------------------------------


def test_scientific_caveat_present_in_session_report():
    """SessionReport must always include the scientific caveat."""
    from analytics.reporting.session_report import SessionReport

    report = SessionReport()
    assert len(report.scientific_caveat) > 50  # Non-trivial caveat
    assert "THC" in report.scientific_caveat or "cannabinoid" in report.scientific_caveat.lower()


def test_scientific_caveat_in_maturity_summary():
    """MaturitySummary must include scientific caveat."""
    from analytics.reporting.session_report import MaturitySummary

    m = MaturitySummary(clear_fraction=0.2, cloudy_fraction=0.6, amber_fraction=0.2)
    assert "quantif" in m.scientific_caveat.lower() or "NOT" in m.scientific_caveat


def test_harvest_recommendation_no_thc_claims():
    """Harvest recommendations must not claim to predict THC%."""
    from analytics.reporting.session_report import _harvest_recommendation

    for cloudy, amber in [(0.8, 0.1), (0.4, 0.4), (0.1, 0.1), (0.3, 0.35)]:
        rec = _harvest_recommendation(cloudy, amber)
        forbidden = ["THC%", "THC content", "high THC", "potency"]
        for phrase in forbidden:
            assert phrase.lower() not in rec.lower(), (
                f"Found forbidden phrase '{phrase}' in recommendation: {rec}"
            )
