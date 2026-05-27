"""
tests.unit.test_analytics_api — Unit tests for analytics API endpoints.

Tests:
  - POST /analytics/calibration: ECE computation, input validation
  - POST /analytics/confidence/histogram: distribution stats
  - Interpretation strings for ECE quality tiers
  - BinStats overconfident / underconfident flags
  - Fraction-of-overconfident-bins computation
  - Edge cases: single sample, all correct, all wrong
"""

from __future__ import annotations

import math
from typing import List

import pytest
from fastapi.testclient import TestClient

from analytics.api.router import router, _interpret_ece, _build_bin_stats, _calibration_result_to_response
from analytics.api.schemas import CalibrationRequest, ConfidenceHistogramRequest
from shared.metrics.calibration_metrics import compute_calibration


# ---------------------------------------------------------------------------
# Test client setup
# ---------------------------------------------------------------------------

from fastapi import FastAPI

_app = FastAPI()
_app.include_router(router)
client = TestClient(_app)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _perfectly_overconfident(n: int = 200) -> tuple[List[float], List[bool]]:
    """conf=0.95, always wrong — extreme overconfidence."""
    return [0.95] * n, [False] * n


def _perfectly_underconfident(n: int = 200) -> tuple[List[float], List[bool]]:
    """conf=0.1, always correct — extreme underconfidence."""
    return [0.1] * n, [True] * n


def _well_calibrated(n: int = 1000) -> tuple[List[float], List[bool]]:
    """Produce a roughly calibrated dataset."""
    import numpy as np
    rng = np.random.default_rng(42)
    confs = np.linspace(0.01, 0.99, n)
    correct = rng.random(n) < confs
    return confs.tolist(), correct.tolist()


# ---------------------------------------------------------------------------
# POST /analytics/calibration  — happy path
# ---------------------------------------------------------------------------

class TestCalibrationEndpoint:

    def test_returns_200_with_direct_payload(self):
        confs, correct = _well_calibrated(500)
        resp = client.post("/analytics/calibration", json={
            "confidences": confs,
            "is_correct": correct,
            "num_bins": 10,
        })
        assert resp.status_code == 200

    def test_ece_in_range(self):
        confs, correct = _well_calibrated(500)
        resp = client.post("/analytics/calibration", json={
            "confidences": confs,
            "is_correct": correct,
        })
        data = resp.json()
        assert 0.0 <= data["ece"] <= 1.0

    def test_mce_geq_ece(self):
        confs, correct = _well_calibrated(500)
        data = client.post("/analytics/calibration", json={
            "confidences": confs,
            "is_correct": correct,
        }).json()
        assert data["mce"] >= data["ece"] - 1e-9

    def test_bin_count_matches_num_bins(self):
        confs, correct = _well_calibrated(500)
        data = client.post("/analytics/calibration", json={
            "confidences": confs,
            "is_correct": correct,
            "num_bins": 15,
        }).json()
        assert len(data["bins"]) == 15
        assert data["num_bins"] == 15

    def test_bin_counts_sum_to_total(self):
        n = 300
        confs, correct = _well_calibrated(n)
        data = client.post("/analytics/calibration", json={
            "confidences": confs,
            "is_correct": correct,
        }).json()
        assert sum(b["count"] for b in data["bins"]) == n
        assert data["total_samples"] == n

    def test_overconfident_model_flagged(self):
        confs, correct = _perfectly_overconfident()
        data = client.post("/analytics/calibration", json={
            "confidences": confs,
            "is_correct": correct,
        }).json()
        assert data["is_overconfident"] is True

    def test_underconfident_model_not_flagged(self):
        confs, correct = _perfectly_underconfident()
        data = client.post("/analytics/calibration", json={
            "confidences": confs,
            "is_correct": correct,
        }).json()
        assert data["is_overconfident"] is False

    def test_histogram_matches_bin_counts(self):
        confs, correct = _well_calibrated(500)
        data = client.post("/analytics/calibration", json={
            "confidences": confs,
            "is_correct": correct,
        }).json()
        hist = data["confidence_histogram"]
        bin_counts = [b["count"] for b in data["bins"]]
        assert hist == bin_counts

    def test_interpretation_string_present_and_nonempty(self):
        confs, correct = _well_calibrated(500)
        data = client.post("/analytics/calibration", json={
            "confidences": confs,
            "is_correct": correct,
        }).json()
        assert isinstance(data["interpretation"], str)
        assert len(data["interpretation"]) > 10

    def test_source_is_direct(self):
        confs, correct = _well_calibrated(200)
        data = client.post("/analytics/calibration", json={
            "confidences": confs,
            "is_correct": correct,
        }).json()
        assert data["source"] == "direct"

    def test_bin_gap_equals_conf_minus_acc(self):
        confs, correct = _well_calibrated(300)
        data = client.post("/analytics/calibration", json={
            "confidences": confs,
            "is_correct": correct,
        }).json()
        for b in data["bins"]:
            if not b["is_empty"]:
                expected_gap = b["mean_confidence"] - b["accuracy"]
                assert abs(b["gap"] - expected_gap) < 1e-5

    def test_abs_gap_is_absolute_value_of_gap(self):
        confs, correct = _well_calibrated(300)
        data = client.post("/analytics/calibration", json={
            "confidences": confs,
            "is_correct": correct,
        }).json()
        for b in data["bins"]:
            assert abs(b["abs_gap"] - abs(b["gap"])) < 1e-6

    def test_single_sample(self):
        data = client.post("/analytics/calibration", json={
            "confidences": [0.8],
            "is_correct": [True],
        }).json()
        assert data["total_samples"] == 1
        assert isinstance(data["ece"], float)


# ---------------------------------------------------------------------------
# POST /analytics/calibration — validation errors
# ---------------------------------------------------------------------------

class TestCalibrationValidation:

    def test_empty_payload_returns_422(self):
        resp = client.post("/analytics/calibration", json={})
        assert resp.status_code == 422

    def test_length_mismatch_returns_422(self):
        resp = client.post("/analytics/calibration", json={
            "confidences": [0.8, 0.9],
            "is_correct": [True],
        })
        assert resp.status_code == 422

    def test_confidence_above_1_returns_422(self):
        resp = client.post("/analytics/calibration", json={
            "confidences": [0.5, 1.5],
            "is_correct": [True, False],
        })
        assert resp.status_code == 422

    def test_confidence_below_0_returns_422(self):
        resp = client.post("/analytics/calibration", json={
            "confidences": [-0.1, 0.5],
            "is_correct": [False, True],
        })
        assert resp.status_code == 422

    def test_num_bins_too_small_returns_422(self):
        resp = client.post("/analytics/calibration", json={
            "confidences": [0.5] * 10,
            "is_correct": [True] * 10,
            "num_bins": 2,
        })
        assert resp.status_code == 422

    def test_num_bins_too_large_returns_422(self):
        resp = client.post("/analytics/calibration", json={
            "confidences": [0.5] * 10,
            "is_correct": [True] * 10,
            "num_bins": 100,
        })
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# POST /analytics/confidence/histogram
# ---------------------------------------------------------------------------

class TestConfidenceHistogram:

    def test_returns_200(self):
        resp = client.post("/analytics/confidence/histogram", json={
            "confidences": [0.3, 0.5, 0.7, 0.9, 0.95],
            "num_bins": 10,
        })
        assert resp.status_code == 200

    def test_bin_count_matches_num_bins(self):
        data = client.post("/analytics/confidence/histogram", json={
            "confidences": list(range(10)),  # 0..9, but some > 1 → 422
        })
        # Actually need valid confidences
        import numpy as np
        confs = np.linspace(0.0, 1.0, 50).tolist()
        data = client.post("/analytics/confidence/histogram", json={
            "confidences": confs,
            "num_bins": 20,
        }).json()
        assert len(data["bins"]) == 20

    def test_bin_counts_sum_to_total(self):
        import numpy as np
        confs = np.linspace(0.01, 0.99, 100).tolist()
        data = client.post("/analytics/confidence/histogram", json={
            "confidences": confs,
            "num_bins": 10,
        }).json()
        assert sum(b["count"] for b in data["bins"]) == data["total"]

    def test_mean_confidence_in_range(self):
        import numpy as np
        confs = np.linspace(0.1, 0.9, 200).tolist()
        data = client.post("/analytics/confidence/histogram", json={
            "confidences": confs,
        }).json()
        assert 0.0 <= data["mean_confidence"] <= 1.0

    def test_fraction_high_confidence_bounds(self):
        confs = [0.9, 0.95, 0.3, 0.2]
        data = client.post("/analytics/confidence/histogram", json={
            "confidences": confs,
        }).json()
        # 2 out of 4 are >= 0.70
        assert abs(data["fraction_high_confidence"] - 0.5) < 1e-6

    def test_out_of_range_confidence_returns_422(self):
        resp = client.post("/analytics/confidence/histogram", json={
            "confidences": [0.5, 1.1, 0.3],
        })
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# _interpret_ece — interpretation tier tests
# ---------------------------------------------------------------------------

class TestInterpretEce:

    def test_excellent_ece(self):
        msg = _interpret_ece(0.01, is_overconfident=False)
        assert "Excellent" in msg

    def test_good_ece(self):
        msg = _interpret_ece(0.03, is_overconfident=True)
        assert "Good" in msg

    def test_moderate_ece(self):
        msg = _interpret_ece(0.07, is_overconfident=True)
        assert "Moderate" in msg

    def test_poor_ece(self):
        msg = _interpret_ece(0.15, is_overconfident=True)
        assert "Poor" in msg

    def test_overconfident_direction_in_message(self):
        msg = _interpret_ece(0.06, is_overconfident=True)
        assert "overconfident" in msg.lower()

    def test_underconfident_direction_in_message(self):
        msg = _interpret_ece(0.06, is_overconfident=False)
        assert "underconfident" in msg.lower()
