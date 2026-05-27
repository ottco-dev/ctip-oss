"""
measurement.application.measurement_pipeline — End-to-end trichome measurement pipeline.

Orchestrates:
  1. Profile selection (from ProfileManager)
  2. Geometric feature extraction (from mask)
  3. Stalk/head segmentation
  4. Pixel → µm conversion with uncertainty propagation
  5. Population statistics (mean, median, std across all instances)

INPUT:  List of Instance objects (with masks populated from segmentation pipeline)
OUTPUT: Instance objects with head_diameter_um, stalk_length_um, total_height_um set
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
from numpy.typing import NDArray

from shared.core.entities import Instance
from shared.core.value_objects import CalibrationScale, Micrometer
from measurement.domain.profile_manager import MicroscopeProfile, ProfileManager
from measurement.domain.measurer import Measurer, TrichomeMeasurements
from measurement.domain.propagation import propagate_linear, propagate_area

logger = logging.getLogger(__name__)


@dataclass
class PopulationStats:
    """Population-level statistics across all measured trichomes."""

    n: int = 0
    """Number of successfully measured instances."""

    head_diameter_mean_um: Optional[float] = None
    head_diameter_std_um: Optional[float] = None
    head_diameter_median_um: Optional[float] = None
    head_diameter_iqr_um: Optional[float] = None

    stalk_length_mean_um: Optional[float] = None
    stalk_length_std_um: Optional[float] = None
    stalk_length_median_um: Optional[float] = None

    total_height_mean_um: Optional[float] = None
    total_height_std_um: Optional[float] = None

    head_area_mean_um2: Optional[float] = None
    head_area_std_um2: Optional[float] = None

    head_stalk_ratio_mean: Optional[float] = None
    head_stalk_ratio_std: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "n": self.n,
            "head_diameter_um": {
                "mean": self.head_diameter_mean_um,
                "std": self.head_diameter_std_um,
                "median": self.head_diameter_median_um,
                "iqr": self.head_diameter_iqr_um,
            },
            "stalk_length_um": {
                "mean": self.stalk_length_mean_um,
                "std": self.stalk_length_std_um,
                "median": self.stalk_length_median_um,
            },
            "total_height_um": {
                "mean": self.total_height_mean_um,
                "std": self.total_height_std_um,
            },
            "head_area_um2": {
                "mean": self.head_area_mean_um2,
                "std": self.head_area_std_um2,
            },
            "head_stalk_ratio": {
                "mean": self.head_stalk_ratio_mean,
                "std": self.head_stalk_ratio_std,
            },
        }


@dataclass
class MeasurementPipelineResult:
    """Complete result of the measurement pipeline."""

    instances: List[Instance]
    measurements: List[TrichomeMeasurements]
    population: PopulationStats
    profile: MicroscopeProfile
    total: int = 0
    measured: int = 0
    skipped_no_mask: int = 0
    skipped_too_small: int = 0
    failed: int = 0


class MeasurementPipeline:
    """
    Full trichome measurement pipeline.

    Requires a calibrated MicroscopeProfile.
    All pixel→µm conversions include uncertainty propagation.
    """

    MIN_AREA_PX = 25.0  # Minimum mask area to attempt measurement

    def __init__(
        self,
        profile: Optional[MicroscopeProfile] = None,
        profile_manager: Optional[ProfileManager] = None,
        profile_id: Optional[str] = None,
    ) -> None:
        if profile is not None:
            self.profile = profile
        elif profile_manager is not None:
            p = profile_manager.get_profile(profile_id or "") if profile_id else None
            self.profile = p or profile_manager.default_profile
        else:
            # Use generic 40× profile as fallback
            from measurement.domain.profile_manager import DEFAULT_PROFILES
            self.profile = DEFAULT_PROFILES["40x_generic"]
            logger.warning(
                "No profile provided; using generic 40× profile. "
                "Calibrate your microscope for accurate measurements."
            )

        self._measurer = Measurer(self.profile)
        logger.info(
            f"MeasurementPipeline initialized: profile={self.profile.name}, "
            f"um_per_pixel={self.profile.um_per_pixel:.4f}"
        )

    def measure_instances(
        self,
        instances: List[Instance],
        *,
        focus_score: Optional[float] = None,
    ) -> MeasurementPipelineResult:
        """
        Measure physical dimensions of all instances in the list.

        Populates Instance.head_diameter_um, stalk_length_um,
        total_height_um, and calibration_scale.

        Args:
            instances:    List of Instance objects with masks.
            focus_score:  Optional image focus quality [0,1] for uncertainty scaling.

        Returns:
            MeasurementPipelineResult with all measurements and population stats.
        """
        measurements: List[TrichomeMeasurements] = []
        skipped_no_mask = 0
        skipped_small = 0
        failed = 0

        for inst in instances:
            if inst.mask is None:
                skipped_no_mask += 1
                continue

            mask_arr: NDArray[np.uint8] = (
                inst.mask.data if hasattr(inst.mask, "data") else np.array(inst.mask)
            )

            if mask_arr.ndim != 2:
                skipped_no_mask += 1
                continue

            area_px = float(mask_arr.sum() / 255) if mask_arr.max() > 1 else float(mask_arr.sum())
            if area_px < self.MIN_AREA_PX:
                skipped_small += 1
                continue

            try:
                m = self._measure_one(mask_arr, focus_score=focus_score)
                measurements.append(m)

                # Write back to Instance
                if m.head_diameter_um is not None:
                    inst.head_diameter_um = Micrometer(m.head_diameter_um)
                if m.stalk_length_um is not None:
                    inst.stalk_length_um = Micrometer(m.stalk_length_um)
                if m.total_height_um is not None:
                    inst.total_height_um = Micrometer(m.total_height_um)

                inst.calibration_scale = CalibrationScale(
                    um_per_pixel=self.profile.um_per_pixel,
                    uncertainty_um_per_pixel=self.profile.uncertainty_um or 0.0,
                    source=self.profile.calibration_method or "profile",
                )

            except Exception as e:
                logger.warning(f"Measurement failed for {inst.id[:8]}: {e}")
                failed += 1

        population = _compute_population_stats(measurements)

        return MeasurementPipelineResult(
            instances=instances,
            measurements=measurements,
            population=population,
            profile=self.profile,
            total=len(instances),
            measured=len(measurements),
            skipped_no_mask=skipped_no_mask,
            skipped_too_small=skipped_small,
            failed=failed,
        )

    def _measure_one(
        self,
        mask: NDArray[np.uint8],
        focus_score: Optional[float] = None,
    ) -> TrichomeMeasurements:
        """Measure a single trichome mask."""
        from morphology.domain.geometric import extract_geometric_descriptors
        from morphology.domain.stalk_detector import detect_stalk_and_head

        geo = extract_geometric_descriptors(mask)
        stalk, head = detect_stalk_and_head(mask)

        focus_unc_px = 0.0
        if focus_score is not None:
            from measurement.domain.propagation import focus_induced_uncertainty
            focus_unc_px = focus_induced_uncertainty(
                focus_score, self.profile.um_per_pixel
            )

        cal_unc = self.profile.uncertainty_um or 0.0

        return self._measurer.measure(
            head_diameter_px=head.head_diameter_px if head else None,
            head_area_px=head.head_area_px if head else None,
            head_circularity=head.head_circularity if head else geo.circularity,
            stalk_length_px=stalk.stalk_length_px if stalk.has_visible_stalk else None,
            stalk_width_px=stalk.stalk_width_px if stalk.has_visible_stalk else None,
            total_height_px=geo.major_axis_px,
            total_area_px=geo.area_px,
        )


def _compute_population_stats(
    measurements: List[TrichomeMeasurements],
) -> PopulationStats:
    """Compute population-level statistics from a list of measurements."""
    if not measurements:
        return PopulationStats(n=0)

    def _stats(values: list[float]) -> tuple:
        if not values:
            return None, None, None, None
        arr = np.array(values)
        q25, q75 = np.percentile(arr, [25, 75])
        return (
            float(arr.mean()),
            float(arr.std()),
            float(np.median(arr)),
            float(q75 - q25),
        )

    head_ds = [m.head_diameter_um for m in measurements if m.head_diameter_um is not None]
    stalk_ls = [m.stalk_length_um for m in measurements if m.stalk_length_um is not None]
    heights = [m.total_height_um for m in measurements if m.total_height_um is not None]
    head_as = [m.head_area_um2 for m in measurements if m.head_area_um2 is not None]
    ratios = [m.head_stalk_ratio for m in measurements if m.head_stalk_ratio is not None]

    hd_mean, hd_std, hd_med, hd_iqr = _stats(head_ds)
    sl_mean, sl_std, sl_med, _ = _stats(stalk_ls)
    ht_mean, ht_std, _, _ = _stats(heights)
    ha_mean, ha_std, _, _ = _stats(head_as)
    r_mean, r_std, _, _ = _stats(ratios)

    return PopulationStats(
        n=len(measurements),
        head_diameter_mean_um=hd_mean,
        head_diameter_std_um=hd_std,
        head_diameter_median_um=hd_med,
        head_diameter_iqr_um=hd_iqr,
        stalk_length_mean_um=sl_mean,
        stalk_length_std_um=sl_std,
        stalk_length_median_um=sl_med,
        total_height_mean_um=ht_mean,
        total_height_std_um=ht_std,
        head_area_mean_um2=ha_mean,
        head_area_std_um2=ha_std,
        head_stalk_ratio_mean=r_mean,
        head_stalk_ratio_std=r_std,
    )
