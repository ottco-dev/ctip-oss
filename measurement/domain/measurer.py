"""
measurement.domain.measurer — Physical measurement of trichome dimensions.

Converts pixel-space geometric measurements to calibrated physical units (µm)
using a MicroscopeProfile.

MEASUREMENTS PERFORMED:
- Head diameter (µm): diameter of the secretory gland head
- Stalk length (µm): length of the peduncular stalk
- Total height (µm): base-to-apex distance of the full trichome
- Head area (µm²): area of the secretory head region
- Head/stalk ratio: dimensionless ratio, biologically diagnostic

UNCERTAINTY PROPAGATION:
Each measurement carries a ±σ uncertainty derived from:
  1. Calibration uncertainty (from stage micrometer or profile spec)
  2. Mask measurement uncertainty (edge detection error ≈ 1 pixel)

Formula:
  σ_dimension = √[(σ_calibration × dimension_px)² + (1px × µm_per_px)²]

Reference:
  Joint Committee for Guides in Metrology (JCGM 100:2008).
  Evaluation of measurement data — Guide to the Expression of
  Uncertainty in Measurement (GUM). BIPM, Sèvres, France.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from measurement.domain.profile_manager import MicroscopeProfile


@dataclass
class TrichomeMeasurements:
    """
    Calibrated physical measurements of a single trichome instance.

    All dimensions in micrometers (µm). Uncertainties are ±1σ estimates.
    """

    # --- Head ---
    head_diameter_um: Optional[float]
    """Equivalent circle diameter of the secretory gland head."""

    head_area_um2: Optional[float]
    """Area of the secretory gland head in µm²."""

    head_circularity: Optional[float]
    """Dimensionless circularity [0,1]. Not affected by calibration."""

    # --- Stalk ---
    stalk_length_um: Optional[float]
    """Length of the peduncular stalk from base to head junction."""

    stalk_width_um: Optional[float]
    """Mean width of the stalk region."""

    # --- Full trichome ---
    total_height_um: Optional[float]
    """Full trichome height (base to apex), from major axis length."""

    total_area_um2: Optional[float]
    """Total mask area in µm²."""

    # --- Derived ---
    head_stalk_ratio: Optional[float]
    """head_diameter / stalk_length. Dimensionless. Diagnostic for type."""

    # --- Uncertainties (±1σ) ---
    head_diameter_uncertainty_um: Optional[float] = None
    stalk_length_uncertainty_um: Optional[float] = None

    # --- Calibration metadata ---
    profile_id: str = ""
    um_per_pixel: float = 1.0
    calibration_method: str = "unknown"

    @property
    def is_stalked(self) -> bool:
        """True if the stalk is clearly visible (>30 µm)."""
        return self.stalk_length_um is not None and self.stalk_length_um > 30.0

    @property
    def morphology_hint(self) -> str:
        """
        Rough morphological type hint from measurements.
        NOT a definitive classification — use MorphologyClassifier.
        """
        if self.head_diameter_um is None:
            return "unknown"
        if self.head_diameter_um < 20:
            return "bulbous"
        if self.is_stalked and self.head_diameter_um > 50:
            return "capitate_stalked"
        return "capitate_sessile"

    def to_dict(self) -> dict:
        return {
            "head_diameter_um": self.head_diameter_um,
            "head_area_um2": self.head_area_um2,
            "head_circularity": self.head_circularity,
            "stalk_length_um": self.stalk_length_um,
            "stalk_width_um": self.stalk_width_um,
            "total_height_um": self.total_height_um,
            "total_area_um2": self.total_area_um2,
            "head_stalk_ratio": self.head_stalk_ratio,
            "uncertainties": {
                "head_diameter_um": self.head_diameter_uncertainty_um,
                "stalk_length_um": self.stalk_length_uncertainty_um,
            },
            "calibration": {
                "profile_id": self.profile_id,
                "um_per_pixel": self.um_per_pixel,
                "method": self.calibration_method,
            },
        }


class Measurer:
    """
    Converts pixel-space trichome measurements to physical units.

    Requires a MicroscopeProfile with calibrated um_per_pixel.
    """

    def __init__(self, profile: MicroscopeProfile) -> None:
        if profile.um_per_pixel <= 0:
            raise ValueError("Profile um_per_pixel must be positive")
        self.profile = profile

    def measure(
        self,
        *,
        head_diameter_px: Optional[float] = None,
        head_area_px: Optional[float] = None,
        head_circularity: Optional[float] = None,
        stalk_length_px: Optional[float] = None,
        stalk_width_px: Optional[float] = None,
        total_height_px: Optional[float] = None,
        total_area_px: Optional[float] = None,
    ) -> TrichomeMeasurements:
        """
        Convert pixel measurements to physical µm measurements.

        Any None inputs produce None outputs. All conversions use the
        profile's um_per_pixel factor with uncertainty propagation.

        Args:
            head_diameter_px:  Head equivalent circle diameter in pixels.
            head_area_px:      Head area in pixels².
            head_circularity:  Head circularity [0,1] (dimensionless, passed through).
            stalk_length_px:   Stalk length in pixels.
            stalk_width_px:    Stalk mean width in pixels.
            total_height_px:   Full trichome height in pixels.
            total_area_px:     Total mask area in pixels².

        Returns:
            TrichomeMeasurements with all values in µm/µm².
        """
        u = self.profile.um_per_pixel
        cal_unc = self.profile.uncertainty_um or 0.0

        def _linear(px: Optional[float]) -> Optional[float]:
            return px * u if px is not None else None

        def _area(px2: Optional[float]) -> Optional[float]:
            return px2 * (u ** 2) if px2 is not None else None

        def _uncertainty(px: Optional[float]) -> Optional[float]:
            """±1σ uncertainty for a linear measurement in µm."""
            if px is None:
                return None
            # Calibration uncertainty + 1-pixel edge uncertainty
            sigma_cal = cal_unc * px       # σ_cal × dimension_px
            sigma_edge = u                 # 1 pixel × µm/px
            return math.sqrt(sigma_cal ** 2 + sigma_edge ** 2)

        head_d_um = _linear(head_diameter_px)
        stalk_l_um = _linear(stalk_length_px)

        head_stalk_ratio: Optional[float] = None
        if head_d_um is not None and stalk_l_um is not None and stalk_l_um > 0:
            head_stalk_ratio = head_d_um / stalk_l_um

        return TrichomeMeasurements(
            head_diameter_um=head_d_um,
            head_area_um2=_area(head_area_px),
            head_circularity=head_circularity,
            stalk_length_um=stalk_l_um,
            stalk_width_um=_linear(stalk_width_px),
            total_height_um=_linear(total_height_px),
            total_area_um2=_area(total_area_px),
            head_stalk_ratio=head_stalk_ratio,
            head_diameter_uncertainty_um=_uncertainty(head_diameter_px),
            stalk_length_uncertainty_um=_uncertainty(stalk_length_px),
            profile_id=self.profile.profile_id,
            um_per_pixel=u,
            calibration_method=self.profile.calibration_method,
        )
