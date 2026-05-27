"""
measurement.domain.propagation — Uncertainty propagation for trichome measurements.

Implements GUM (Guide to the Expression of Uncertainty in Measurement) methods
for combining measurement uncertainties from multiple sources.

UNCERTAINTY SOURCES IN TRICHOME MEASUREMENT:
1. Calibration uncertainty (σ_cal): from stage micrometer precision
2. Pixel edge uncertainty (σ_edge): ≈1 pixel from mask boundary detection
3. Focus uncertainty (σ_focus): blur causes edge position uncertainty
4. Sampling uncertainty (σ_sample): biological variability across trichomes

PROPAGATION LAW (GUM §5.1):
For y = f(x₁, x₂, ..., xₙ):
  u²(y) = Σᵢ (∂f/∂xᵢ)² · u²(xᵢ) + 2 Σᵢ<ⱼ (∂f/∂xᵢ)(∂f/∂xⱼ) · u(xᵢ,xⱼ)

For linear measurements y = c · x:
  u(y) = |c| · u(x)

For area measurements y = x²:
  u(y) = 2x · u(x)

Reference:
  JCGM 100:2008. Evaluation of measurement data —
  Guide to the Expression of Uncertainty in Measurement (GUM).
  BIPM/IEC/IFCC/ILAC/ISO/IUPAC/IUPAP/OIML.
  https://www.bipm.org/documents/20126/2071204/JCGM_100_2008_E.pdf
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional


@dataclass
class MeasurementWithUncertainty:
    """A single measurement with its combined ±1σ uncertainty."""

    value: float
    """Measured value in the stated unit."""

    uncertainty: float
    """Combined ±1σ standard uncertainty (same unit as value)."""

    unit: str = ""
    """Physical unit string for display (e.g., 'µm', 'µm²')."""

    coverage_factor: float = 2.0
    """
    k factor for expanded uncertainty U = k × u.
    k=2 corresponds to ~95% confidence for Gaussian distributions.
    """

    @property
    def expanded_uncertainty(self) -> float:
        """Expanded uncertainty U = k × u at ~95% confidence."""
        return self.coverage_factor * self.uncertainty

    @property
    def relative_uncertainty(self) -> float:
        """Relative uncertainty u/value. Dimensionless."""
        if abs(self.value) < 1e-12:
            return float("inf")
        return self.uncertainty / abs(self.value)

    def __repr__(self) -> str:
        return (
            f"{self.value:.3f} ± {self.uncertainty:.3f} {self.unit} "
            f"(k={self.coverage_factor})"
        )


def combine_uncertainties(*uncertainties: float) -> float:
    """
    Combine independent uncertainties in quadrature (GUM §5.1.2).

    u_combined = √(u₁² + u₂² + ... + uₙ²)
    """
    return math.sqrt(sum(u ** 2 for u in uncertainties))


def propagate_linear(
    value_px: float,
    um_per_pixel: float,
    *,
    calibration_uncertainty_um: float = 0.0,
    edge_uncertainty_px: float = 1.0,
    focus_uncertainty_px: float = 0.0,
) -> MeasurementWithUncertainty:
    """
    Propagate uncertainty for a linear pixel measurement converted to µm.

    Model: d_um = d_px × um_per_pixel

    Uncertainty sources:
      σ_cal   = calibration_uncertainty_um × d_px  (calibration error)
      σ_edge  = edge_uncertainty_px × um_per_pixel  (1-pixel edge error)
      σ_focus = focus_uncertainty_px × um_per_pixel (blur-induced error)

    Args:
        value_px:                  Measured distance in pixels.
        um_per_pixel:              Calibration factor µm/px.
        calibration_uncertainty_um: σ of the calibration factor itself (µm/px).
        edge_uncertainty_px:       Pixel edge detection uncertainty (default 1 px).
        focus_uncertainty_px:      Additional uncertainty from blur (px).

    Returns:
        MeasurementWithUncertainty in µm.
    """
    value_um = value_px * um_per_pixel

    sigma_cal = calibration_uncertainty_um * value_px
    sigma_edge = edge_uncertainty_px * um_per_pixel
    sigma_focus = focus_uncertainty_px * um_per_pixel

    combined_u = combine_uncertainties(sigma_cal, sigma_edge, sigma_focus)

    return MeasurementWithUncertainty(
        value=value_um,
        uncertainty=combined_u,
        unit="µm",
    )


def propagate_area(
    area_px2: float,
    um_per_pixel: float,
    *,
    calibration_uncertainty_um: float = 0.0,
    edge_uncertainty_px: float = 1.0,
) -> MeasurementWithUncertainty:
    """
    Propagate uncertainty for an area measurement.

    Model: A_um2 = A_px × um_per_pixel²

    For area, the uncertainty is approximately:
      u(A_um2) ≈ 2 × √(A_px) × um_per_pixel × u(side)

    This uses the perimeter-based approximation: area error ∝ √(area) × edge_error.

    Args:
        area_px2:                  Area in pixels².
        um_per_pixel:              Calibration factor µm/px.
        calibration_uncertainty_um: Calibration uncertainty σ in µm/px.
        edge_uncertainty_px:       Edge detection uncertainty in pixels.

    Returns:
        MeasurementWithUncertainty in µm².
    """
    value_um2 = area_px2 * (um_per_pixel ** 2)

    # Side length uncertainty: approximate side ≈ √area
    side_px = math.sqrt(max(area_px2, 1))
    sigma_cal = calibration_uncertainty_um * side_px
    sigma_edge = edge_uncertainty_px * um_per_pixel

    # Area uncertainty = 2 × side × side_uncertainty
    sigma_area_um2 = 2.0 * side_px * um_per_pixel * combine_uncertainties(
        sigma_cal, sigma_edge
    )

    return MeasurementWithUncertainty(
        value=value_um2,
        uncertainty=sigma_area_um2,
        unit="µm²",
    )


def propagate_ratio(
    numerator: MeasurementWithUncertainty,
    denominator: MeasurementWithUncertainty,
) -> MeasurementWithUncertainty:
    """
    Propagate uncertainty for a ratio r = a / b.

    GUM formula:
      (u(r)/r)² = (u(a)/a)² + (u(b)/b)²

    Args:
        numerator:   Measurement with uncertainty for the numerator.
        denominator: Measurement with uncertainty for the denominator.

    Returns:
        MeasurementWithUncertainty for the ratio (dimensionless).
    """
    if abs(denominator.value) < 1e-12:
        return MeasurementWithUncertainty(value=float("inf"), uncertainty=float("inf"))

    ratio = numerator.value / denominator.value
    rel_u = combine_uncertainties(
        numerator.relative_uncertainty,
        denominator.relative_uncertainty,
    )
    sigma_ratio = abs(ratio) * rel_u

    return MeasurementWithUncertainty(
        value=ratio,
        uncertainty=sigma_ratio,
        unit="",
    )


def focus_induced_uncertainty(
    focus_score: float,
    pixel_size_um: float,
    *,
    max_blur_px: float = 3.0,
) -> float:
    """
    Estimate edge uncertainty contribution from image blur.

    As focus score drops, the apparent edge position becomes less precise.
    Maps focus_score [0,1] → additional_uncertainty_px [0, max_blur_px].

    Args:
        focus_score:   Image focus quality [0,1]. 1=perfectly sharp.
        pixel_size_um: Calibration factor µm/px.
        max_blur_px:   Maximum blur-induced uncertainty at focus_score=0.

    Returns:
        Edge uncertainty in pixels due to defocus.
    """
    # Inverse: high score → low blur uncertainty
    blur_px = max_blur_px * (1.0 - max(0.0, min(1.0, focus_score)))
    return blur_px
