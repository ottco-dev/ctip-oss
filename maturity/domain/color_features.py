"""
maturity.domain.color_features — Color-based trichome maturity feature extraction.

SCIENTIFIC FOUNDATION:
━━━━━━━━━━━━━━━━━━━━━
Trichome color changes are caused by a series of biochemical processes:

1. CLEAR PHASE:
   The trichome head is fully transparent/glassy due to the accumulation
   of terpene precursors and early cannabinoid biosynthesis intermediates.
   The secretory cavity is filling but not yet dense with secondary metabolites.

2. CLOUDY/MILKY PHASE:
   The trichome head becomes opaque/milky as cannabinoid acids (THCA, CBDA)
   and terpenes accumulate, filling the secretory cavity.
   The optical cloudiness is caused by light scattering from the dense
   resinous mixture rather than from direct pigmentation.
   Grower consensus: This is the peak accumulation phase.
   Scientific note: Direct causality (cloudy = maximum THC) is INFERRED
   from phenotypic observation + paired chromatography in a limited number
   of studies. Strain variation is enormous.

3. AMBER PHASE:
   The amber/golden color results from:
   a) THC → CBN degradation via photo-oxidation (UV + heat)
   b) Oxidative polymerization of terpenes → color compounds
   c) Enzymatic browning (similar to fruit ripening)
   Reference: ElSohly, M.A. et al. (2000). "Potency Trends of Δ9-THC."
   Forensic Science International 115:123-134.

4. DEGRADED PHASE:
   Brown-to-black coloration, collapsed heads, burst secretory cavities.
   Advanced oxidation/degradation. Significantly reduced secondary
   metabolite content (both THC and terpenes are volatile/degradable).

WHAT THIS SYSTEM CAN AND CANNOT DO:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CAN:
✓ Classify optical color state with confidence intervals
✓ Estimate relative maturity distribution across a sample
✓ Detect degraded trichomes reliably
✓ Provide explainable feature outputs (which color features drove the classification)
✓ Track color shift trends over time (time series)

CANNOT:
✗ Directly measure THC, CBD, CBN concentrations
✗ Replace GC-MS or HPLC for cannabinoid quantification
✗ Determine exact harvest timing (too many biological variables)
✗ Account for strain-specific color variation
✗ Control for lighting, white balance, or sensor differences

Reference bibliography:
  Potter, D.J. (2009). PhD thesis, King's College London.
  Fischedick, J.T. et al. (2010). Phytochemistry 71(17-18):2058-2073.
  Chandra, S. et al. (2017). Epilepsy & Behavior 70:302-312.
  Tanney, C.A.S. et al. (2021). Front. Plant Sci. 12:815778.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from shared.core.enums import MaturityStage


@dataclass
class ColorFeatureVector:
    """
    Complete color feature vector for maturity classification.

    All features are normalized to [0, 1] range for model compatibility.
    Raw values are preserved for scientific interpretation.
    """

    # HSV features (most important for maturity)
    mean_hue: float          # 0-1 (normalized from 0-180 OpenCV range)
    std_hue: float           # Hue distribution width
    mean_saturation: float   # 0-1
    mean_value: float        # 0-1 (brightness)
    hue_amber_fraction: float  # Fraction of pixels in amber hue range
    hue_clear_fraction: float  # Fraction of pixels in clear/colorless range

    # LAB features (perceptually uniform — better for human-like color comparison)
    mean_l: float            # Lightness 0-1 (normalized from 0-100)
    mean_a: float            # Green-Red axis 0-1 (normalized from -128 to 127)
    mean_b: float            # Blue-Yellow axis 0-1 (normalized from -128 to 127)
    amber_yellowing_score: float  # LAB b* channel score for yellow/amber shift

    # Translucency proxy (unique to microscopy)
    mean_brightness: float   # Overall brightness — clear trichomes are brighter
    contrast: float          # Local contrast — cloudy trichomes have lower contrast
    grayness: float          # How gray vs colored — clear trichomes are less saturated

    # Texture-based features (not in this class, see texture_features.py)

    # Raw values for export
    raw_hue_hist: list[float] | None = None

    @property
    def amber_ratio(self) -> float:
        """Alias for hue_amber_fraction — backward compat with tests."""
        return self.hue_amber_fraction

    @property
    def hue_mean(self) -> float:
        """Alias for mean_hue — backward compat with tests."""
        return self.mean_hue

    @property
    def feature_vector(self) -> NDArray[np.float32]:
        """Return feature vector as numpy array for model input."""
        return np.array([
            self.mean_hue,
            self.std_hue,
            self.mean_saturation,
            self.mean_value,
            self.hue_amber_fraction,
            self.hue_clear_fraction,
            self.mean_l,
            self.mean_a,
            self.mean_b,
            self.amber_yellowing_score,
            self.mean_brightness,
            self.contrast,
            self.grayness,
        ], dtype=np.float32)

    def to_dict(self) -> dict[str, Any]:
        return {
            "mean_hue_norm": self.mean_hue,
            "std_hue_norm": self.std_hue,
            "mean_saturation": self.mean_saturation,
            "mean_value": self.mean_value,
            "hue_amber_fraction": self.hue_amber_fraction,
            "hue_clear_fraction": self.hue_clear_fraction,
            "mean_L_lab": self.mean_l,
            "mean_a_lab": self.mean_a,
            "mean_b_lab": self.mean_b,
            "amber_yellowing_score": self.amber_yellowing_score,
            "brightness": self.mean_brightness,
            "contrast": self.contrast,
            "grayness": self.grayness,
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Hue range definitions for color classification
# Based on OpenCV HSV convention: Hue ∈ [0, 180]
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Amber/orange: Hue 15-35 in OpenCV (30-70 in standard 0-360 range)
AMBER_HUE_LOW = 10
AMBER_HUE_HIGH = 30

# Clear/white: Very low saturation (any hue with S < 30)
CLEAR_SATURATION_THRESHOLD = 30

# Cloudy/milky: High value (bright) + low saturation
CLOUDY_VALUE_THRESHOLD = 180
CLOUDY_SATURATION_MAX = 60

# Degraded/brown: Hue 5-15 with high saturation + low value
DEGRADED_HUE_HIGH = 15
DEGRADED_VALUE_MAX = 150


def extract_color_features(
    trichome_crop: NDArray[np.uint8],
    head_mask: NDArray[np.bool_] | None = None,
) -> ColorFeatureVector:
    """
    Extract color features from a trichome crop image.

    Analysis is performed on the trichome head region only (if mask provided).
    Analyzing the stalk region would introduce noise since stalks are typically
    colorless/transparent regardless of maturity stage.

    Args:
        trichome_crop: RGB crop of the trichome region (H, W, 3) uint8
        head_mask: Optional binary mask for the head region only.
                   If None, analyzes entire crop.

    Returns:
        ColorFeatureVector with all extracted features.
    """
    h, w = trichome_crop.shape[:2]

    # Apply head mask if provided
    if head_mask is not None and head_mask.shape == (h, w):
        pixels = trichome_crop[head_mask]
    else:
        # Fallback: use center region (likely head area)
        margin = max(1, min(h, w) // 6)
        pixels = trichome_crop[margin:-margin, margin:-margin].reshape(-1, 3)

    if len(pixels) == 0:
        return _zero_features()

    # Convert pixels to float for precision
    pixels_f = pixels.astype(np.float32)

    # ── HSV Analysis ────────────────────────────────────────
    pixels_bgr = pixels[:, ::-1]  # RGB → BGR for OpenCV
    pixels_hsv = cv2.cvtColor(pixels_bgr.reshape(1, -1, 3), cv2.COLOR_BGR2HSV)[0]
    hues = pixels_hsv[:, 0].astype(np.float32)      # 0-180
    sats = pixels_hsv[:, 1].astype(np.float32)      # 0-255
    vals = pixels_hsv[:, 2].astype(np.float32)      # 0-255

    mean_hue = float(hues.mean()) / 180.0
    std_hue = float(hues.std()) / 180.0
    mean_sat = float(sats.mean()) / 255.0
    mean_val = float(vals.mean()) / 255.0

    # Amber fraction: pixels in amber hue range with sufficient saturation
    amber_mask = (
        (hues >= AMBER_HUE_LOW) &
        (hues <= AMBER_HUE_HIGH) &
        (sats > 40)
    )
    amber_fraction = float(amber_mask.sum()) / len(pixels)

    # Clear fraction: very low saturation (white/gray/transparent appearance)
    clear_mask = sats < CLEAR_SATURATION_THRESHOLD
    clear_fraction = float(clear_mask.sum()) / len(pixels)

    # ── LAB Analysis ────────────────────────────────────────
    pixels_lab = cv2.cvtColor(pixels_bgr.reshape(1, -1, 3), cv2.COLOR_BGR2LAB)[0]
    l_ch = pixels_lab[:, 0].astype(np.float32)     # 0-255 (OpenCV LAB L range)
    a_ch = pixels_lab[:, 1].astype(np.float32)     # 0-255 (centered at 128)
    b_ch = pixels_lab[:, 2].astype(np.float32)     # 0-255 (centered at 128)

    mean_l = float(l_ch.mean()) / 255.0
    # Normalize a and b to [0,1] (they range 0-255 in OpenCV, centered at 128)
    mean_a = float(a_ch.mean()) / 255.0
    mean_b = float(b_ch.mean()) / 255.0

    # Amber yellowing: high b* (yellow direction) indicates amber shift
    # b* > 128 in OpenCV = yellow-ish
    amber_yellowing = float(np.clip((b_ch.mean() - 128) / 127, 0, 1))

    # ── Brightness/Contrast Analysis ─────────────────────────
    gray_pixels = 0.299 * pixels_f[:, 0] + 0.587 * pixels_f[:, 1] + 0.114 * pixels_f[:, 2]
    mean_brightness = float(gray_pixels.mean()) / 255.0
    contrast = float(gray_pixels.std()) / 128.0

    # Grayness: how close R,G,B are to each other (low = gray/colorless = clear)
    r, g, b_chan = pixels_f[:, 0], pixels_f[:, 1], pixels_f[:, 2]
    color_spread = float(np.mean(np.abs(r - g) + np.abs(g - b_chan) + np.abs(r - b_chan))) / 510.0
    grayness = 1.0 - min(color_spread, 1.0)  # High grayness = colorless = likely clear

    # ── Hue histogram for export ─────────────────────────────
    hue_hist, _ = np.histogram(hues, bins=18, range=(0, 180))
    hue_hist_norm = (hue_hist / max(hue_hist.sum(), 1)).tolist()

    return ColorFeatureVector(
        mean_hue=float(np.clip(mean_hue, 0, 1)),
        std_hue=float(np.clip(std_hue, 0, 1)),
        mean_saturation=float(np.clip(mean_sat, 0, 1)),
        mean_value=float(np.clip(mean_val, 0, 1)),
        hue_amber_fraction=float(np.clip(amber_fraction, 0, 1)),
        hue_clear_fraction=float(np.clip(clear_fraction, 0, 1)),
        mean_l=float(np.clip(mean_l, 0, 1)),
        mean_a=float(np.clip(mean_a, 0, 1)),
        mean_b=float(np.clip(mean_b, 0, 1)),
        amber_yellowing_score=float(np.clip(amber_yellowing, 0, 1)),
        mean_brightness=float(np.clip(mean_brightness, 0, 1)),
        contrast=float(np.clip(contrast, 0, 1)),
        grayness=float(np.clip(grayness, 0, 1)),
        raw_hue_hist=hue_hist_norm,
    )


def rule_based_maturity_estimate(
    features: ColorFeatureVector,
) -> tuple[MaturityStage, float]:
    """
    Heuristic rule-based maturity classification.

    Uses botanically-grounded thresholds derived from literature and
    domain expertise. Intended as:
    1. Baseline for model comparison
    2. Sanity check for ML model outputs
    3. Fallback when no trained model is available

    Returns:
        Tuple of (MaturityStage, confidence estimate in [0,1])

    IMPORTANT: These thresholds are approximate and strain-dependent.
    Calibrate against ground truth data from your specific microscope setup.
    """
    # Strong signals
    # Amber: high amber fraction + low clear fraction + yellowing
    if features.hue_amber_fraction > 0.40 and features.amber_yellowing_score > 0.30:
        conf = min(features.hue_amber_fraction * 1.5, 1.0)
        return MaturityStage.AMBER, conf

    # Clear: very high grayness + low saturation + high brightness
    if (features.hue_clear_fraction > 0.60
            and features.mean_saturation < 0.20
            and features.mean_brightness > 0.70):
        conf = features.hue_clear_fraction
        return MaturityStage.CLEAR, conf

    # Degraded: very low brightness + moderate/high amber hue
    if features.mean_brightness < 0.35 and features.amber_yellowing_score > 0.15:
        return MaturityStage.DEGRADED, 0.65

    # Mixed: significant amber + cloudy present
    if features.hue_amber_fraction > 0.15 and features.mean_saturation > 0.20:
        conf = 0.55
        return MaturityStage.CLOUDY_AMBER_MIX, conf

    # Cloudy: high value, low saturation, not clear (opaque white)
    if (features.mean_value > 0.60
            and features.mean_saturation < 0.35
            and features.hue_clear_fraction < 0.50):
        conf = min(features.mean_value * 1.2, 0.80)
        return MaturityStage.CLOUDY, conf

    # Default fallback
    return MaturityStage.UNKNOWN, 0.30


def _zero_features() -> ColorFeatureVector:
    """Return zero-valued features for empty/invalid regions."""
    return ColorFeatureVector(
        mean_hue=0.0, std_hue=0.0, mean_saturation=0.0, mean_value=0.0,
        hue_amber_fraction=0.0, hue_clear_fraction=0.0,
        mean_l=0.0, mean_a=0.5, mean_b=0.5,
        amber_yellowing_score=0.0, mean_brightness=0.0,
        contrast=0.0, grayness=0.0,
    )
