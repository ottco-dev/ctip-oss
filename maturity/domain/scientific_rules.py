"""
maturity.domain.scientific_rules — Epistemic constraints for trichome maturity claims.

This module encodes the BOUNDARIES of what we can and cannot scientifically claim
based on optical microscopy of trichome color/morphology.

PURPOSE:
These rules prevent the system from making overconfident claims that go beyond
what the optical evidence supports. Every output should be appropriately hedged.

EPISTEMIC LEVELS:
1. DIRECTLY OBSERVABLE (high confidence):
   - Trichome color (optical state)
   - Structural morphology (size, shape, type)
   - Population distribution (% clear, cloudy, amber)

2. INFERRED WITH SCIENTIFIC SUPPORT (medium confidence, cite literature):
   - Color state ↔ maturation stage (supported by studies with caveats)
   - Amber color ↔ oxidative degradation (well-established chemistry)

3. UNCERTAIN / REQUIRES ADDITIONAL METHODS (low confidence, explicit caveat):
   - Color ↔ THC concentration (highly variable, strain-dependent)
   - "Optimal harvest" timing (biological complexity)
   - Specific cannabinoid ratios (requires chromatography)

SCIENTIFIC REFERENCES ENCODED IN RULES:
  [1] Fischedick, J.T. et al. (2010). Phytochemistry 71:2058-2073.
  [2] Potter, D.J. (2009). PhD thesis, King's College London.
  [3] ElSohly, M.A. et al. (2000). Forensic Sci. Int. 115:123-134.
  [4] Tanney, C.A.S. et al. (2021). Front. Plant Sci. 12:815778.
  [5] Chandra, S. et al. (2017). Epilepsy & Behavior 70:302-312.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import NamedTuple

from shared.core.enums import MaturityStage


class ClaimStrength(str, Enum):
    """Epistemic strength of a scientific claim."""
    DIRECTLY_OBSERVABLE = "directly_observable"
    INFERRED = "inferred"
    SPECULATIVE = "speculative"
    PROHIBITED = "prohibited"


@dataclass(frozen=True)
class ScientificClaim:
    """A claim with its epistemic strength and supporting evidence."""
    statement: str
    strength: ClaimStrength
    caveat: str | None
    reference: str | None
    can_report: bool = True


# ─────────────────────────────────────────────────────────────────────────────
# APPROVED CLAIMS — what we CAN say
# ─────────────────────────────────────────────────────────────────────────────

APPROVED_CLAIMS: dict[str, ScientificClaim] = {
    "color_state": ScientificClaim(
        statement="The optical color state of trichome heads has been classified.",
        strength=ClaimStrength.DIRECTLY_OBSERVABLE,
        caveat=None,
        reference=None,
    ),

    "population_distribution": ScientificClaim(
        statement="The proportion of trichomes in each color state has been measured.",
        strength=ClaimStrength.DIRECTLY_OBSERVABLE,
        caveat="Proportions are from the analyzed image area. "
               "Full-sample representativeness depends on sampling protocol.",
        reference=None,
    ),

    "morphology": ScientificClaim(
        statement="Trichome type (stalked bulbous, sessile, capitate) has been classified.",
        strength=ClaimStrength.DIRECTLY_OBSERVABLE,
        caveat="Classification accuracy depends on image resolution and focus quality.",
        reference=None,
    ),

    "color_maturation_correlation": ScientificClaim(
        statement="Clear → cloudy transition correlates with cannabinoid accumulation "
                  "in multiple studied cultivars.",
        strength=ClaimStrength.INFERRED,
        caveat="Correlation strength varies significantly by cultivar, "
               "environmental conditions, and tissue age. "
               "Not a universal biological law.",
        reference="Fischedick et al. (2010), Phytochemistry 71:2058-2073; "
                  "Potter (2009), PhD thesis, King's College London.",
    ),

    "amber_oxidation": ScientificClaim(
        statement="Amber/brown coloration is consistent with THC→CBN oxidative degradation "
                  "and terpene polymerization.",
        strength=ClaimStrength.INFERRED,
        caveat="Amber coloration may also result from strain-specific pigmentation "
               "or lighting artifacts. Chemical confirmation requires chromatography.",
        reference="ElSohly et al. (2000), Forensic Sci. Int. 115:123-134.",
    ),

    "degradation_assessment": ScientificClaim(
        statement="Trichomes showing brown/black coloration or structural deformation "
                  "are classified as degraded.",
        strength=ClaimStrength.DIRECTLY_OBSERVABLE,
        caveat="Degradation classification is based on visual indicators only.",
        reference=None,
    ),

    "size_measurement": ScientificClaim(
        statement="Trichome head diameter and stalk length can be measured in pixels "
                  "and converted to µm using calibration data.",
        strength=ClaimStrength.DIRECTLY_OBSERVABLE,
        caveat="Accuracy depends on calibration quality and image magnification accuracy.",
        reference=None,
    ),
}


# ─────────────────────────────────────────────────────────────────────────────
# PROHIBITED CLAIMS — what we CANNOT say
# ─────────────────────────────────────────────────────────────────────────────

PROHIBITED_CLAIMS: dict[str, ScientificClaim] = {
    "thc_concentration": ScientificClaim(
        statement="THC/CBD/CBN concentration cannot be determined from color alone.",
        strength=ClaimStrength.PROHIBITED,
        caveat="Color is an INDIRECT optical proxy. Many factors (genetics, environment, "
               "post-harvest handling) affect the color-potency relationship. "
               "Use GC-MS or HPLC for quantification.",
        reference="Tanney et al. (2021), Front. Plant Sci. 12:815778.",
        can_report=False,
    ),

    "optimal_harvest_timing": ScientificClaim(
        statement="Optimal harvest timing cannot be precisely determined from trichome color alone.",
        strength=ClaimStrength.PROHIBITED,
        caveat="'Optimal' depends on desired effect profile, cultivar, processing method, "
               "and individual user goals. Trichome color is one of several indicators. "
               "This system provides objective color data, not harvest recommendations.",
        reference="Potter (2009), PhD thesis.",
        can_report=False,
    ),

    "cannabinoid_ratio": ScientificClaim(
        statement="THC:CBD or other cannabinoid ratios cannot be inferred from microscopy.",
        strength=ClaimStrength.PROHIBITED,
        caveat="Different cannabinoids are biosynthetically related but not optically distinguishable.",
        reference="Chandra et al. (2017), Epilepsy & Behavior 70:302-312.",
        can_report=False,
    ),

    "strain_identification": ScientificClaim(
        statement="Strain/cultivar identification is not possible from trichome morphology.",
        strength=ClaimStrength.PROHIBITED,
        caveat="Trichome morphology varies within strains and overlaps between strains. "
               "Genetic testing is required for cultivar identification.",
        reference=None,
        can_report=False,
    ),
}


# ─────────────────────────────────────────────────────────────────────────────
# Stage-specific caveats
# ─────────────────────────────────────────────────────────────────────────────

STAGE_CAVEATS: dict[MaturityStage, str] = {
    MaturityStage.CLEAR: (
        "Clear trichomes indicate early maturation stage. "
        "Cannabinoid biosynthesis is ongoing. "
        "THC accumulation is generally low but strain-dependent. [Ref: Fischedick 2010]"
    ),
    MaturityStage.CLOUDY: (
        "Cloudy/milky trichomes indicate active cannabinoid accumulation. "
        "Multiple studies suggest this phase correlates with near-peak THCA content "
        "in many cultivars, but significant strain variation exists. "
        "Direct THC measurement requires chromatographic analysis."
    ),
    MaturityStage.AMBER: (
        "Amber coloration is consistent with oxidative processes including THC→CBN conversion. "
        "Extent of chemical change cannot be quantified from color alone. "
        "Some amber development is expected and normal; "
        "the significance depends on overall population distribution."
    ),
    MaturityStage.DEGRADED: (
        "Structural degradation detected. This may indicate post-harvest deterioration, "
        "mechanical damage, or advanced oxidation. "
        "Analytical testing is recommended to assess sample quality."
    ),
    MaturityStage.UNKNOWN: (
        "Classification confidence is insufficient for reliable maturity assessment. "
        "Image quality, lighting conditions, or atypical morphology may be contributing factors."
    ),
}


# ─────────────────────────────────────────────────────────────────────────────
# Validation and enforcement functions
# ─────────────────────────────────────────────────────────────────────────────

def validate_claim(claim_key: str) -> tuple[bool, ScientificClaim | None, str]:
    """
    Validate whether a claim is scientifically appropriate to make.

    Args:
        claim_key: Key from APPROVED_CLAIMS or PROHIBITED_CLAIMS

    Returns:
        (is_allowed, claim_object, reason_message)
    """
    if claim_key in APPROVED_CLAIMS:
        claim = APPROVED_CLAIMS[claim_key]
        return True, claim, f"Approved: {claim.strength.value}"

    if claim_key in PROHIBITED_CLAIMS:
        claim = PROHIBITED_CLAIMS[claim_key]
        return False, claim, f"Prohibited: {claim.caveat}"

    return False, None, f"Unknown claim key: '{claim_key}'. Not in approved or prohibited lists."


def get_stage_caveat(stage: MaturityStage) -> str:
    """Return the scientific caveat for a given maturity stage."""
    return STAGE_CAVEATS.get(stage, STAGE_CAVEATS[MaturityStage.UNKNOWN])


def get_report_preamble() -> str:
    """
    Standard scientific preamble for all maturity analysis reports.
    Should be included at the top of every analysis output.
    """
    return (
        "SCIENTIFIC DISCLOSURE: This analysis classifies the OPTICAL COLOR STATE "
        "of trichome heads observed under microscopy. It does not measure, estimate, "
        "or infer cannabinoid (THC, CBD, CBN) concentrations, ratios, or potency. "
        "Color classification is an indirect proxy for biochemical state, subject to "
        "significant biological variability. For quantitative cannabinoid analysis, "
        "use GC-MS or HPLC chromatography."
    )


def check_confidence_threshold(
    confidence: float,
    stage: MaturityStage,
    min_confidence: float = 0.55,
) -> tuple[MaturityStage, str]:
    """
    Apply minimum confidence threshold — fall back to UNKNOWN if too uncertain.

    Args:
        confidence: Model confidence score [0, 1]
        stage: Predicted maturity stage
        min_confidence: Threshold below which stage is set to UNKNOWN

    Returns:
        (final_stage, caveat_message)
    """
    if confidence < min_confidence:
        caveat = (
            f"Confidence ({confidence:.2f}) below threshold ({min_confidence:.2f}). "
            f"Stage classified as UNKNOWN. Original prediction: {stage.value}."
        )
        return MaturityStage.UNKNOWN, caveat

    caveat = get_stage_caveat(stage)
    return stage, caveat


def distribution_summary_statement(
    clear_pct: float,
    cloudy_pct: float,
    amber_pct: float,
    degraded_pct: float,
) -> str:
    """
    Generate a scientifically appropriate summary statement for a distribution.

    Args:
        clear_pct, cloudy_pct, amber_pct, degraded_pct: Percentages (0-100)

    Returns:
        Summary string with appropriate hedging
    """
    parts = []

    if cloudy_pct >= 60:
        parts.append(
            f"{cloudy_pct:.0f}% of examined trichomes appear cloudy/opaque, "
            "consistent with active cannabinoid accumulation in many cultivars"
        )
    if amber_pct >= 20:
        parts.append(
            f"{amber_pct:.0f}% show amber coloration, "
            "consistent with oxidative biochemical changes"
        )
    if clear_pct >= 30:
        parts.append(
            f"{clear_pct:.0f}% appear clear/transparent, "
            "suggesting continued maturation is ongoing"
        )
    if degraded_pct >= 15:
        parts.append(
            f"{degraded_pct:.0f}% show degradation indicators — "
            "sample quality review recommended"
        )

    if not parts:
        parts.append("Mixed trichome population observed. Distribution appears unremarkable.")

    summary = ". ".join(parts) + "."
    return summary + " " + get_report_preamble()
