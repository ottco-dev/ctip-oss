"""
shared.core.enums — Domain enumerations.

Scientific rationale for maturity stages:
- Clear/translucent: Trichomes are still accumulating secondary metabolites.
  THC biosynthesis is ongoing. Harvest is premature.
- Cloudy/milky: Maximum THC accumulation phase. Terpene profile is at peak.
  Optimal for potency-focused harvest (literature consensus, though direct
  causal measurement requires chromatography).
- Amber: THC → CBN degradation has begun via oxidation and enzymatic action.
  Sedative effects increase due to CBN accumulation. Preferred by some for
  body-effect profiles.
- Mixed cloudy/amber: Balanced degradation state. Most commercial harvest window.

Reference:
  Potter, D.J. (2009). The propagation, characterisation and optimisation of
  Cannabis sativa L. as a phytopharmaceutical. PhD thesis, King's College London.

  Chandra, S. et al. (2017). Cannabis cultivation: methodological issues for
  obtaining medical-grade product. Epilepsy & Behavior, 70, 302-312.
"""

from enum import Enum, IntEnum, auto


class MaturityStage(str, Enum):
    """
    Trichome maturity stages based on color and translucency.

    Ordered from least mature to most degraded.
    Do NOT use this classification alone to make harvest decisions.
    Biological variability, strain differences, and environmental factors
    all affect the relationship between visual appearance and chemical state.
    """

    CLEAR = "clear"
    """
    Fully translucent/clear trichomes.
    Interpretation: Active biosynthesis phase.
    Visual: Glass-like, fully transparent head.
    Scientific note: Cannot confirm THC level without chromatography.
    """

    CLOUDY = "cloudy"
    """
    Milky/cloudy/opaque trichomes.
    Interpretation: Peak secondary metabolite accumulation (literature consensus).
    Visual: White, opaque, non-transparent head.
    Scientific note: Grower consensus; direct causal link unproven in peer-review
    without paired GC-MS data.
    """

    AMBER = "amber"
    """
    Amber/orange/brown colored trichomes.
    Interpretation: THC → CBN degradation via oxidation.
    Visual: Yellow to dark amber coloration.
    Scientific note: Color change is caused by oxidative degradation.
    CBN content increase confirmed in literature.
    Reference: ElSohly et al. (2000). Potency trends of delta9-THC.
    """

    CLOUDY_AMBER_MIX = "cloudy_amber_mix"
    """
    Mixed population of cloudy and amber trichomes.
    Most common commercial harvest state.
    """

    DEGRADED = "degraded"
    """
    Significantly degraded trichomes.
    Collapsed, burst, or heavily oxidized.
    Indicates post-peak senescence or physical damage.
    """

    UNKNOWN = "unknown"
    """
    Insufficient information to classify.
    Use when confidence < threshold or image quality is poor.
    """


class TrichomeType(str, Enum):
    """
    Botanical trichome morphology classification.

    Based on standard cannabis trichome taxonomy:
    - Glandular trichomes: Produce and store secondary metabolites
    - Non-glandular: Structural, not metabolically active

    Reference:
      Tanney, C.A.S. et al. (2021). Cannabis Glandular Trichomes:
      A Cellular Metabolite Factory. Frontiers in Plant Science, 12, 815778.

      Turner, J.C. et al. (1981). Interrelationships of glandular trichomes
      and cannabinoid content. American Journal of Botany, 68(6), 853-862.
    """

    BULBOUS = "bulbous"
    """
    Smallest glandular type. 10-15 µm diameter.
    Found on all aerial surfaces. Metabolite content minimal.
    Head: 1-4 secretory cells. No visible stalk.
    """

    CAPITATE_SESSILE = "capitate_sessile"
    """
    Medium glandular type. 25-100 µm diameter.
    Short or no stalk. Higher metabolite content than bulbous.
    Found on leaves and bracts.
    """

    CAPITATE_STALKED = "capitate_stalked"
    """
    Largest glandular type. 150-500 µm total height.
    Prominent stalk + large multicellular head.
    Highest cannabinoid and terpene concentration.
    Most abundant on calyxes during flowering.
    Primary target for this analysis system.
    """

    NON_GLANDULAR = "non_glandular"
    """
    Cystolithic/non-glandular trichomes.
    No secretory function. Structural role.
    Should be excluded from maturity analysis.
    """

    UNKNOWN = "unknown"
    """Cannot be classified from current view/resolution."""


class ImageQuality(str, Enum):
    """Image quality assessment for analysis suitability."""

    EXCELLENT = "excellent"
    """Sharp, well-lit, no artifacts. Suitable for all analyses."""

    GOOD = "good"
    """Slightly suboptimal but usable. Most analyses valid."""

    ACCEPTABLE = "acceptable"
    """Usable with caveats. Some analyses may be less reliable."""

    POOR = "poor"
    """Too blurry, over/underexposed, or artifact-heavy. Flagged for review."""

    UNUSABLE = "unusable"
    """Cannot be used. Reject from dataset."""


class AnnotationSource(str, Enum):
    """Source/origin of an annotation."""

    HUMAN_EXPERT = "human_expert"
    """Annotated by a trained human expert. Highest trust."""

    HUMAN_ASSISTED = "human_assisted"
    """Human-corrected AI pre-annotation. High trust."""

    VLM_AUTO = "vlm_auto"
    """Fully automated VLM annotation. Requires human review before use."""

    MODEL_PSEUDO = "model_pseudo"
    """Pseudo-label from trained model. For active learning only."""

    SYNTHETIC = "synthetic"
    """Synthetically generated annotation. Use only for pre-training."""


class ModelBackend(str, Enum):
    """Inference backend selection."""

    PYTORCH = "pytorch"
    ONNX = "onnx"
    TENSORRT = "tensorrt"


class NMSStrategy(str, Enum):
    """Non-Maximum Suppression strategy."""

    STANDARD = "standard"
    """Standard IoU-based NMS."""

    SOFT_NMS = "soft_nms"
    """Soft NMS — reduces scores of overlapping boxes rather than eliminating them.
    Better for dense trichome fields where overlap is biologically expected."""

    WEIGHTED_BOXES_FUSION = "weighted_boxes_fusion"
    """WBF — ensemble fusion approach. Best for multi-model ensembles."""


class AugmentationStrength(IntEnum):
    """Augmentation intensity levels for training."""

    NONE = 0
    LIGHT = 1
    MODERATE = 2
    HEAVY = 3
    EXTREME = 4


class DatasetSplit(str, Enum):
    """Standard dataset splits."""

    TRAIN = "train"
    VALIDATION = "val"
    TEST = "test"
    HOLDOUT = "holdout"  # Never used during development


class ExportFormat(str, Enum):
    """Supported export formats."""

    YOLO = "yolo"
    COCO = "coco"
    PASCAL_VOC = "pascal_voc"
    LABELME = "labelme"
    CVAT_XML = "cvat_xml"
    LABEL_STUDIO_JSON = "label_studio_json"
    FIFTYONE = "fiftyone"
