"""
vlm_labeling.prompts.trichome_prompts — Structured VLM prompt templates.

DESIGN PRINCIPLES:
1. Output MUST be JSON-constrained (parse-able, not free text)
2. Prompts are structured to minimize hallucination
3. Include explicit "unknown" options to prevent forced classification
4. Confidence scoring built into prompt contract
5. Scientific terminology is correct and consistent

HALLUCINATION MITIGATION:
- Request specific observable features, not interpretations
- Provide explicit class lists (VLMs hallucinate less with closed-set prompts)
- Request confidence scores (VLMs with low confidence say so)
- Cross-check with rule-based system (see filtering.hallucination)

HUMAN-IN-LOOP REQUIREMENT:
VLM outputs are PSEUDO-LABELS only.
They MUST go through human review before entering the training dataset.
The system enforces this via AnnotationSource.VLM_AUTO flag.

VLM CAPABILITY NOTES (as of 2025):
- Moondream-2B: Good for binary classification, weak at precise counts
- Florence-2-Large: Excellent for detection + captioning
- Qwen2-VL-7B: Best overall quality but requires 8GB VRAM (quantized)
- MiniCPM-V-2.6: Good quality at 6GB VRAM
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import json


@dataclass
class PromptTemplate:
    """A structured VLM prompt with output schema."""

    name: str
    system_prompt: str
    user_prompt_template: str
    output_schema: dict[str, Any]
    model_compatibility: list[str]
    """Which VLM models this prompt works well with."""

    notes: str = ""

    def format_user_prompt(self, **kwargs: Any) -> str:
        """Fill template with variables."""
        return self.user_prompt_template.format(**kwargs)

    def validate_response(self, response: str) -> tuple[bool, dict[str, Any] | None]:
        """
        Validate and parse VLM response against expected schema.

        Returns:
            (is_valid, parsed_dict) tuple.
        """
        # Try to extract JSON from response (VLMs often add surrounding text)
        json_str = self._extract_json(response)
        if json_str is None:
            return False, None

        try:
            parsed = json.loads(json_str)
        except json.JSONDecodeError:
            return False, None

        # Validate required keys
        for key in self.output_schema.get("required", []):
            if key not in parsed:
                return False, None

        return True, parsed

    @staticmethod
    def _extract_json(text: str) -> str | None:
        """Extract JSON block from VLM response text."""
        # Try direct parse first
        try:
            json.loads(text.strip())
            return text.strip()
        except json.JSONDecodeError:
            pass

        # Look for ```json ... ``` block
        if "```json" in text:
            start = text.find("```json") + 7
            end = text.find("```", start)
            if end > start:
                return text[start:end].strip()

        # Look for { ... } block
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end > start:
            candidate = text[start:end + 1]
            try:
                json.loads(candidate)
                return candidate
            except json.JSONDecodeError:
                pass

        return None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PROMPT TEMPLATES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

MATURITY_CLASSIFICATION_PROMPT = PromptTemplate(
    name="maturity_classification",
    system_prompt=(
        "You are an expert botanist analyzing cannabis trichome microscopy images. "
        "You must classify trichome maturity based ONLY on what you can directly observe. "
        "Do NOT speculate about THC content or potency — that requires chromatography. "
        "Respond with valid JSON only. No additional text."
    ),
    user_prompt_template=(
        "Analyze this cannabis trichome microscopy image.\n\n"
        "Classify the OVERALL maturity stage of the trichome population visible.\n"
        "Choose from:\n"
        "- 'clear': Trichomes are transparent/glassy, still developing\n"
        "- 'cloudy': Trichomes are white/milky/opaque, peak accumulation phase\n"
        "- 'amber': Trichomes show yellow/amber coloration, degradation occurring\n"
        "- 'cloudy_amber_mix': Mixed population of cloudy and amber trichomes\n"
        "- 'degraded': Brown/collapsed trichomes, advanced degradation\n"
        "- 'unknown': Cannot determine from this image quality/angle\n\n"
        "Respond with this exact JSON structure:\n"
        '{{"maturity_stage": "<stage>", '
        '"confidence": <0.0-1.0>, '
        '"amber_fraction_estimate": <0.0-1.0>, '
        '"cloudy_fraction_estimate": <0.0-1.0>, '
        '"clear_fraction_estimate": <0.0-1.0>, '
        '"observations": "<brief description of what you observe>", '
        '"image_quality": "good"|"acceptable"|"poor"}}'
    ),
    output_schema={
        "required": ["maturity_stage", "confidence"],
        "properties": {
            "maturity_stage": {
                "enum": ["clear", "cloudy", "amber", "cloudy_amber_mix", "degraded", "unknown"]
            },
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        }
    },
    model_compatibility=["moondream", "florence2", "qwen2vl", "minicpm"],
    notes="Primary maturity classification prompt. Works across all supported VLMs.",
)


TRICHOME_DETECTION_COUNT_PROMPT = PromptTemplate(
    name="trichome_count",
    system_prompt=(
        "You are analyzing a cannabis microscopy image to count trichome structures. "
        "Be conservative — only count structures you can clearly identify. "
        "Respond with valid JSON only."
    ),
    user_prompt_template=(
        "Count the visible trichome structures in this microscopy image.\n\n"
        "Types to count:\n"
        "- capitate_stalked: Large trichomes with visible stalk and round head (dominant type)\n"
        "- capitate_sessile: Medium trichomes with short/no stalk\n"
        "- bulbous: Tiny round trichomes (<20µm)\n\n"
        "Important: Only count structures clearly visible. "
        "If image is blurry or trichomes are off-frame, note this.\n\n"
        "Respond with:\n"
        '{{"capitate_stalked_count": <integer>, '
        '"capitate_sessile_count": <integer>, '
        '"bulbous_count": <integer>, '
        '"total_visible": <integer>, '
        '"counting_confidence": <0.0-1.0>, '
        '"image_suitable_for_counting": true|false, '
        '"notes": "<any relevant observations>"}}'
    ),
    output_schema={
        "required": ["total_visible", "counting_confidence", "image_suitable_for_counting"],
        "properties": {
            "total_visible": {"type": "integer", "minimum": 0},
            "counting_confidence": {"type": "number", "minimum": 0, "maximum": 1},
        }
    },
    model_compatibility=["qwen2vl", "minicpm", "florence2"],
    notes=(
        "Counting tasks are harder for smaller VLMs. "
        "Moondream tends to overcount. Use qwen2vl or florence2 for counting."
    ),
)


TRICHOME_MORPHOLOGY_PROMPT = PromptTemplate(
    name="morphology_classification",
    system_prompt=(
        "You are a plant biology expert specializing in cannabis glandular trichomes. "
        "Analyze the morphological type of trichomes in microscopy images. "
        "Respond with valid JSON only. No speculation beyond what is visible."
    ),
    user_prompt_template=(
        "Analyze the dominant trichome morphology type visible in this microscopy image.\n\n"
        "Trichome types:\n"
        "- 'capitate_stalked': Has clear stalk + large round head. Most common on calyxes.\n"
        "  Typical total height: 150-500 µm\n"
        "- 'capitate_sessile': Small head, little or no visible stalk.\n"
        "  Typical head diameter: 25-100 µm\n"
        "- 'bulbous': Very small round structures.\n"
        "  Typical head diameter: 10-15 µm\n"
        "- 'non_glandular': Hair-like, no round head, not producing resin\n"
        "- 'mixed': Multiple types visible\n"
        "- 'unknown': Cannot determine clearly\n\n"
        '{{"dominant_type": "<type>", '
        '"confidence": <0.0-1.0>, '
        '"stalk_visible": true|false, '
        '"head_shape": "round"|"elongated"|"irregular"|"not_visible", '
        '"mixed_types_present": true|false, '
        '"estimated_head_diameter_relative": "tiny"|"small"|"medium"|"large"|"very_large"}}'
    ),
    output_schema={
        "required": ["dominant_type", "confidence"],
        "properties": {
            "dominant_type": {
                "enum": [
                    "capitate_stalked", "capitate_sessile", "bulbous",
                    "non_glandular", "mixed", "unknown"
                ]
            }
        }
    },
    model_compatibility=["qwen2vl", "minicpm", "florence2", "moondream"],
    notes="Morphology classification. All VLMs tested, qwen2vl best accuracy.",
)


IMAGE_QUALITY_PROMPT = PromptTemplate(
    name="image_quality",
    system_prompt=(
        "You are a microscopy expert assessing image quality for scientific analysis. "
        "Respond with valid JSON only."
    ),
    user_prompt_template=(
        "Assess the quality of this cannabis trichome microscopy image.\n\n"
        "Evaluate:\n"
        "1. Focus quality: Are trichomes sharply focused?\n"
        "2. Lighting quality: Is exposure appropriate? Any glare or shadows?\n"
        "3. Content suitability: Are trichomes clearly visible and analyzable?\n"
        "4. Artifacts: Any significant artifacts (chromatic aberration, motion blur)?\n\n"
        '{{"overall_quality": "excellent"|"good"|"acceptable"|"poor"|"unusable", '
        '"focus_quality": "sharp"|"slightly_blurry"|"blurry"|"very_blurry", '
        '"lighting_quality": "good"|"overexposed"|"underexposed"|"uneven", '
        '"analyzable": true|false, '
        '"reject_reason": "<if unusable, explain why, else null>", '
        '"confidence": <0.0-1.0>}}'
    ),
    output_schema={
        "required": ["overall_quality", "analyzable", "confidence"],
        "properties": {
            "overall_quality": {
                "enum": ["excellent", "good", "acceptable", "poor", "unusable"]
            }
        }
    },
    model_compatibility=["moondream", "florence2", "qwen2vl", "minicpm"],
    notes="Quality screening prompt. Fast with moondream. Use before expensive analysis.",
)


# Registry of all available prompts
PROMPT_REGISTRY: dict[str, PromptTemplate] = {
    "maturity_classification": MATURITY_CLASSIFICATION_PROMPT,
    "trichome_count": TRICHOME_DETECTION_COUNT_PROMPT,
    "morphology_classification": TRICHOME_MORPHOLOGY_PROMPT,
    "image_quality": IMAGE_QUALITY_PROMPT,
}


def get_prompt(name: str) -> PromptTemplate:
    """Retrieve a prompt template by name."""
    if name not in PROMPT_REGISTRY:
        raise KeyError(
            f"Prompt '{name}' not found. "
            f"Available prompts: {list(PROMPT_REGISTRY.keys())}"
        )
    return PROMPT_REGISTRY[name]


def get_maturity_prompt(image_context: str = "") -> str:
    """
    Return the formatted maturity analysis user prompt.
    
    Args:
        image_context: Optional additional context about the image.
    
    Returns:
        Formatted prompt string ready for VLM input.
    """
    # Try both key names for backward compatibility
    for key in ("maturity_analysis", "maturity_classification"):
        if key in PROMPT_REGISTRY:
            template = get_prompt(key)
            return template.format_user_prompt(image_context=image_context)
    raise KeyError("No maturity prompt found in registry")

