"""
vlm_labeling.florence2.florence_labeler — Florence-2-Large VLM backend.

Florence-2-Large (Microsoft): ~770M parameters, 8 GB VRAM (fp16).
Handles structured detection + captioning tasks natively.

Capabilities used here:
- <CAPTION> — dense image description
- <DETAILED_CAPTION> — structured observation
- <REGION_TO_DESCRIPTION> — describe individual trichome region
- <OPEN_VOCABULARY_DETECTION> — detect objects by text query
- Custom prompt engineering to extract maturity/morphology labels

VRAM requirements:
- fp16: ~3.5 GB (Florence-2-Large)
- fp32: ~7.0 GB (too large for RTX 4060 with other services running)
- Default: fp16, RTX 4060 safe when used standalone

Reference:
  Xiao, B. et al. (2023). Florence-2: Advancing a Unified Representation
  for a Variety of Vision Tasks. arXiv:2311.06242
"""

from __future__ import annotations

import gc
import json
import logging
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

class Florence2Variant(str, Enum):
    LARGE = "microsoft/Florence-2-large"
    BASE = "microsoft/Florence-2-base"
    LARGE_FT = "microsoft/Florence-2-large-ft"
    BASE_FT = "microsoft/Florence-2-base-ft"


@dataclass
class Florence2Config:
    """Configuration for Florence-2 inference."""

    model_id: str = Florence2Variant.LARGE
    """HuggingFace model repository."""

    torch_dtype: str = "float16"
    """Compute dtype. float16 recommended for RTX 4060 (3.5 GB VRAM)."""

    device_map: str = "cuda"
    """Device placement."""

    trust_remote_code: bool = True
    """Required for Florence-2 custom model code."""

    # Generation parameters
    max_new_tokens: int = 512
    do_sample: bool = False
    num_beams: int = 3

    # Retry logic
    max_retries: int = 2
    retry_delay_s: float = 0.5

    # VRAM estimates (GB)
    vram_fp16_gb: float = 3.5
    vram_fp32_gb: float = 7.0


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class Florence2MaturityResult:
    """Maturity analysis result from Florence-2."""

    maturity_stage: str
    """One of: clear, cloudy, amber, mixed, unknown."""

    clear_fraction: float
    cloudy_fraction: float
    amber_fraction: float

    confidence: float
    """Estimated confidence [0, 1]."""

    raw_caption: str
    """Raw model output before parsing."""

    is_valid: bool = True
    parsing_errors: list[str] = field(default_factory=list)

    SCIENTIFIC_CAVEAT: str = (
        "Maturity stage is an observable optical property. "
        "No inference about cannabinoid content (THC, CBD, CBN) "
        "can be made from visual inspection alone."
    )


@dataclass
class Florence2QualityResult:
    """Image quality assessment result."""

    overall_quality: str
    """One of: high, medium, low, unusable."""

    is_in_focus: bool
    focus_score: float  # 0–1
    has_debris: bool
    adequate_lighting: bool
    confidence: float
    raw_caption: str
    is_valid: bool = True


@dataclass
class Florence2DetectionResult:
    """Open-vocabulary detection result."""

    labels: list[str]
    boxes: list[list[float]]  # (x1, y1, x2, y2) normalized [0,1]
    scores: list[float]
    raw_response: dict[str, Any] = field(default_factory=dict)
    is_valid: bool = True


@dataclass
class Florence2RegionResult:
    """Region description result for a specific image crop."""

    description: str
    maturity_hint: str | None
    morphology_hint: str | None
    confidence: float
    is_valid: bool = True


# ---------------------------------------------------------------------------
# Prompt templates for Florence-2
# ---------------------------------------------------------------------------

_MATURITY_SYSTEM_PROMPT = (
    "You are analyzing a cannabis trichome under a microscope. "
    "Describe the trichome heads' color and opacity.\n"
    "Classify maturity stage as ONE of: clear, cloudy, amber, mixed.\n"
    "Estimate fractions (0.0-1.0) summing to 1.0: clear_fraction, cloudy_fraction, amber_fraction.\n"
    "Respond ONLY as valid JSON:\n"
    '{"maturity_stage": "...", "clear_fraction": 0.0, "cloudy_fraction": 0.0, '
    '"amber_fraction": 0.0, "observations": "..."}'
)

_QUALITY_SYSTEM_PROMPT = (
    "Assess the quality of this microscopy image.\n"
    "Respond ONLY as valid JSON:\n"
    '{"overall_quality": "high|medium|low|unusable", "is_in_focus": true|false, '
    '"focus_score": 0.0-1.0, "has_debris": true|false, "adequate_lighting": true|false, '
    '"observations": "..."}'
)

_MORPHOLOGY_PROMPT = (
    "Identify trichome types visible in this cannabis microscopy image.\n"
    "Types: capitate_stalked, capitate_sessile, bulbous, non_glandular.\n"
    "Respond ONLY as valid JSON:\n"
    '{"dominant_type": "...", "types_present": ["..."], '
    '"density": "sparse|moderate|dense", "observations": "..."}'
)

# Florence-2 native task tokens
_TASK_CAPTION = "<CAPTION>"
_TASK_DETAILED_CAPTION = "<DETAILED_CAPTION>"
_TASK_OVD = "<OPEN_VOCABULARY_DETECTION>"
_TASK_REGION_DESC = "<REGION_TO_DESCRIPTION>"


# ---------------------------------------------------------------------------
# Main Florence-2 Labeler
# ---------------------------------------------------------------------------

class Florence2Labeler:
    """
    Florence-2-Large wrapper for trichome analysis.

    Usage::

        labeler = Florence2Labeler(config)
        labeler.load()
        result = labeler.label_maturity(image_array)
        labeler.unload()  # free VRAM

    VRAM: ~3.5 GB (fp16) — safe on RTX 4060 when no other GPU task is running.
    """

    def __init__(self, config: Florence2Config | None = None) -> None:
        self.config = config or Florence2Config()
        self._model: Any | None = None
        self._processor: Any | None = None
        self._device: str | None = None
        self._is_loaded: bool = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load Florence-2 model and processor into GPU memory."""
        if self._is_loaded:
            return

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoProcessor
        except ImportError as e:
            raise ImportError(
                "transformers>=4.41.0 required for Florence-2. "
                "Install: pip install transformers>=4.41.0"
            ) from e

        logger.info(
            "Loading Florence-2 (%s, dtype=%s)",
            self.config.model_id,
            self.config.torch_dtype,
        )
        t0 = time.monotonic()

        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map.get(self.config.torch_dtype, torch.float16)

        self._model = AutoModelForCausalLM.from_pretrained(
            self.config.model_id,
            torch_dtype=torch_dtype,
            trust_remote_code=self.config.trust_remote_code,
        ).to(self.config.device_map)
        self._model.eval()

        self._processor = AutoProcessor.from_pretrained(
            self.config.model_id,
            trust_remote_code=self.config.trust_remote_code,
        )

        import torch
        self._device = (
            self.config.device_map
            if self.config.device_map != "auto"
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self._is_loaded = True
        elapsed = time.monotonic() - t0
        logger.info("Florence-2 loaded in %.1fs", elapsed)

    def unload(self) -> None:
        """Release model and free GPU memory."""
        if not self._is_loaded:
            return

        try:
            import torch
        except ImportError:
            pass

        del self._model
        del self._processor
        self._model = None
        self._processor = None

        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except Exception:
            pass

        gc.collect()
        self._is_loaded = False
        logger.info("Florence-2 unloaded, VRAM released")

    def __enter__(self) -> "Florence2Labeler":
        self.load()
        return self

    def __exit__(self, *args: Any) -> None:
        self.unload()

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    # ------------------------------------------------------------------
    # Core inference
    # ------------------------------------------------------------------

    def _run_task(
        self,
        image: NDArray[np.uint8],
        task_token: str,
        text_input: str | None = None,
    ) -> dict[str, Any]:
        """
        Run a Florence-2 task on an image.

        Args:
            image: HWC uint8 numpy array.
            task_token: Florence-2 task token (e.g. "<CAPTION>").
            text_input: Optional additional text (e.g. for OVD: "trichome head").

        Returns:
            Parsed Florence-2 response dict.
        """
        if not self._is_loaded:
            raise RuntimeError("Model not loaded. Call .load() first.")

        try:
            import torch
            from PIL import Image as PILImage
        except ImportError as e:
            raise ImportError("PIL and torch required for Florence-2 inference") from e

        # Convert numpy → PIL
        pil_image = PILImage.fromarray(image.astype(np.uint8))

        prompt = task_token if text_input is None else f"{task_token}{text_input}"

        inputs = self._processor(
            text=prompt,
            images=pil_image,
            return_tensors="pt",
        ).to(self._device, self._get_dtype())

        with torch.no_grad():
            generated_ids = self._model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=self.config.max_new_tokens,
                do_sample=self.config.do_sample,
                num_beams=self.config.num_beams,
            )

        generated_text = self._processor.batch_decode(
            generated_ids, skip_special_tokens=False
        )[0]

        parsed = self._processor.post_process_generation(
            generated_text,
            task=task_token,
            image_size=(pil_image.width, pil_image.height),
        )
        return parsed

    def _get_dtype(self) -> Any:
        """Return torch dtype matching config."""
        import torch
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        return dtype_map.get(self.config.torch_dtype, torch.float16)

    def _run_custom_prompt(
        self,
        image: NDArray[np.uint8],
        system_prompt: str,
    ) -> str:
        """
        Run Florence-2 with a custom descriptive prompt and parse text output.

        Uses <DETAILED_CAPTION> + appends system prompt instructions.
        Returns raw text output.
        """
        if not self._is_loaded:
            raise RuntimeError("Model not loaded. Call .load() first.")

        try:
            import torch
            from PIL import Image as PILImage
        except ImportError as e:
            raise ImportError("PIL and torch required") from e

        pil_image = PILImage.fromarray(image.astype(np.uint8))

        # Use DETAILED_CAPTION task token with custom instruction appended
        full_prompt = f"{_TASK_DETAILED_CAPTION}{system_prompt}"

        inputs = self._processor(
            text=full_prompt,
            images=pil_image,
            return_tensors="pt",
        ).to(self._device, self._get_dtype())

        with torch.no_grad():
            generated_ids = self._model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=self.config.max_new_tokens,
                do_sample=self.config.do_sample,
                num_beams=self.config.num_beams,
            )

        return self._processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0]

    # ------------------------------------------------------------------
    # Label maturity
    # ------------------------------------------------------------------

    def label_maturity(
        self,
        image: NDArray[np.uint8],
    ) -> Florence2MaturityResult:
        """
        Analyze trichome maturity from a microscopy image.

        Args:
            image: HWC uint8 numpy array (crop or full field of view).

        Returns:
            Florence2MaturityResult with stage classification and fractions.
        """
        errors: list[str] = []
        raw_text = ""

        for attempt in range(self.config.max_retries + 1):
            try:
                raw_text = self._run_custom_prompt(image, _MATURITY_SYSTEM_PROMPT)
                parsed = self._extract_json(raw_text)
                if parsed:
                    return self._build_maturity_result(parsed, raw_text, errors)
            except Exception as e:
                errors.append(f"attempt {attempt}: {e}")
                if attempt < self.config.max_retries:
                    time.sleep(self.config.retry_delay_s)

        # Fallback: try to extract from raw text
        return Florence2MaturityResult(
            maturity_stage="unknown",
            clear_fraction=0.0,
            cloudy_fraction=0.0,
            amber_fraction=0.0,
            confidence=0.0,
            raw_caption=raw_text,
            is_valid=False,
            parsing_errors=errors,
        )

    def _build_maturity_result(
        self,
        parsed: dict[str, Any],
        raw_text: str,
        errors: list[str],
    ) -> Florence2MaturityResult:
        """Build maturity result from parsed JSON dict."""
        stage = str(parsed.get("maturity_stage", "unknown")).lower()
        if stage not in {"clear", "cloudy", "amber", "mixed", "unknown"}:
            errors.append(f"Invalid stage: {stage}")
            stage = "unknown"

        cf = float(parsed.get("clear_fraction", 0.0))
        mf = float(parsed.get("cloudy_fraction", 0.0))
        af = float(parsed.get("amber_fraction", 0.0))

        # Clamp and renormalize
        cf = max(0.0, min(1.0, cf))
        mf = max(0.0, min(1.0, mf))
        af = max(0.0, min(1.0, af))
        total = cf + mf + af
        if total > 0.0:
            cf, mf, af = cf / total, mf / total, af / total
        else:
            errors.append("All fractions are zero — setting uniform")
            cf = mf = af = 1.0 / 3.0

        # Confidence heuristic: if stage matches dominant fraction
        dominant_match = {
            "clear": cf,
            "cloudy": mf,
            "amber": af,
            "mixed": min(cf, mf, af) * 3,
        }.get(stage, 0.5)
        confidence = min(0.85, 0.55 + dominant_match * 0.3)

        return Florence2MaturityResult(
            maturity_stage=stage,
            clear_fraction=cf,
            cloudy_fraction=mf,
            amber_fraction=af,
            confidence=confidence,
            raw_caption=raw_text,
            is_valid=len([e for e in errors if "Invalid" in e]) == 0,
            parsing_errors=errors,
        )

    # ------------------------------------------------------------------
    # Quality screening
    # ------------------------------------------------------------------

    def assess_quality(
        self,
        image: NDArray[np.uint8],
    ) -> Florence2QualityResult:
        """
        Assess microscopy image quality.

        Args:
            image: HWC uint8 numpy array.

        Returns:
            Florence2QualityResult with focus, debris, lighting assessment.
        """
        errors: list[str] = []
        raw_text = ""

        for attempt in range(self.config.max_retries + 1):
            try:
                raw_text = self._run_custom_prompt(image, _QUALITY_SYSTEM_PROMPT)
                parsed = self._extract_json(raw_text)
                if parsed:
                    return self._build_quality_result(parsed, raw_text)
            except Exception as e:
                errors.append(str(e))
                if attempt < self.config.max_retries:
                    time.sleep(self.config.retry_delay_s)

        return Florence2QualityResult(
            overall_quality="unknown",
            is_in_focus=False,
            focus_score=0.0,
            has_debris=False,
            adequate_lighting=False,
            confidence=0.0,
            raw_caption=raw_text,
            is_valid=False,
        )

    def _build_quality_result(
        self,
        parsed: dict[str, Any],
        raw_text: str,
    ) -> Florence2QualityResult:
        quality = str(parsed.get("overall_quality", "unknown")).lower()
        if quality not in {"high", "medium", "low", "unusable", "unknown"}:
            quality = "unknown"

        focus_score = float(parsed.get("focus_score", 0.5))
        focus_score = max(0.0, min(1.0, focus_score))

        confidence = 0.70 if quality != "unknown" else 0.30

        return Florence2QualityResult(
            overall_quality=quality,
            is_in_focus=bool(parsed.get("is_in_focus", focus_score > 0.5)),
            focus_score=focus_score,
            has_debris=bool(parsed.get("has_debris", False)),
            adequate_lighting=bool(parsed.get("adequate_lighting", True)),
            confidence=confidence,
            raw_caption=raw_text,
            is_valid=quality != "unknown",
        )

    # ------------------------------------------------------------------
    # Morphology
    # ------------------------------------------------------------------

    def label_morphology(
        self,
        image: NDArray[np.uint8],
    ) -> dict[str, Any]:
        """
        Identify trichome types and density from image.

        Returns:
            dict with dominant_type, types_present, density, confidence, raw_caption.
        """
        raw_text = ""
        for attempt in range(self.config.max_retries + 1):
            try:
                raw_text = self._run_custom_prompt(image, _MORPHOLOGY_PROMPT)
                parsed = self._extract_json(raw_text)
                if parsed:
                    return {
                        "dominant_type": str(parsed.get("dominant_type", "unknown")),
                        "types_present": list(parsed.get("types_present", [])),
                        "density": str(parsed.get("density", "moderate")),
                        "confidence": 0.65,
                        "raw_caption": raw_text,
                        "is_valid": True,
                    }
            except Exception as e:
                logger.debug("Morphology attempt %d failed: %s", attempt, e)
                if attempt < self.config.max_retries:
                    time.sleep(self.config.retry_delay_s)

        return {
            "dominant_type": "unknown",
            "types_present": [],
            "density": "unknown",
            "confidence": 0.0,
            "raw_caption": raw_text,
            "is_valid": False,
        }

    # ------------------------------------------------------------------
    # Open-vocabulary detection
    # ------------------------------------------------------------------

    def detect_trichomes(
        self,
        image: NDArray[np.uint8],
        query: str = "trichome head . trichome stalk . cannabis trichome",
    ) -> Florence2DetectionResult:
        """
        Run open-vocabulary detection for trichomes.

        Args:
            image: HWC uint8 numpy array.
            query: Text query (dot-separated objects for OVD).

        Returns:
            Florence2DetectionResult with boxes (normalized [0,1]).
        """
        if not self._is_loaded:
            raise RuntimeError("Model not loaded.")

        try:
            result = self._run_task(image, _TASK_OVD, text_input=query)
            ovd_result = result.get(_TASK_OVD, {})

            labels = ovd_result.get("labels", [])
            bboxes = ovd_result.get("bboxes", [])

            h, w = image.shape[:2]
            normalized_boxes = [
                [b[0] / w, b[1] / h, b[2] / w, b[3] / h]
                for b in bboxes
            ]
            # Florence-2 OVD doesn't return scores — use placeholder
            scores = [0.70] * len(labels)

            return Florence2DetectionResult(
                labels=labels,
                boxes=normalized_boxes,
                scores=scores,
                raw_response=ovd_result,
                is_valid=len(labels) > 0,
            )
        except Exception as e:
            logger.warning("OVD detection failed: %s", e)
            return Florence2DetectionResult(
                labels=[],
                boxes=[],
                scores=[],
                is_valid=False,
            )

    def describe_region(
        self,
        image: NDArray[np.uint8],
        box: tuple[float, float, float, float],
    ) -> Florence2RegionResult:
        """
        Describe a specific region of the image (e.g. individual trichome crop).

        Args:
            image: Full image HWC uint8.
            box: (x1, y1, x2, y2) in pixel coordinates.

        Returns:
            Florence2RegionResult with textual description.
        """
        if not self._is_loaded:
            raise RuntimeError("Model not loaded.")

        try:
            h, w = image.shape[:2]
            x1, y1, x2, y2 = [max(0, int(v)) for v in box]
            x2, y2 = min(w, x2), min(h, y2)
            crop = image[y1:y2, x1:x2]

            if crop.size == 0:
                return Florence2RegionResult(
                    description="",
                    maturity_hint=None,
                    morphology_hint=None,
                    confidence=0.0,
                    is_valid=False,
                )

            result = self._run_task(crop, _TASK_DETAILED_CAPTION)
            description = result.get(_TASK_DETAILED_CAPTION, "")

            maturity_hint = self._extract_maturity_hint(description)
            morphology_hint = self._extract_morphology_hint(description)

            return Florence2RegionResult(
                description=description,
                maturity_hint=maturity_hint,
                morphology_hint=morphology_hint,
                confidence=0.60 if description else 0.0,
                is_valid=bool(description),
            )
        except Exception as e:
            logger.warning("Region description failed: %s", e)
            return Florence2RegionResult(
                description="",
                maturity_hint=None,
                morphology_hint=None,
                confidence=0.0,
                is_valid=False,
            )

    # ------------------------------------------------------------------
    # Dense caption (for dataset exploration)
    # ------------------------------------------------------------------

    def caption_image(self, image: NDArray[np.uint8]) -> str:
        """Return a brief caption for the image (uses <CAPTION> task)."""
        if not self._is_loaded:
            raise RuntimeError("Model not loaded.")
        try:
            result = self._run_task(image, _TASK_CAPTION)
            return result.get(_TASK_CAPTION, "")
        except Exception as e:
            logger.warning("Caption failed: %s", e)
            return ""

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_json(text: str) -> dict[str, Any] | None:
        """
        Extract first JSON object from model output text.

        Handles:
        - ```json ... ``` code blocks
        - Raw JSON embedded in prose
        - Multiple JSON-like structures (takes first valid one)
        """
        # Strip markdown code fences
        text = re.sub(r"```json\s*", "", text)
        text = re.sub(r"```\s*", "", text)

        # Try to find JSON object
        json_pattern = re.compile(r"\{[^{}]*\}", re.DOTALL)
        matches = json_pattern.findall(text)

        for match in matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue

        # Try parsing entire text as JSON
        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            return None

    @staticmethod
    def _extract_maturity_hint(text: str) -> str | None:
        """Extract maturity keywords from description text."""
        text_lower = text.lower()
        if any(w in text_lower for w in ["amber", "brown", "orange", "oxidiz"]):
            return "amber"
        if any(w in text_lower for w in ["milky", "cloudy", "opaque", "white", "turbid"]):
            return "cloudy"
        if any(w in text_lower for w in ["clear", "transparent", "translucent"]):
            return "clear"
        return None

    @staticmethod
    def _extract_morphology_hint(text: str) -> str | None:
        """Extract morphology keywords from description text."""
        text_lower = text.lower()
        if any(w in text_lower for w in ["stalk", "stalked", "peduncle", "long"]):
            return "capitate_stalked"
        if any(w in text_lower for w in ["sessile", "flat", "attached"]):
            return "capitate_sessile"
        if any(w in text_lower for w in ["bulbous", "small", "tiny", "round"]):
            return "bulbous"
        return None

    # ------------------------------------------------------------------
    # VRAM info
    # ------------------------------------------------------------------

    @property
    def vram_required_gb(self) -> float:
        """Estimated VRAM requirement."""
        if "float32" in self.config.torch_dtype:
            return self.config.vram_fp32_gb
        return self.config.vram_fp16_gb

    def get_vram_usage_gb(self) -> float | None:
        """Return current GPU VRAM usage in GB, or None if not on GPU."""
        try:
            import torch
            if torch.cuda.is_available() and self._is_loaded:
                allocated = torch.cuda.memory_allocated()
                return allocated / (1024 ** 3)
        except Exception:
            pass
        return None
