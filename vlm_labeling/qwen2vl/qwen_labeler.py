"""
vlm_labeling.qwen2vl.qwen_labeler — Qwen2-VL-7B VLM backend.

Qwen2-VL-7B (Alibaba): 7B parameters, strongest visual reasoning.
Requires 4-bit quantization (BitsAndBytes) to fit on RTX 4060.

VRAM estimates:
- 4-bit quant: ~5.5 GB  ← default, RTX 4060 safe when standalone
- 8-bit quant: ~9.5 GB  ← exceeds 8 GB limit, NOT recommended
- fp16 (no quant): ~16 GB ← requires A100/H100

Reference:
  Qwen2-VL: Enhancing Vision-Language Model's Perception of the World
  at Any Resolution. Bai et al. (2024). arXiv:2409.12191.
"""

from __future__ import annotations

import gc
import json
import logging
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

class QwenQuantization(str, Enum):
    BITS_4 = "4bit"   # ~5.5 GB VRAM — recommended for RTX 4060
    BITS_8 = "8bit"   # ~9.5 GB — too large for RTX 4060
    NONE = "none"     # ~16 GB — requires datacenter GPU


@dataclass
class QwenVLConfig:
    """Configuration for Qwen2-VL inference."""

    model_id: str = "Qwen/Qwen2-VL-7B-Instruct"
    """HuggingFace model ID."""

    quantization: QwenQuantization | str = QwenQuantization.BITS_4
    """Quantization level. Use '4bit' for RTX 4060."""

    device_map: str = "cuda"

    # Generation
    max_new_tokens: int = 512
    temperature: float = 0.1
    do_sample: bool = False  # greedy for determinism

    # Retry
    max_retries: int = 2
    retry_delay_s: float = 0.5

    # VRAM estimates (GB)
    vram_4bit_gb: float = 5.5
    vram_8bit_gb: float = 9.5
    vram_fp16_gb: float = 16.0


# ---------------------------------------------------------------------------
# Result types (compatible with Moondream + Florence2 results)
# ---------------------------------------------------------------------------

@dataclass
class QwenMaturityResult:
    """Maturity analysis from Qwen2-VL."""

    maturity_stage: str
    clear_fraction: float
    cloudy_fraction: float
    amber_fraction: float
    confidence: float
    raw_response: str
    reasoning: str
    is_valid: bool = True
    parsing_errors: list[str] = field(default_factory=list)

    SCIENTIFIC_CAVEAT: str = (
        "Maturity stage is an observable optical property. "
        "THC, CBD, or CBN content cannot be inferred from visual appearance."
    )


@dataclass
class QwenQualityResult:
    """Image quality assessment from Qwen2-VL."""

    overall_quality: str
    is_in_focus: bool
    focus_score: float
    has_debris: bool
    adequate_lighting: bool
    usable_for_training: bool
    confidence: float
    raw_response: str
    is_valid: bool = True


@dataclass
class QwenMorphologyResult:
    """Morphology classification from Qwen2-VL."""

    dominant_type: str
    types_present: list[str]
    density: str
    count_estimate: int | None
    confidence: float
    raw_response: str
    is_valid: bool = True


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_MATURITY_MESSAGES_TEMPLATE = """You are an expert botanist analyzing cannabis trichomes under a microscope.

Examine the image carefully and:
1. Identify the maturity stage based on trichome head opacity and color
2. Estimate the fraction of trichomes in each maturity stage

IMPORTANT: Respond ONLY with valid JSON. Do not add explanations outside the JSON.

Required JSON format:
{{
  "maturity_stage": "clear|cloudy|amber|mixed|unknown",
  "clear_fraction": 0.0,
  "cloudy_fraction": 0.0,
  "amber_fraction": 0.0,
  "confidence": 0.0,
  "reasoning": "brief explanation of visual observations"
}}

Rules:
- clear_fraction + cloudy_fraction + amber_fraction must sum to 1.0
- All fractions must be between 0.0 and 1.0
- maturity_stage should match the dominant fraction
- confidence should reflect certainty of assessment (0.0-1.0)"""

_QUALITY_MESSAGES_TEMPLATE = """You are a microscopy image quality analyst.

Assess the quality of this trichome microscopy image for use in a machine learning dataset.

Respond ONLY with valid JSON:
{{
  "overall_quality": "high|medium|low|unusable",
  "is_in_focus": true|false,
  "focus_score": 0.0,
  "has_debris": true|false,
  "adequate_lighting": true|false,
  "usable_for_training": true|false,
  "confidence": 0.0,
  "observations": "brief notes"
}}"""

_MORPHOLOGY_MESSAGES_TEMPLATE = """You are an expert in cannabis trichome biology.

Analyze this microscopy image and identify the trichome types present.

Trichome types:
- capitate_stalked: large head on visible stalk (most common on calyxes)
- capitate_sessile: head directly on leaf surface, no visible stalk
- bulbous: very small, spherical, distributed throughout plant
- non_glandular: hair-like, no secretory head

Respond ONLY with valid JSON:
{{
  "dominant_type": "capitate_stalked|capitate_sessile|bulbous|non_glandular|mixed|unknown",
  "types_present": ["..."],
  "density": "sparse|moderate|dense",
  "count_estimate": null,
  "confidence": 0.0,
  "observations": "brief notes"
}}"""


# ---------------------------------------------------------------------------
# Qwen2-VL Labeler
# ---------------------------------------------------------------------------

class QwenVLLabeler:
    """
    Qwen2-VL-7B wrapper for trichome analysis.

    Uses 4-bit quantization by default to fit within RTX 4060 8 GB VRAM.
    Provides the highest-quality VLM reasoning of all supported backends.

    Usage::

        labeler = QwenVLLabeler(config)
        labeler.load()
        result = labeler.label_maturity(image_array)
        labeler.unload()
    """

    def __init__(self, config: QwenVLConfig | None = None) -> None:
        self.config = config or QwenVLConfig()
        self._model: Any | None = None
        self._processor: Any | None = None
        self._is_loaded: bool = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load Qwen2-VL model with quantization into GPU memory."""
        if self._is_loaded:
            return

        try:
            import torch
            from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
        except ImportError as e:
            raise ImportError(
                "transformers>=4.45.0 required for Qwen2-VL. "
                "Install: pip install transformers>=4.45.0 qwen_vl_utils"
            ) from e

        logger.info(
            "Loading Qwen2-VL (%s, quantization=%s)",
            self.config.model_id,
            self.config.quantization,
        )
        t0 = time.monotonic()

        model_kwargs: dict[str, Any] = {
            "device_map": self.config.device_map,
        }

        quant = self.config.quantization
        if quant == QwenQuantization.BITS_4 or quant == "4bit":
            try:
                from transformers import BitsAndBytesConfig
                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                )
            except ImportError:
                logger.warning("bitsandbytes not found, using float16 (may OOM)")
                model_kwargs["torch_dtype"] = torch.float16
        elif quant == QwenQuantization.BITS_8 or quant == "8bit":
            try:
                from transformers import BitsAndBytesConfig
                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_8bit=True,
                )
            except ImportError:
                model_kwargs["torch_dtype"] = torch.float16
        else:
            model_kwargs["torch_dtype"] = torch.float16

        self._model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.config.model_id,
            **model_kwargs,
        )

        # Min/max pixels for dynamic resolution
        min_pixels = 256 * 28 * 28
        max_pixels = 1280 * 28 * 28

        self._processor = AutoProcessor.from_pretrained(
            self.config.model_id,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )

        self._is_loaded = True
        elapsed = time.monotonic() - t0
        logger.info("Qwen2-VL loaded in %.1fs", elapsed)

    def unload(self) -> None:
        """Release model and free GPU memory."""
        if not self._is_loaded:
            return

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
        logger.info("Qwen2-VL unloaded")

    def __enter__(self) -> "QwenVLLabeler":
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

    def _run_inference(
        self,
        image: NDArray[np.uint8],
        system_prompt: str,
    ) -> str:
        """
        Run Qwen2-VL inference with image + system prompt.

        Args:
            image: HWC uint8 numpy array.
            system_prompt: Instruction/task description.

        Returns:
            Raw model output string.
        """
        if not self._is_loaded:
            raise RuntimeError("Model not loaded. Call .load() first.")

        try:
            import torch
            from PIL import Image as PILImage
        except ImportError as e:
            raise ImportError("PIL and torch required") from e

        pil_image = PILImage.fromarray(image.astype(np.uint8))

        # Qwen2-VL chat format
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": pil_image,
                    },
                    {
                        "type": "text",
                        "text": system_prompt,
                    },
                ],
            }
        ]

        # Apply chat template
        text = self._processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # Process inputs
        inputs = self._processor(
            text=[text],
            images=[pil_image],
            padding=True,
            return_tensors="pt",
        )

        # Move to device
        device = next(self._model.parameters()).device
        inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=self.config.do_sample,
                temperature=self.config.temperature if self.config.do_sample else None,
            )

        # Decode only generated tokens (not the input prompt)
        input_len = inputs["input_ids"].shape[1]
        generated_ids = output_ids[:, input_len:]
        response = self._processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        return response.strip()

    # ------------------------------------------------------------------
    # Label maturity
    # ------------------------------------------------------------------

    def label_maturity(
        self,
        image: NDArray[np.uint8],
    ) -> QwenMaturityResult:
        """
        Classify trichome maturity from a microscopy image.

        Returns:
            QwenMaturityResult with stage, fractions, confidence, reasoning.
        """
        errors: list[str] = []
        raw_response = ""

        for attempt in range(self.config.max_retries + 1):
            try:
                raw_response = self._run_inference(image, _MATURITY_MESSAGES_TEMPLATE)
                parsed = _extract_json(raw_response)
                if parsed:
                    return self._build_maturity_result(parsed, raw_response, errors)
                else:
                    errors.append(f"attempt {attempt}: JSON extraction failed")
            except Exception as e:
                errors.append(f"attempt {attempt}: {e}")
                if attempt < self.config.max_retries:
                    time.sleep(self.config.retry_delay_s)

        return QwenMaturityResult(
            maturity_stage="unknown",
            clear_fraction=0.0,
            cloudy_fraction=0.0,
            amber_fraction=0.0,
            confidence=0.0,
            raw_response=raw_response,
            reasoning="",
            is_valid=False,
            parsing_errors=errors,
        )

    def _build_maturity_result(
        self,
        parsed: dict[str, Any],
        raw: str,
        errors: list[str],
    ) -> QwenMaturityResult:
        stage = str(parsed.get("maturity_stage", "unknown")).lower()
        if stage not in {"clear", "cloudy", "amber", "mixed", "unknown"}:
            errors.append(f"Invalid stage: {stage}")
            stage = "unknown"

        cf = float(parsed.get("clear_fraction", 0.0))
        mf = float(parsed.get("cloudy_fraction", 0.0))
        af = float(parsed.get("amber_fraction", 0.0))

        cf = max(0.0, min(1.0, cf))
        mf = max(0.0, min(1.0, mf))
        af = max(0.0, min(1.0, af))

        total = cf + mf + af
        if total > 0.01:
            cf, mf, af = cf / total, mf / total, af / total
        else:
            cf, mf, af = 1 / 3, 1 / 3, 1 / 3
            errors.append("All fractions zero — normalized to uniform")

        confidence = max(0.0, min(1.0, float(parsed.get("confidence", 0.65))))
        reasoning = str(parsed.get("reasoning", ""))

        return QwenMaturityResult(
            maturity_stage=stage,
            clear_fraction=cf,
            cloudy_fraction=mf,
            amber_fraction=af,
            confidence=confidence,
            raw_response=raw,
            reasoning=reasoning,
            is_valid=stage != "unknown",
            parsing_errors=errors,
        )

    # ------------------------------------------------------------------
    # Quality assessment
    # ------------------------------------------------------------------

    def assess_quality(self, image: NDArray[np.uint8]) -> QwenQualityResult:
        """Assess microscopy image quality."""
        raw_response = ""
        for attempt in range(self.config.max_retries + 1):
            try:
                raw_response = self._run_inference(image, _QUALITY_MESSAGES_TEMPLATE)
                parsed = _extract_json(raw_response)
                if parsed:
                    quality = str(parsed.get("overall_quality", "unknown")).lower()
                    if quality not in {"high", "medium", "low", "unusable"}:
                        quality = "unknown"
                    return QwenQualityResult(
                        overall_quality=quality,
                        is_in_focus=bool(parsed.get("is_in_focus", False)),
                        focus_score=max(0.0, min(1.0, float(parsed.get("focus_score", 0.5)))),
                        has_debris=bool(parsed.get("has_debris", False)),
                        adequate_lighting=bool(parsed.get("adequate_lighting", True)),
                        usable_for_training=bool(parsed.get("usable_for_training", quality in {"high", "medium"})),
                        confidence=max(0.0, min(1.0, float(parsed.get("confidence", 0.70)))),
                        raw_response=raw_response,
                        is_valid=quality != "unknown",
                    )
            except Exception as e:
                logger.debug("Quality attempt %d failed: %s", attempt, e)
                if attempt < self.config.max_retries:
                    time.sleep(self.config.retry_delay_s)

        return QwenQualityResult(
            overall_quality="unknown",
            is_in_focus=False,
            focus_score=0.0,
            has_debris=False,
            adequate_lighting=False,
            usable_for_training=False,
            confidence=0.0,
            raw_response=raw_response,
            is_valid=False,
        )

    # ------------------------------------------------------------------
    # Morphology
    # ------------------------------------------------------------------

    def label_morphology(self, image: NDArray[np.uint8]) -> QwenMorphologyResult:
        """Identify trichome types and density."""
        raw_response = ""
        for attempt in range(self.config.max_retries + 1):
            try:
                raw_response = self._run_inference(image, _MORPHOLOGY_MESSAGES_TEMPLATE)
                parsed = _extract_json(raw_response)
                if parsed:
                    dominant = str(parsed.get("dominant_type", "unknown")).lower()
                    return QwenMorphologyResult(
                        dominant_type=dominant,
                        types_present=list(parsed.get("types_present", [])),
                        density=str(parsed.get("density", "moderate")),
                        count_estimate=parsed.get("count_estimate"),
                        confidence=max(0.0, min(1.0, float(parsed.get("confidence", 0.65)))),
                        raw_response=raw_response,
                        is_valid=dominant != "unknown",
                    )
            except Exception as e:
                logger.debug("Morphology attempt %d failed: %s", attempt, e)
                if attempt < self.config.max_retries:
                    time.sleep(self.config.retry_delay_s)

        return QwenMorphologyResult(
            dominant_type="unknown",
            types_present=[],
            density="unknown",
            count_estimate=None,
            confidence=0.0,
            raw_response=raw_response,
            is_valid=False,
        )

    # ------------------------------------------------------------------
    # VRAM info
    # ------------------------------------------------------------------

    @property
    def vram_required_gb(self) -> float:
        quant = self.config.quantization
        if quant == QwenQuantization.BITS_4 or quant == "4bit":
            return self.config.vram_4bit_gb
        if quant == QwenQuantization.BITS_8 or quant == "8bit":
            return self.config.vram_8bit_gb
        return self.config.vram_fp16_gb


# ---------------------------------------------------------------------------
# Shared utility (also used by florence_labeler)
# ---------------------------------------------------------------------------

def _extract_json(text: str) -> dict[str, Any] | None:
    """Extract first valid JSON object from model output."""
    text = re.sub(r"```json\s*", "", text)
    text = re.sub(r"```\s*", "", text)

    json_pattern = re.compile(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)?\}", re.DOTALL)
    for match in json_pattern.findall(text):
        try:
            return json.loads(match)
        except json.JSONDecodeError:
            continue

    # Greedy fallback
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            pass

    return None
