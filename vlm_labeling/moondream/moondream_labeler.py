"""
vlm_labeling.moondream.moondream_labeler — Moondream-2B VLM backend.

MODEL: vikhyatk/moondream2 (2B parameters)
Default quantization: none (FP16) → ~4.2 GB VRAM

NOTE: bitsandbytes 4-bit/8-bit quantization is NOT supported for moondream2.
The model uses custom functional linear calls (F.linear(x, w.weight, w.bias))
that bypass bitsandbytes module interception, resulting in Half×Byte dtype errors.

CAPABILITY PROFILE:
- Best at: binary/multi-class classification, captioning, simple Q&A
- Acceptable: maturity stage classification, morphology type
- Weak: precise counting (tends to overcount), fine measurement
- Speed: ~0.8s per image on RTX 4060 (4-bit)

HARDWARE REQUIREMENT:
- Minimum: 3 GB VRAM (4-bit quant)
- Recommended: 4 GB VRAM (for headroom)
- RTX 4060 (8 GB): runs comfortably with room for detection model

IMPLEMENTATION NOTES:
- Uses transformers AutoModel + AutoTokenizer
- Supports 4-bit quantization via bitsandbytes
- Batch inference not supported by Moondream (sequential only)
- Context window: 2048 tokens (sufficient for all trichome prompts)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from shared.logging.logger import get_logger
from vlm_labeling.prompts.trichome_prompts import (
    PromptTemplate,
    MATURITY_CLASSIFICATION_PROMPT,
    TRICHOME_DETECTION_COUNT_PROMPT,
    TRICHOME_MORPHOLOGY_PROMPT,
    IMAGE_QUALITY_PROMPT,
    PROMPT_REGISTRY,
)

logger = get_logger(__name__)


def _default_device() -> str:
    try:
        from backend.utils.compute import get_torch_device
        return get_torch_device()
    except ImportError:
        import torch
        return "cuda:0" if torch.cuda.is_available() else "cpu"


@dataclass
class MoondreamConfig:
    """Configuration for Moondream-2B VLM."""

    model_id: str = "vikhyatk/moondream2"
    """HuggingFace model ID."""

    revision: str = "2025-01-09"
    """Model revision for reproducibility."""

    quantization: str = "none"
    """
    Quantization level:
    - 'none': FP16, ~4.2 GB VRAM (default — bitsandbytes incompatible with moondream2's
      custom functional linear calls that access w.weight directly)
    - '4bit': NOT supported for moondream2 (F.linear with Byte weight fails)
    - '8bit': NOT supported for moondream2 (same issue)
    """

    device: str = field(default_factory=lambda: _default_device())
    max_new_tokens: int = 512
    """Maximum tokens to generate per response."""

    trust_remote_code: bool = True
    """Required for Moondream."""

    cache_dir: str | None = None
    """Model cache directory. None = HuggingFace default (~/.cache/huggingface)."""

    max_image_size: int = 378
    """
    Moondream-2 native resolution: 378×378.
    Images are resized internally by the model processor.
    """

    retry_on_invalid_json: int = 2
    """Number of retries if VLM returns invalid JSON."""


@dataclass
class VLMInferenceResult:
    """Result from a single VLM inference call."""

    prompt_name: str
    raw_response: str
    parsed_response: dict[str, Any] | None
    is_valid: bool
    inference_time_s: float
    retry_count: int = 0

    # Metadata
    model_id: str = ""
    image_path: str = ""

    @property
    def confidence(self) -> float:
        """Extract confidence from parsed response."""
        if self.parsed_response is None:
            return 0.0
        return float(self.parsed_response.get("confidence", 0.0))

    @property
    def maturity_stage(self) -> str | None:
        """Extract maturity stage if this is a maturity result."""
        if self.parsed_response is None:
            return None
        return self.parsed_response.get("maturity_stage")

    @property
    def overall_quality(self) -> str | None:
        """Extract quality assessment if this is a quality result."""
        if self.parsed_response is None:
            return None
        return self.parsed_response.get("overall_quality")

    @property
    def is_analyzable(self) -> bool:
        """Whether image was deemed analyzable."""
        if self.parsed_response is None:
            return False
        return bool(self.parsed_response.get("analyzable", True))


class MoondreamLabeler:
    """
    Moondream-2B VLM wrapper for trichome labeling.

    Usage:
        labeler = MoondreamLabeler()
        labeler.load()

        result = labeler.label_image(
            image=image_array,
            prompt_name="maturity_classification"
        )
        print(result.maturity_stage, result.confidence)

        labeler.unload()

    Thread safety: NOT thread-safe. Use one instance per thread.
    """

    MODEL_NAME = "moondream"
    VRAM_REQUIREMENT_GB = {
        "none": 4.2,
        "4bit": 4.2,  # falls back to FP16 — bitsandbytes not supported
        "8bit": 4.2,  # falls back to FP16 — bitsandbytes not supported
    }

    def __init__(self, config: MoondreamConfig | None = None) -> None:
        self._config = config or MoondreamConfig()
        self._model: Any = None
        self._tokenizer: Any = None
        self._is_loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    @property
    def vram_requirement_gb(self) -> float:
        return self.VRAM_REQUIREMENT_GB.get(self._config.quantization, 4.2)

    def load(self) -> None:
        """
        Load Moondream model and tokenizer.

        Imports are deferred to avoid loading transformers at module level.
        First load takes 30-90s (download). Subsequent loads from cache: ~5s.
        """
        if self._is_loaded:
            logger.info("Moondream already loaded, skipping")
            return

        logger.info(
            "Loading Moondream-2B",
            model_id=self._config.model_id,
            quantization=self._config.quantization,
            vram_gb=self.vram_requirement_gb,
        )

        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch

            # Build quantization config
            model_kwargs: dict[str, Any] = {
                "trust_remote_code": self._config.trust_remote_code,
                "revision": self._config.revision,
            }

            if self._config.cache_dir:
                model_kwargs["cache_dir"] = self._config.cache_dir

            device = self._config.device
            if device.startswith("cuda") and not torch.cuda.is_available():
                logger.warning("CUDA not available, falling back to CPU")
                device = "cpu"

            if self._config.quantization == "4bit":
                try:
                    from transformers import BitsAndBytesConfig
                    bnb_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_use_double_quant=True,
                    )
                    model_kwargs["quantization_config"] = bnb_config
                    # bitsandbytes places the model on GPU automatically — no device_map needed
                except ImportError:
                    logger.warning(
                        "bitsandbytes not available, falling back to FP16. "
                        "Install with: pip install bitsandbytes"
                    )
                    model_kwargs["torch_dtype"] = torch.float16

            elif self._config.quantization == "8bit":
                try:
                    from transformers import BitsAndBytesConfig
                    bnb_config = BitsAndBytesConfig(load_in_8bit=True)
                    model_kwargs["quantization_config"] = bnb_config
                    # bitsandbytes handles device placement automatically
                except ImportError:
                    logger.warning("bitsandbytes not available, falling back to FP16")
                    model_kwargs["torch_dtype"] = torch.float16

            else:  # none / FP16
                model_kwargs["torch_dtype"] = torch.float16

            # Load tokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(
                self._config.model_id,
                revision=self._config.revision,
                trust_remote_code=self._config.trust_remote_code,
                cache_dir=self._config.cache_dir,
            )

            # Load model
            self._model = AutoModelForCausalLM.from_pretrained(
                self._config.model_id,
                **model_kwargs,
            )

            # Move to device only when not using quantization (quantization handles placement)
            if self._config.quantization == "none":
                self._model = self._model.to(device)

            self._model.eval()
            self._is_loaded = True

            logger.info(
                "Moondream loaded successfully",
                quantization=self._config.quantization,
            )

        except Exception as e:
            logger.error("Failed to load Moondream", error=str(e))
            raise RuntimeError(f"Failed to load Moondream model: {e}") from e

    def unload(self) -> None:
        """Unload model and free VRAM."""
        if not self._is_loaded:
            return

        import torch
        import gc

        self._model = None
        self._tokenizer = None
        self._is_loaded = False

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        gc.collect()
        logger.info("Moondream unloaded, VRAM freed")

    def label_image(
        self,
        image: NDArray[np.uint8],
        prompt_name: str = "maturity_classification",
        extra_context: str | None = None,
    ) -> VLMInferenceResult:
        """
        Run VLM inference on a single image.

        Args:
            image: RGB uint8 array (H, W, 3).
            prompt_name: Key from PROMPT_REGISTRY.
            extra_context: Optional additional context appended to user prompt.

        Returns:
            VLMInferenceResult with parsed JSON response.
        """
        if not self._is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        prompt_template = PROMPT_REGISTRY[prompt_name]
        return self._run_inference(
            image=image,
            prompt_template=prompt_template,
            extra_context=extra_context,
        )

    def label_maturity(
        self,
        image: NDArray[np.uint8],
    ) -> VLMInferenceResult:
        """Convenience method: classify trichome maturity."""
        return self.label_image(image, "maturity_classification")

    def assess_quality(
        self,
        image: NDArray[np.uint8],
    ) -> VLMInferenceResult:
        """Convenience method: assess image quality before expensive analysis."""
        return self.label_image(image, "image_quality")

    def label_morphology(
        self,
        image: NDArray[np.uint8],
    ) -> VLMInferenceResult:
        """Convenience method: classify dominant trichome morphology type."""
        return self.label_image(image, "morphology_classification")

    def count_trichomes(
        self,
        image: NDArray[np.uint8],
    ) -> VLMInferenceResult:
        """
        Count trichomes in image.

        NOTE: Moondream is weak at counting. Expect lower accuracy.
        Use qwen2vl or florence2 for counting tasks.
        """
        logger.warning(
            "Moondream counting accuracy is limited. "
            "Consider using qwen2vl or florence2 for counting tasks."
        )
        return self.label_image(image, "trichome_count")

    def _run_inference(
        self,
        image: NDArray[np.uint8],
        prompt_template: PromptTemplate,
        extra_context: str | None = None,
    ) -> VLMInferenceResult:
        """Core inference loop with retry logic."""
        import torch
        from PIL import Image as PILImage

        # Convert numpy array to PIL Image
        pil_image = PILImage.fromarray(image)

        user_prompt = prompt_template.format_user_prompt()
        if extra_context:
            user_prompt = f"{user_prompt}\n\nAdditional context: {extra_context}"

        # Build messages (Moondream uses specific chat format)
        messages = [
            {"role": "system", "content": prompt_template.system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        best_result: VLMInferenceResult | None = None
        retry_count = 0
        max_retries = self._config.retry_on_invalid_json

        for attempt in range(max_retries + 1):
            t_start = time.perf_counter()

            try:
                raw_response = self._call_model(pil_image, messages)
                t_end = time.perf_counter()
                inference_time = t_end - t_start

                is_valid, parsed = prompt_template.validate_response(raw_response)

                result = VLMInferenceResult(
                    prompt_name=prompt_template.name,
                    raw_response=raw_response,
                    parsed_response=parsed,
                    is_valid=is_valid,
                    inference_time_s=inference_time,
                    retry_count=retry_count,
                    model_id=self._config.model_id,
                )

                if is_valid:
                    return result

                # Store best attempt even if invalid
                if best_result is None or (not best_result.is_valid):
                    best_result = result

                if attempt < max_retries:
                    retry_count += 1
                    logger.debug(
                        "Invalid JSON response, retrying",
                        attempt=attempt + 1,
                        max_retries=max_retries,
                        raw_response_preview=raw_response[:200],
                    )
                    # Add stronger JSON constraint hint on retry
                    messages = [
                        {"role": "system", "content": prompt_template.system_prompt},
                        {
                            "role": "user",
                            "content": (
                                user_prompt
                                + "\n\nIMPORTANT: Respond with ONLY valid JSON. "
                                "No markdown, no explanation, just the JSON object."
                            ),
                        },
                    ]

            except Exception as e:
                t_end = time.perf_counter()
                logger.error(
                    "Moondream inference error",
                    attempt=attempt,
                    error=str(e),
                )
                if attempt == max_retries:
                    return VLMInferenceResult(
                        prompt_name=prompt_template.name,
                        raw_response=f"ERROR: {e}",
                        parsed_response=None,
                        is_valid=False,
                        inference_time_s=t_end - t_start,
                        retry_count=retry_count,
                        model_id=self._config.model_id,
                    )

        return best_result or VLMInferenceResult(
            prompt_name=prompt_template.name,
            raw_response="",
            parsed_response=None,
            is_valid=False,
            inference_time_s=0.0,
            model_id=self._config.model_id,
        )

    def _call_model(
        self,
        pil_image: Any,  # PIL.Image.Image
        messages: list[dict[str, str]],
    ) -> str:
        """
        Low-level model call.

        Moondream-2 uses its own generate interface:
        model.generate(image, question) → answer
        """
        import torch

        # Moondream-2 uses the native generate interface
        # Extract system + user text
        system_text = ""
        user_text = ""
        for msg in messages:
            if msg["role"] == "system":
                system_text = msg["content"]
            elif msg["role"] == "user":
                user_text = msg["content"]

        # Combined prompt for Moondream
        full_prompt = f"{system_text}\n\n{user_text}" if system_text else user_text

        with torch.inference_mode():
            # Moondream-2 API: encode_image + answer_question
            if hasattr(self._model, "encode_image") and hasattr(self._model, "answer_question"):
                # Native Moondream-2 API
                image_embeds = self._model.encode_image(pil_image)
                answer = self._model.answer_question(
                    image_embeds=image_embeds,
                    question=full_prompt,
                    tokenizer=self._tokenizer,
                    max_new_tokens=self._config.max_new_tokens,
                )
            else:
                # Fallback: standard transformers generate
                inputs = self._tokenizer(
                    text=full_prompt,
                    images=pil_image,
                    return_tensors="pt",
                )
                # Move inputs to device
                device = next(self._model.parameters()).device
                inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}

                output_ids = self._model.generate(
                    **inputs,
                    max_new_tokens=self._config.max_new_tokens,
                    do_sample=False,  # Greedy decoding for determinism
                    temperature=1.0,
                    pad_token_id=self._tokenizer.eos_token_id,
                )

                # Decode only newly generated tokens
                input_len = inputs["input_ids"].shape[1]
                new_tokens = output_ids[0][input_len:]
                answer = self._tokenizer.decode(new_tokens, skip_special_tokens=True)

        return answer.strip()

    def __repr__(self) -> str:
        status = "loaded" if self._is_loaded else "not loaded"
        return (
            f"MoondreamLabeler("
            f"model='{self._config.model_id}', "
            f"quant={self._config.quantization}, "
            f"vram={self.vram_requirement_gb:.1f}GB, "
            f"status={status})"
        )
