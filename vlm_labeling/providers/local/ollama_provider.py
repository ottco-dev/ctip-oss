"""
vlm_labeling.providers.local.ollama_provider — Ollama local LLM provider.

NOT a VLM — does not process images.  Takes structured analysis JSON produced
by the CTIP detection/maturity pipeline and generates a human-readable
scientific report narrative via Ollama's local REST API.

Ollama exposes an OpenAI-compatible HTTP interface at http://localhost:11434.
No GPU semaphore is needed here: Ollama manages its own GPU context and runs
as a separate process.

SCIENTIFIC CONSTRAINT:
    Narratives MUST NOT claim or imply THC / cannabinoid concentrations.
    Optical maturity assessment only.  THC-related fields are stripped from
    analysis_result before any prompt is constructed (see build_prompt).
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from typing import Any

import aiohttp

from shared.logging.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Forbidden fields — stripped before prompt construction
# ---------------------------------------------------------------------------
_THC_FORBIDDEN_KEYS: frozenset[str] = frozenset(
    {
        "thc_percentage",
        "thc_content",
        "thc_level",
        "thc_concentration",
        "thc_estimate",
        "cannabinoid_content",
        "cannabinoid_percentage",
        "cannabinoid_concentration",
        "cannabinoid_level",
        "cannabinoid_estimate",
        "potency",
        "potency_estimate",
        "potency_percentage",
        "terpene_content",
        "terpene_percentage",
    }
)

# ---------------------------------------------------------------------------
# Style and language prompt fragments
# ---------------------------------------------------------------------------
_STYLE_INSTRUCTIONS: dict[str, str] = {
    "scientific": (
        "Write a concise scientific report narrative (3–5 paragraphs) suitable "
        "for a peer-reviewed microscopy analysis paper.  Use precise botanical "
        "and analytical chemistry terminology.  Cite specific counts, "
        "percentages and confidence values from the data."
    ),
    "summary": (
        "Write a brief plain-language summary (1–2 paragraphs) suitable for a "
        "cultivator or operator.  Avoid jargon; focus on actionable insights "
        "about optical trichome maturity."
    ),
    "technical": (
        "Write a detailed technical report narrative (4–6 paragraphs) suitable "
        "for an internal QA/QC log.  Include detection statistics, confidence "
        "intervals, morphology breakdown, and optical maturity distribution with "
        "exact numerical values."
    ),
}

_LANGUAGE_INSTRUCTIONS: dict[str, str] = {
    "en": "Respond exclusively in English.",
    "de": "Respond exclusively in German (Deutsch).",
    "es": "Respond exclusively in Spanish (Español).",
}

_DEFAULT_SYSTEM_PROMPT = (
    "You are a scientific report writer for a cannabis trichome microscopy "
    "analysis platform (CTIP).  You receive structured analysis results and "
    "produce clear, precise, scientifically defensible narratives.  "
    "CRITICAL CONSTRAINT: You must NEVER mention, estimate, imply or speculate "
    "about THC content, cannabinoid concentrations or potency.  Your role is "
    "to describe optical trichome maturity observations only."
)


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class OllamaConfig:
    """Runtime configuration for OllamaProvider."""

    base_url: str = "http://localhost:11434"
    model: str = "llama3.2:3b"
    timeout_s: float = 30.0
    max_tokens: int = 1024
    temperature: float = 0.3          # low temperature for reproducible scientific text
    system_prompt: str = ""           # empty → use _DEFAULT_SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------

class OllamaProvider:
    """
    Ollama local LLM for structured report narrative generation.

    NOT a VLM — does not process images.  Takes structured analysis JSON
    and returns a human-readable scientific narrative.

    Communication: HTTP POST to localhost:11434 (ollama REST API).
    No GPU semaphore needed: Ollama manages its own GPU context.
    """

    def __init__(self, config: OllamaConfig | None = None) -> None:
        self._config = config or OllamaConfig()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def config(self) -> OllamaConfig:
        return self._config

    # ------------------------------------------------------------------
    # Availability / discovery
    # ------------------------------------------------------------------

    async def is_available(self) -> bool:
        """
        Probe Ollama availability.

        Sends GET /api/tags.  Returns True iff Ollama responds with HTTP 200.
        Returns False on any connection error or non-200 status.
        """
        url = f"{self._config.base_url}/api/tags"
        timeout = aiohttp.ClientTimeout(total=5.0)
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url) as response:
                    return response.status == 200
        except Exception as exc:
            logger.debug("Ollama availability check failed", error=str(exc))
            return False

    async def list_models(self) -> list[str]:
        """
        Return installed model names from GET /api/tags.

        Returns an empty list on any error (Ollama not running, parse error, etc.).
        """
        url = f"{self._config.base_url}/api/tags"
        timeout = aiohttp.ClientTimeout(total=10.0)
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url) as response:
                    if response.status != 200:
                        logger.warning(
                            "Ollama /api/tags returned non-200",
                            status=response.status,
                        )
                        return []
                    data = await response.json()
                    models = data.get("models", [])
                    return [m["name"] for m in models if "name" in m]
        except Exception as exc:
            logger.warning("Failed to list Ollama models", error=str(exc))
            return []

    # ------------------------------------------------------------------
    # Narrative generation
    # ------------------------------------------------------------------

    def build_prompt(
        self,
        analysis_result: dict[str, Any],
        style: str,
        language: str,
    ) -> str:
        """
        Construct the user prompt for narrative generation.

        Strips all THC/cannabinoid-related fields from analysis_result before
        embedding data in the prompt.  This is a hard safety requirement.

        Args:
            analysis_result: Structured dict from the CTIP detection pipeline.
                Expected keys:
                    total_detections: int
                    type_distribution: dict
                    maturity_distribution: dict
                    harvest_recommendation: str (optical maturity only)
                    confidence_stats: dict
                    session_id: str
                    timestamp: str
            style: One of "scientific" | "summary" | "technical"
            language: One of "en" | "de" | "es"

        Returns:
            Formatted prompt string ready to send to Ollama.
        """
        # --- Safety: strip forbidden keys from a shallow copy ---
        safe_result = {
            k: v
            for k, v in analysis_result.items()
            if k.lower() not in _THC_FORBIDDEN_KEYS
        }

        # Log if any keys were filtered (audit trail)
        removed = set(analysis_result.keys()) - set(safe_result.keys())
        if removed:
            logger.warning(
                "OllamaProvider: stripped THC/cannabinoid fields from analysis_result "
                "before prompt construction (scientific constraint enforced)",
                removed_keys=sorted(removed),
            )

        style_instruction = _STYLE_INSTRUCTIONS.get(
            style, _STYLE_INSTRUCTIONS["scientific"]
        )
        language_instruction = _LANGUAGE_INSTRUCTIONS.get(
            language, _LANGUAGE_INSTRUCTIONS["en"]
        )

        try:
            data_block = json.dumps(safe_result, indent=2, default=str)
        except (TypeError, ValueError) as exc:
            logger.warning("Could not JSON-serialise analysis_result", error=str(exc))
            data_block = str(safe_result)

        prompt = (
            f"ANALYSIS DATA:\n"
            f"```json\n{data_block}\n```\n\n"
            f"INSTRUCTIONS:\n"
            f"{style_instruction}\n\n"
            f"{language_instruction}\n\n"
            f"MANDATORY CONSTRAINT: This is a microscopy optical analysis only.  "
            f"Do NOT include any claims about THC, cannabinoid concentrations, "
            f"pharmaceutical strength, or chemical content of any kind.  "
            f"Focus exclusively on what is optically observable in the trichome "
            f"microscopy analysis."
        )
        return prompt

    async def generate_narrative(
        self,
        analysis_result: dict[str, Any],
        style: str = "scientific",
        language: str = "en",
    ) -> str:
        """
        Generate a human-readable narrative from a structured analysis result.

        Calls POST /api/chat (non-streaming) on the local Ollama instance.

        Args:
            analysis_result: Structured dict from the CTIP detection pipeline.
            style: Narrative style — "scientific" | "summary" | "technical"
            language: Output language — "en" | "de" | "es"

        Returns:
            Generated narrative string.

        Raises:
            aiohttp.ClientError: On network/connection failure.
            asyncio.TimeoutError: If Ollama does not respond within timeout_s.
            RuntimeError: If Ollama returns a non-200 status or unexpected payload.
        """
        system = self._config.system_prompt or _DEFAULT_SYSTEM_PROMPT
        user_prompt = self.build_prompt(analysis_result, style, language)

        payload = {
            "model": self._config.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user_prompt},
            ],
            "options": {
                "temperature": self._config.temperature,
                "num_predict": self._config.max_tokens,
            },
            "stream": False,
        }

        url = f"{self._config.base_url}/api/chat"
        timeout = aiohttp.ClientTimeout(total=self._config.timeout_s)

        logger.debug(
            "Sending narrative generation request to Ollama",
            model=self._config.model,
            style=style,
            language=language,
        )

        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(url, json=payload) as response:
                if response.status != 200:
                    body = await response.text()
                    raise RuntimeError(
                        f"Ollama /api/chat returned HTTP {response.status}: {body[:500]}"
                    )
                data = await response.json()

        # Ollama /api/chat non-streaming response structure:
        # {"message": {"role": "assistant", "content": "..."}, ...}
        try:
            narrative = data["message"]["content"]
        except (KeyError, TypeError) as exc:
            raise RuntimeError(
                f"Unexpected Ollama response structure: {exc}. Got keys: {list(data.keys())}"
            ) from exc

        logger.info(
            "Narrative generated",
            model=self._config.model,
            style=style,
            language=language,
            chars=len(narrative),
        )
        return narrative

    # ------------------------------------------------------------------
    # Model management
    # ------------------------------------------------------------------

    async def pull_model(self, model_name: str) -> AsyncGenerator[dict[str, Any], None]:
        """
        Stream model download progress from POST /api/pull.

        Yields progress dicts of the form::

            {"status": "downloading", "completed": int, "total": int}

        or simply ``{"status": "success"}`` when the pull completes.

        Raises:
            RuntimeError: If Ollama returns a non-200 status.
        """
        url = f"{self._config.base_url}/api/pull"
        payload = {"name": model_name, "stream": True}
        # Model pulls can take minutes — use a long timeout
        timeout = aiohttp.ClientTimeout(total=600.0)

        logger.info("Starting Ollama model pull", model=model_name)

        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(url, json=payload) as response:
                if response.status != 200:
                    body = await response.text()
                    raise RuntimeError(
                        f"Ollama /api/pull returned HTTP {response.status}: {body[:500]}"
                    )
                async for line in response.content:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    yield chunk
                    if chunk.get("status") == "success":
                        logger.info("Ollama model pull complete", model=model_name)
                        return
