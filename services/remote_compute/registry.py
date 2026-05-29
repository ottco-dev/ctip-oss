"""
services.remote_compute.registry — Remote compute backend factory.

USAGE:
    from services.remote_compute.registry import get_compute_backend, list_backends

    backends = list_backends()                  # metadata for all backends
    backend = get_compute_backend("modal")      # instantiated, ready to use
    result = await backend.run_vlm_inference(image, prompt, model_id)

BACKEND IDs:
    "modal"      — Modal serverless GPU ($30/month free credit)
    "replicate"  — Replicate hosted models (pay-per-prediction)

ENV VARS:
    MODAL_TOKEN_ID        / MODAL_TOKEN_SECRET  (from: modal token new)
    REPLICATE_API_KEY                           (from: replicate.com)
"""

from __future__ import annotations

import os
from typing import Any

from services.remote_compute.base import RemoteComputeBackend, ComputeBackendInfo

_ENV_MAP: dict[str, list[str]] = {
    "modal":     ["MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET"],
    "replicate": ["REPLICATE_API_KEY"],
}


def list_backends() -> list[dict[str, Any]]:
    """Return metadata + availability for all remote compute backends."""
    from services.remote_compute.modal_backend import _INFO as MODAL_INFO
    from services.remote_compute.replicate_backend import _INFO as REPLICATE_INFO

    infos = {
        "modal":     MODAL_INFO,
        "replicate": REPLICATE_INFO,
    }

    result = []
    for bid, info in infos.items():
        env_vars = _ENV_MAP.get(bid, [])
        has_keys = all(bool(os.getenv(v, "")) for v in env_vars)
        result.append({
            "backend_id": bid,
            "name": info.name,
            "kind": info.kind.value,
            "free_tier": info.free_tier,
            "free_tier_note": info.free_tier_note,
            "signup_url": info.signup_url,
            "gpu_tiers": [g.value for g in info.gpu_tiers],
            "cost_per_hour": info.cost_per_hour,
            "available": has_keys,
            "required_env_vars": env_vars,
        })

    return result


def get_compute_backend(backend_id: str) -> RemoteComputeBackend:
    """
    Get an instantiated remote compute backend.

    Args:
        backend_id: "modal" or "replicate"

    Returns:
        Configured RemoteComputeBackend.

    Raises:
        ValueError: If backend_id is unknown.
        ImportError: If required package not installed.
    """
    if backend_id == "modal":
        from services.remote_compute.modal_backend import ModalBackend
        return ModalBackend()
    elif backend_id == "replicate":
        from services.remote_compute.replicate_backend import ReplicateBackend
        return ReplicateBackend()
    else:
        raise ValueError(
            f"Unknown compute backend: '{backend_id}'. "
            "Available: modal, replicate"
        )
