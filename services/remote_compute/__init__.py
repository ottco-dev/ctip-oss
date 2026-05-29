"""
services.remote_compute — Optional remote GPU compute offloading.

Allows compute-intensive tasks (training, batch inference, large VLMs) to run
on cloud GPU infrastructure instead of the local RTX 4060.

Supported backends:
    modal       — Serverless GPU (A10G/A100). Free $30/month. Best for training.
    replicate   — Pay-per-prediction. Hosted open-source models.
    hf_spaces   — Hugging Face Inference Endpoints. Free shared, paid dedicated.

Use cases:
    - Training runs that exceed local VRAM (>8GB models)
    - Large batch VLM inference (100k+ images)
    - SAM2 large / Florence-2 large without local GPU
    - Parallel model benchmarking

Usage:
    from services.remote_compute import get_compute_backend
    backend = get_compute_backend("modal")  # or "replicate", "hf_spaces"
    result = await backend.run_inference(image, model_id="sam2-large")
"""
