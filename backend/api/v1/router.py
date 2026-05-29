"""backend.api.v1.router — Aggregate all v1 API routes."""

from fastapi import APIRouter

from backend.api.v1 import (
    system,
    training,
    datasets,
    annotation,
    video,
    reports,
    models,
    inference,
    labelstudio,
    experiments,  # TDB-008: extracted from inline block
    setup,
    containers,
    model_tests,
    settings,
    tensorrt,
    morphology_training,
    dataset_streaming,
    distributed_training,
)

router = APIRouter()

router.include_router(setup.router)
router.include_router(containers.router)
router.include_router(system.router)
router.include_router(training.router)
router.include_router(datasets.router)
router.include_router(annotation.router)
router.include_router(video.router)
router.include_router(reports.router)
router.include_router(models.router)
router.include_router(inference.router)
router.include_router(model_tests.router)
router.include_router(labelstudio.router)
router.include_router(experiments.router)
router.include_router(settings.router)
router.include_router(tensorrt.router)
router.include_router(morphology_training.router)
router.include_router(dataset_streaming.router)
router.include_router(distributed_training.router)

# ── Optional sub-service routers ───────────────────────────────────

# training.api.router routes are merged into backend.api.v1.training — do not include here
# (avoids duplicated /training/start, /training/stop routes)

try:
    from segmentation.api.router import router as segmentation_router
    router.include_router(segmentation_router)
except Exception:
    pass

try:
    from active_learning.api.router import router as al_router
    router.include_router(al_router)
except Exception:
    pass

try:
    from annotation.api.router import router as annotation_pipeline_router
    router.include_router(annotation_pipeline_router)
except Exception:
    pass

try:
    from vlm_labeling.api.router import router as vlm_router
    router.include_router(vlm_router)
except Exception:
    pass

try:
    from backend.api.v1.vlm_providers import router as vlm_providers_router
    router.include_router(vlm_providers_router)
except Exception:
    pass

try:
    from backend.api.v1.ollama import router as ollama_router
    router.include_router(ollama_router)
except Exception:
    pass

try:
    from backend.api.v1.vlm_ensemble import router as vlm_ensemble_router
    router.include_router(vlm_ensemble_router)
except Exception:
    pass

try:
    from backend.api.v1.remote_compute import router as remote_compute_router
    router.include_router(remote_compute_router)
except Exception:
    pass

# ── New science module routers ─────────────────────────────────────

try:
    from focus.api.router import router as focus_router
    router.include_router(focus_router)
except Exception:
    pass

try:
    from maturity.api.router import router as maturity_router
    router.include_router(maturity_router)
except Exception:
    pass

try:
    from morphology.api.router import router as morphology_router
    router.include_router(morphology_router)
except Exception:
    pass

try:
    from measurement.api.router import router as measurement_router
    router.include_router(measurement_router)
except Exception:
    pass

try:
    from video_pipeline.api.router import router as video_pipeline_router
    router.include_router(video_pipeline_router)
except Exception:
    pass

try:
    from analytics.api.router import router as analytics_router
    router.include_router(analytics_router)
except Exception:
    pass

try:
    from backend.api.v1.tracking import router as tracking_router
    router.include_router(tracking_router)
except Exception:
    pass
