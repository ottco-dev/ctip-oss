"""
backend.api.v1.morphology_training — REST API for CNN morphology classifier training.

Endpoints:
    POST /morphology/training/start    — Start CNN training as background task
    GET  /morphology/training/status   — Current training status
    POST /morphology/training/evaluate — Evaluate a saved checkpoint
    POST /morphology/training/export   — Export checkpoint to ONNX
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/morphology/training", tags=["morphology-training"])

# ── In-process training state (single-job model) ──────────────────────────────

_training_state: dict[str, Any] = {
    "state": "idle",          # idle | running | completed | failed
    "trainer": None,
    "config": None,
    "error": None,
    "summary": None,
}


# ── Request / response schemas ────────────────────────────────────────────────


class TrainingStartRequest(BaseModel):
    model_arch: str = Field(
        default="efficientnet_b0",
        description="Backbone: 'efficientnet_b0' or 'mobilenet_v3_small'",
    )
    num_classes: int = Field(default=4, ge=2, le=100)
    input_size: int = Field(default=224, ge=32, le=1024)
    batch_size: int = Field(default=32, ge=1, le=256)
    learning_rate: float = Field(default=1e-4, gt=0.0)
    epochs: int = Field(default=50, ge=1, le=500)
    dropout: float = Field(default=0.3, ge=0.0, lt=1.0)
    data_dir: str = Field(default="./data/morphology_crops")
    output_dir: str = Field(default="./data/models/morphology")
    use_fp16: bool = True
    augment: bool = True
    early_stopping_patience: int = Field(default=10, ge=1)
    val_split: float = Field(default=0.2, gt=0.0, lt=1.0)
    seed: int = 42


class EvaluateRequest(BaseModel):
    model_path: str = Field(description="Path to .pt checkpoint to evaluate")
    data_dir: str = Field(
        default="./data/morphology_crops",
        description="Root data directory (same structure as training)",
    )


class ExportRequest(BaseModel):
    model_path: str = Field(description="Path to .pt checkpoint")
    output_path: str = Field(
        default="./data/models/morphology/morphology_cnn.onnx",
        description="Destination .onnx file path",
    )


# ── Background worker ─────────────────────────────────────────────────────────


async def _run_training(config_dict: dict) -> None:
    """Async wrapper that runs the blocking CNN training in a thread pool."""
    from morphology.training.cnn_trainer import MorphologyCNNConfig, MorphologyCNNTrainer

    try:
        _training_state["state"] = "running"
        _training_state["error"] = None
        _training_state["summary"] = None

        config = MorphologyCNNConfig(**config_dict)
        trainer = MorphologyCNNTrainer(config)
        _training_state["trainer"] = trainer
        _training_state["config"] = config_dict

        loop = asyncio.get_event_loop()

        def _progress_cb(status: dict) -> None:
            _training_state.update(status)

        summary = await loop.run_in_executor(
            None,
            lambda: trainer.train(progress_callback=_progress_cb),
        )

        _training_state["state"] = "completed"
        _training_state["summary"] = summary
        logger.info("Morphology CNN training completed — best val acc: %.4f", summary.get("best_val_acc", 0.0))

    except Exception as exc:  # noqa: BLE001
        _training_state["state"] = "failed"
        _training_state["error"] = str(exc)
        logger.exception("Morphology CNN training failed: %s", exc)


# ── Endpoints ─────────────────────────────────────────────────────────────────


@router.post("/start")
async def training_start(
    request: TrainingStartRequest,
    background_tasks: BackgroundTasks,
) -> dict[str, Any]:
    """
    Start CNN morphology classification training as a background task.

    Only one training job may run at a time. Returns HTTP 409 if a job is
    already running.
    """
    if _training_state["state"] == "running":
        raise HTTPException(
            status_code=409,
            detail="A training job is already running. "
                   "Wait for it to complete or check /morphology/training/status.",
        )

    config_dict = request.model_dump()
    background_tasks.add_task(_run_training, config_dict)

    _training_state["state"] = "queued"
    _training_state["config"] = config_dict
    _training_state["error"] = None
    _training_state["summary"] = None

    return {
        "status": "queued",
        "message": "Morphology CNN training started in background",
        "config": config_dict,
    }


@router.get("/status")
async def training_status() -> dict[str, Any]:
    """
    Return the current state of the training job.

    States: idle | queued | running | completed | failed
    """
    trainer = _training_state.get("trainer")
    per_epoch: dict = {}
    if trainer is not None:
        per_epoch = trainer.training_history

    return {
        "state": _training_state.get("state", "idle"),
        "epoch": _training_state.get("epoch"),
        "train_loss": _training_state.get("train_loss"),
        "val_loss": _training_state.get("val_loss"),
        "train_acc": _training_state.get("train_acc"),
        "val_acc": _training_state.get("val_acc"),
        "best_val_loss": _training_state.get("best_val_loss"),
        "early_stop_counter": _training_state.get("early_stop_counter"),
        "error": _training_state.get("error"),
        "summary": _training_state.get("summary"),
        "config": _training_state.get("config"),
        "history": per_epoch,
    }


@router.post("/evaluate")
async def training_evaluate(request: EvaluateRequest) -> dict[str, Any]:
    """
    Evaluate a saved morphology CNN checkpoint.

    Runs per-class accuracy, confusion matrix, top-1 / top-5 accuracy,
    precision, recall, and F1 for each class.
    """
    from morphology.training.cnn_trainer import MorphologyCNNConfig, MorphologyCNNTrainer

    config = MorphologyCNNConfig(data_dir=request.data_dir)
    trainer = MorphologyCNNTrainer(config)

    loop = asyncio.get_event_loop()
    try:
        metrics = await loop.run_in_executor(
            None,
            lambda: trainer.evaluate(request.model_path),
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {exc}") from exc

    return {"status": "ok", "metrics": metrics}


@router.post("/export")
async def training_export(request: ExportRequest) -> dict[str, Any]:
    """
    Export a trained morphology CNN checkpoint to ONNX format.

    The exported model can be loaded by the inference pipeline for
    production trichome morphology classification.
    """
    from morphology.training.cnn_trainer import MorphologyCNNConfig, MorphologyCNNTrainer

    config = MorphologyCNNConfig()
    trainer = MorphologyCNNTrainer(config)

    loop = asyncio.get_event_loop()
    try:
        onnx_path = await loop.run_in_executor(
            None,
            lambda: trainer.export_onnx(request.model_path, request.output_path),
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"ONNX export failed: {exc}") from exc

    return {
        "status": "ok",
        "onnx_path": onnx_path,
        "message": f"Model exported to {onnx_path}",
    }
