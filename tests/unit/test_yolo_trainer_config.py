"""TrainingConfig → Ultralytics arguments: the effective batch must reach the optimizer."""

from __future__ import annotations

from training.pipelines.yolo_trainer import TrainingConfig


def test_effective_batch_is_passed_as_nominal_batch():
    cfg = TrainingConfig(batch_size=4, gradient_accumulation_steps=4)
    kw = cfg.to_ultralytics_kwargs()
    assert kw["batch"] == 4
    assert kw["nbs"] == 16          # Ultralytics steps the optimizer every nbs / batch iterations


def test_tiny_dataset_steps_every_iteration():
    kw = TrainingConfig(batch_size=2, gradient_accumulation_steps=1).to_ultralytics_kwargs()
    assert kw["nbs"] == kw["batch"] == 2
