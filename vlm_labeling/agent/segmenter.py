"""Point-prompted segmentation with SAM2 (Ultralytics implementation, weights downloaded on first use)."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

BATCH = 48  # points per SAM2 call - 256 overflows 8 GB VRAM at 1920 px


class PointSegmenter(Protocol):
    def segment(self, image_bgr: NDArray[np.uint8], points: list[tuple[int, int]]) -> list[NDArray[np.bool_]]:
        """One boolean mask (H, W) per point."""
        ...


class SAM2PointSegmenter:
    def __init__(self, weights: str | Path = "sam2_t.pt", device: str | None = None) -> None:
        self._weights = str(weights)
        self._device = device
        self._model = None

    def _load(self):
        if self._model is None:
            from ultralytics import SAM

            self._model = SAM(self._weights)
        return self._model

    def segment(self, image_bgr, points):
        model = self._load()
        masks: list[NDArray[np.bool_]] = []
        for i in range(0, len(points), BATCH):
            batch = [list(map(int, p)) for p in points[i:i + BATCH]]
            kwargs = {"device": self._device} if self._device else {}
            r = model(image_bgr, points=[[p] for p in batch], labels=[[1] for _ in batch], verbose=False, **kwargs)[0]
            if r.masks is None:
                masks.extend(np.zeros(image_bgr.shape[:2], bool) for _ in batch)
            else:
                masks.extend(m.astype(bool) for m in r.masks.data.cpu().numpy())
        return masks
