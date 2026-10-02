"""An agent's labelling session on one image: points → SAM2 masks → shape check → pending label record."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from vlm_labeling.agent.masks import head_only, polygon_of, shape_of
from vlm_labeling.agent.render import View, render_view
from vlm_labeling.agent.segmenter import PointSegmenter

KINDS = ("stalked", "sessile", "bulbous", "non-glandular")


@dataclass
class Head:
    id: int
    point: tuple[int, int]
    kind: str
    status: str                       # ok | trimmed | rejected
    box: tuple[int, int, int, int]    # x0, y0, x1, y1
    polygon: list[list[float]]
    diameter_px: float
    circularity: float


@dataclass
class LabelRecord:
    """What is saved next to the image: always pending until a person reviews it."""

    image: str
    width: int
    height: int
    heads: list[Head]
    annotator: str
    note: str = ""
    status: str = "pending_review"
    created_at: float = field(default_factory=time.time)
    tool: str = "ctip-agent-labeler/1"

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=1)


class AgentLabelSession:
    def __init__(self, image_path: str | Path, segmenter: PointSegmenter) -> None:
        self.path = Path(image_path)
        img = cv2.imread(str(self.path))
        if img is None:
            raise FileNotFoundError(f"cannot read image: {self.path}")
        self.image: NDArray[np.uint8] = img
        self.height, self.width = img.shape[:2]
        self._segmenter = segmenter
        self._heads: dict[int, Head] = {}
        self._next = 0

    @property
    def heads(self) -> list[Head]:
        return [self._heads[k] for k in sorted(self._heads)]

    def view(self, region: tuple[int, int, int, int] | None = None, grid: int | None = None, show_heads: bool = True) -> View:
        return render_view(self.image, region, [asdict(h) for h in self.heads if h.status != "rejected"] if show_heads else None, grid)

    def add(self, points: list[tuple[int, int]], kinds: list[str] | None = None) -> list[Head]:
        """Segment one head per point. Points outside the image raise; rejected masks are kept for the report."""
        if not points:
            return []
        kinds = kinds or ["stalked"] * len(points)
        if len(kinds) != len(points):
            raise ValueError("kinds must have one entry per point")
        for k in kinds:
            if k not in KINDS:
                raise ValueError(f"unknown kind {k!r}; use one of {', '.join(KINDS)}")
        pts = []
        for x, y in points:
            if not (0 <= x < self.width and 0 <= y < self.height):
                raise ValueError(f"point ({x}, {y}) is outside the image {self.width}x{self.height}")
            pts.append((int(x), int(y)))
        added = []
        for (x, y), kind, mask in zip(pts, kinds, self._segmenter.segment(self.image, pts), strict=True):
            mask, status = head_only(mask, (x, y))
            if status == "rejected" or not mask.any():
                head = Head(self._next, (x, y), kind, "rejected", (x, y, x, y), [], 0.0, 0.0)
            else:
                ys, xs = np.where(mask)
                shp = shape_of(mask)
                head = Head(self._next, (x, y), kind, status, (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1),
                            polygon_of(mask), round(float(2 * np.sqrt(mask.sum() / np.pi)), 1), round(shp.circularity, 2))
            self._heads[self._next] = head
            self._next += 1
            added.append(head)
        return added

    def remove(self, ids: list[int]) -> list[int]:
        return [i for i in ids if self._heads.pop(i, None) is not None]

    def set_kind(self, ids: list[int], kind: str) -> None:
        if kind not in KINDS:
            raise ValueError(f"unknown kind {kind!r}")
        for i in ids:
            if i in self._heads:
                self._heads[i].kind = kind

    def record(self, annotator: str, note: str = "") -> LabelRecord:
        return LabelRecord(str(self.path), self.width, self.height, [h for h in self.heads if h.status != "rejected"], annotator, note)

    def save(self, annotator: str, note: str = "", out_dir: str | Path | None = None) -> tuple[Path, Path]:
        """Write <image>.ctip-labels.json (provenance, polygons) and a YOLO txt with one class per morphology."""
        rec = self.record(annotator, note)
        out = Path(out_dir) if out_dir else self.path.parent / "agent_labels"
        out.mkdir(parents=True, exist_ok=True)
        js = out / f"{self.path.stem}.ctip-labels.json"
        js.write_text(rec.to_json())
        txt = out / f"{self.path.stem}.txt"
        lines = []
        for h in rec.heads:
            x0, y0, x1, y1 = h.box
            lines.append(f"{KINDS.index(h.kind)} {(x0 + x1) / 2 / self.width:.6f} {(y0 + y1) / 2 / self.height:.6f} "
                         f"{(x1 - x0) / self.width:.6f} {(y1 - y0) / self.height:.6f}")
        txt.write_text("\n".join(lines) + ("\n" if lines else ""))
        return js, txt

    def summary(self) -> str:
        hs = self.heads
        rows = [f"{h.id:>3}  ({h.point[0]:>5},{h.point[1]:>5})  {h.kind:<13} {h.status:<8} Ø {h.diameter_px:>5.1f}px  circ {h.circularity:.2f}"
                for h in hs]
        kept = sum(h.status != "rejected" for h in hs)
        return f"{self.path.name} {self.width}x{self.height}: {kept} heads kept, {len(hs) - kept} rejected\n" + "\n".join(rows)
