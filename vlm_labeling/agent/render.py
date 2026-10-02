"""Images for an agent: a coordinate grid in original pixels, optional zoom window, numbered head overlays."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

MAX_SIDE = 1280
KIND_COLOURS = {  # BGR, matching the web UI's trichome colours
    "stalked": (238, 211, 34), "sessile": (153, 211, 52), "bulbous": (250, 139, 167), "non-glandular": (60, 146, 251),
}


@dataclass(frozen=True)
class View:
    jpeg: bytes
    region: tuple[int, int, int, int]  # x0, y0, x1, y1 in original pixels
    scale: float                       # rendered px per original px
    grid: int                          # grid spacing in original px


def nice_grid(span: int) -> int:
    """About 8-12 grid lines across the visible span, on round numbers."""
    for step in (10, 20, 25, 50, 100, 200, 250, 500, 1000):
        if span / step <= 12:
            return step
    return 2000


def render_view(
    image_bgr: NDArray[np.uint8],
    region: tuple[int, int, int, int] | None = None,
    heads: list[dict] | None = None,
    grid: int | None = None,
) -> View:
    """
    Crop to `region` (original pixels), scale to at most 1280 px, draw a labelled grid whose numbers are ORIGINAL
    pixel coordinates (so the agent can read point positions directly), and outline existing heads with their ids.
    """
    H, W = image_bgr.shape[:2]
    x0, y0, x1, y1 = region or (0, 0, W, H)
    x0, y0 = max(0, int(x0)), max(0, int(y0))
    x1, y1 = min(W, int(x1)), min(H, int(y1))
    if x1 - x0 < 16 or y1 - y0 < 16:
        raise ValueError(f"region too small or outside the image ({W}x{H}): {(x0, y0, x1, y1)}")
    crop = image_bgr[y0:y1, x0:x1]
    s = min(MAX_SIDE / max(crop.shape[:2]), 4.0)
    out = cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
    step = grid or nice_grid(max(x1 - x0, y1 - y0))
    font, fs = cv2.FONT_HERSHEY_SIMPLEX, 0.45

    def label(text: str, org: tuple[int, int]) -> None:
        cv2.putText(out, text, org, font, fs, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, text, org, font, fs, (255, 255, 255), 1, cv2.LINE_AA)

    overlay = out.copy()
    for gx in range((x0 // step + 1) * step, x1, step):
        px = int((gx - x0) * s)
        cv2.line(overlay, (px, 0), (px, out.shape[0]), (255, 255, 255), 1)
        label(str(gx), (px + 3, 14))
    for gy in range((y0 // step + 1) * step, y1, step):
        py = int((gy - y0) * s)
        cv2.line(overlay, (0, py), (out.shape[1], py), (255, 255, 255), 1)
        label(str(gy), (3, py - 4))
    out = cv2.addWeighted(overlay, 0.35, out, 0.65, 0)

    for h in heads or []:
        colour = KIND_COLOURS.get(h.get("kind", ""), (255, 0, 255))
        poly = np.array(h.get("polygon") or [], dtype=np.float32)
        if len(poly) >= 3:
            pts = ((poly - [x0, y0]) * s).astype(np.int32)
            cv2.polylines(out, [pts], True, colour, 2, cv2.LINE_AA)
        px, py = int((h["point"][0] - x0) * s), int((h["point"][1] - y0) * s)
        cv2.circle(out, (px, py), 3, (0, 0, 255), -1)
        label(str(h["id"]), (px + 5, py - 5))

    ok, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        raise RuntimeError("could not encode the view")
    return View(buf.tobytes(), (x0, y0, x1, y1), s, step)
