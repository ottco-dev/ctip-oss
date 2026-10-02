"""
CTIP labelling tools for AI agents over MCP (stdio).

    claude mcp add ctip-label -- .venv/bin/python -m apps.mcp.label_server     # or use the repo's .mcp.json

The agent opens an image, reads coordinates off a grid, zooms where heads are small, places one point per
glandular head, checks SAM2's masks, and saves the result as a *pending* label (JSON + YOLO txt) and/or pushes it to
Label Studio as a prediction. A person reviews every label before it can become training data.

Images are read only from CTIP_LABEL_ROOT (default: the configured images dir, ./data/images).
"""

from __future__ import annotations

import functools
import os
from pathlib import Path

from mcp.server.mcpserver import Image, MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from vlm_labeling.agent.segmenter import SAM2PointSegmenter
from vlm_labeling.agent.session import KINDS, AgentLabelSession

GUIDE = """You label glandular trichome heads on microscope photos for CTIP.
1. open_image, then zoom (view with a region) wherever heads are small or dense - the grid numbers are original pixels.
2. add_heads with one point at the CENTRE of each head (not the stalk). kind: stalked (head on a visible stalk),
   sessile (head directly on the surface), bulbous (tiny head, few cells), non-glandular (hair, no head).
3. Check the returned overlay: remove_heads for masks on background, stalks or two heads at once; add missed heads.
   'trimmed' means the shape check cut a stalk off - fine. 'rejected' masks are not kept.
4. Label only what you can see. Skip blurry heads you cannot place within a few pixels. Never guess maturity or
   cannabinoid content.
5. save (pending files) and/or push_to_label_studio. Say how many heads you labelled and what you were unsure about.
A person reviews every label; nothing you save goes straight into training data."""

server = MCPServer(name="ctip-label", title="CTIP trichome labelling", instructions=GUIDE, version="1")
_sessions: dict[str, AgentLabelSession] = {}
_segmenter: SAM2PointSegmenter | None = None


def _explained(fn):
    """Turn expected failures into a message the agent can act on (instead of a generic tool error)."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except (ValueError, FileNotFoundError, PermissionError) as e:
            raise ToolError(str(e)) from e
        except Exception as e:                 # network errors from Label Studio and the like
            if type(e).__module__.startswith("requests"):
                raise ToolError(f"Label Studio: {e}") from e
            raise
    return wrapper


def _root() -> Path:
    root = os.environ.get("CTIP_LABEL_ROOT")
    if not root:
        try:
            from backend.config import get_settings

            root = get_settings().images_dir
        except Exception:
            root = "data/images"
    return Path(root).expanduser().resolve()


def _resolve(image: str) -> Path:
    """Image path inside the label root (no escaping it with ../ or absolute paths elsewhere)."""
    root = _root()
    p = (root / image).resolve() if not Path(image).is_absolute() else Path(image).resolve()
    if root != p and root not in p.parents:
        raise ValueError(f"{image} is outside the label root {root}")
    if not p.is_file():
        raise FileNotFoundError(f"no such image: {p.relative_to(root) if root in p.parents else p}")
    return p


def _session(image: str) -> AgentLabelSession:
    global _segmenter
    p = _resolve(image)
    key = str(p)
    if key not in _sessions:
        if _segmenter is None:
            weights = Path(os.environ.get("CTIP_SAM2_WEIGHTS", _root().parent / "models" / "sam2_t.pt"))
            _segmenter = SAM2PointSegmenter(weights if weights.exists() else "sam2_t.pt")
        _sessions[key] = AgentLabelSession(p, _segmenter)
    return _sessions[key]


def _view_reply(s: AgentLabelSession, region=None, grid=None, title: str = "") -> list:
    v = s.view(tuple(region) if region else None, grid)
    x0, y0, x1, y1 = v.region
    text = (f"{title}{s.path.name}: {s.width}x{s.height} px. Showing x {x0}-{x1}, y {y0}-{y1} at {v.scale:.2f}x; "
            f"grid every {v.grid} px, labels are original pixel coordinates.")
    return [Image(data=v.jpeg, format="jpeg"), text]


@server.tool(description="List images under the label root, with how many heads are already saved for each.")
@_explained
def list_images(folder: str = "", pattern: str = "*.jpg") -> str:
    root = _root()
    base = _resolve_dir(folder)
    rows = []
    for p in sorted(base.rglob(pattern))[:500]:
        saved = p.parent / "agent_labels" / f"{p.stem}.ctip-labels.json"
        rows.append(f"{p.relative_to(root)}{'  [labels saved]' if saved.exists() else ''}")
    return f"label root: {root}\n" + ("\n".join(rows) if rows else "no images found")


def _resolve_dir(folder: str) -> Path:
    root = _root()
    p = (root / folder).resolve()
    if root != p and root not in p.parents:
        raise ValueError(f"{folder} is outside the label root {root}")
    return p


@server.tool(description="Open an image for labelling and show it with a coordinate grid (original pixels).")
@_explained
def open_image(image: str) -> list:
    return _view_reply(_session(image), title="Opened ")


@server.tool(description="Show a region (zoom). x0,y0,x1,y1 in original pixels; grid optional spacing in px.")
@_explained
def view(image: str, x0: int = 0, y0: int = 0, x1: int = 0, y1: int = 0, grid: int = 0) -> list:
    s = _session(image)
    region = (x0, y0, x1 or s.width, y1 or s.height)
    return _view_reply(s, region, grid or None)


@server.tool(description="Add heads: one [x, y] point per head centre (original px); kinds per point: "
                         "stalked | sessile | bulbous | non-glandular (default stalked). Returns overlay + table.")
@_explained
def add_heads(image: str, points: list[list[int]], kinds: list[str] | None = None) -> list:
    s = _session(image)
    added = s.add([(int(p[0]), int(p[1])) for p in points], kinds)
    rejected = [h.id for h in added if h.status == "rejected"]
    reply = _view_reply(s, title=f"Added {len(added) - len(rejected)} heads"
                        + (f", rejected {rejected} (not round - check the point)" if rejected else "") + ". ")
    return [*reply, s.summary()]


@server.tool(description="Remove heads by id (wrong mask, background, duplicate).")
@_explained
def remove_heads(image: str, ids: list[int]) -> list:
    s = _session(image)
    gone = s.remove(ids)
    return [*_view_reply(s, title=f"Removed {gone}. "), s.summary()]


@server.tool(description=f"Change the morphology kind of heads: {', '.join(KINDS)}.")
@_explained
def set_kind(image: str, ids: list[int], kind: str) -> str:
    s = _session(image)
    s.set_kind(ids, kind)
    return s.summary()


@server.tool(description="Save the labels as pending review: <image>.ctip-labels.json (provenance, polygons) and a "
                         "YOLO .txt in agent_labels/ next to the image.")
@_explained
def save(image: str, note: str = "") -> str:
    s = _session(image)
    annotator = "claude-code" + (f" ({os.environ['CLAUDE_MODEL']})" if os.environ.get("CLAUDE_MODEL") else "")
    js, txt = s.save(annotator, note)
    return f"saved {len(s.record(annotator).heads)} heads as pending review:\n{js}\n{txt}"


@server.tool(description="Push the labels to Label Studio as predictions (model_version claude-code-agent) for "
                         "human review. Needs LABEL_STUDIO_URL and LABEL_STUDIO_API_KEY (env or CTIP .env).")
@_explained
def push_to_label_studio(image: str, project: str = "CTIP - agent labels (review)") -> str:
    from backend.api.v1.setup import TRICHOME_LABEL_CONFIG
    from vlm_labeling.agent.label_studio import LabelStudio

    url = os.environ.get("LABEL_STUDIO_URL")
    token = os.environ.get("LABEL_STUDIO_API_KEY")
    if not (url and token):
        from backend.config import get_settings

        st = get_settings()
        url, token = url or st.label_studio_url, token or st.label_studio_api_key
    if not token:
        raise ValueError("no Label Studio API key - set LABEL_STUDIO_API_KEY")
    s = _session(image)
    ls = LabelStudio(url.rstrip("/"), token)
    pid = ls.project(project, TRICHOME_LABEL_CONFIG)
    task = ls.push(pid, s.path, s.record("claude-code"))
    return f"task {task} in project {pid} ({project}) with {len(s.record('claude-code').heads)} predicted heads - " \
           f"review it at {url.rstrip('/')}/projects/{pid}/data?task={task}"


def main() -> None:
    server.run("stdio")


if __name__ == "__main__":
    main()
