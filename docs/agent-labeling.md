# Labelling with Claude Code

CTIP ships an MCP server that turns an AI coding agent — Claude Code, or any MCP client with vision — into a
trichome annotator. The agent does what a careful person does with a click tool: look, zoom, click the centre of
each glandular head, check the mask, fix mistakes. SAM2 draws the outlines; a person approves every label.

```
image ──► open_image / view (grid in original pixels, zoom)
      ──► add_heads(points, kinds) ──► SAM2-tiny masks ──► shape check (stalks trimmed, odd shapes rejected)
      ──► remove_heads / set_kind ──► save (pending JSON + YOLO txt)  ──► person reviews
                                   └► push_to_label_studio (prediction "claude-code-agent") ──► person reviews
```

## Setup

```bash
uv pip install -e ".[dev,agent]"          # mcp SDK; SAM2 weights (sam2_t.pt, 74 MB) download on first use
claude                                    # in the repo: .mcp.json registers the "ctip-label" server
/label-trichomes commons_trichomes/       # slash command with the labelling guideline
```

Outside the repo: `claude mcp add ctip-label -- /path/to/ctip-oss/.venv/bin/python -m apps.mcp.label_server`
(run it from the repo directory).

| Variable | Default | Meaning |
|---|---|---|
| `CTIP_LABEL_ROOT` | `IMAGES_DIR` (`./data/images`) | the only folder the tools may read |
| `CTIP_SAM2_WEIGHTS` | `data/models/sam2_t.pt`, else download | SAM2 checkpoint |
| `LABEL_STUDIO_URL`, `LABEL_STUDIO_API_KEY` | CTIP `.env` | for `push_to_label_studio` |
| `CLAUDE_MODEL` | – | recorded as the annotator in saved labels |

## Tools

| Tool | What it does |
|---|---|
| `list_images(folder, pattern)` | images under the label root, marked when labels are saved |
| `open_image(image)` | starts a session, returns the image with a coordinate grid |
| `view(image, x0, y0, x1, y1, grid)` | zoom into a region (up to 4×); grid numbers stay original pixels |
| `add_heads(image, points, kinds)` | one point per head centre → SAM2 masks → shape check; overlay + table |
| `remove_heads(image, ids)`, `set_kind(image, ids, kind)` | corrections |
| `save(image, note)` | `agent_labels/<image>.ctip-labels.json` (polygons, provenance, `pending_review`) + YOLO `.txt` |
| `push_to_label_studio(image, project)` | uploads the image as a task with the labels as a prediction |

Kinds: `stalked`, `sessile`, `bulbous`, `non-glandular` (YOLO class ids 0–3, the CTIP Label Studio template).

## Rules that keep it honest

- **Human gate.** Saved files say `pending_review`; Label Studio stores them as predictions. CTIP's dataset export
  uses human-confirmed annotations only (unless `use_predictions` is set on purpose).
- **Only visible things.** The guideline tells the agent to skip heads it cannot place within a few pixels and
  never to judge maturity or cannabinoid content.
- **Sandboxed.** The tools read images only below `CTIP_LABEL_ROOT`; paths that leave it are refused.
- **Provenance.** Every record names the annotator (`claude-code`, plus `CLAUDE_MODEL` if set), the tool version
  and the time, so agent labels can be audited or excluded later.

## What to expect

In the October 2026 test run on a Wikimedia Commons photo, click labelling with SAM2 placed clean masks on 24 of
24 heads after the shape check trimmed 9 stalks; heads under ~20 px need zooming. Agent labels are pre-labels:
plan for a reviewer to fix misses and false positives, especially on blur and on small sessile heads.
