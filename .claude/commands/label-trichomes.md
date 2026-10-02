---
description: Label glandular trichome heads with the CTIP MCP tools (SAM2 masks, human review afterwards)
argument-hint: "[image or folder under data/images, e.g. commons_trichomes/commons_16.jpg]"
allowed-tools: mcp__ctip-label__list_images, mcp__ctip-label__open_image, mcp__ctip-label__view, mcp__ctip-label__add_heads, mcp__ctip-label__remove_heads, mcp__ctip-label__set_kind, mcp__ctip-label__save
---

Label the glandular trichome heads in: $ARGUMENTS
(If that is a folder or empty, call `list_images` and work through the images that have no saved labels yet.)

Work like a careful human annotator:

1. `open_image`. Read the grid: its numbers are ORIGINAL pixel coordinates.
2. Zoom with `view` (region x0, y0, x1, y1) into every area with heads - small or dense heads need a zoom where a
   head is at least ~30 px on screen. Place points from the zoomed view's grid.
3. `add_heads` with one point at the CENTRE of each head - never on the stalk. Choose `kinds`:
   - `stalked`: round head on a visible stalk (capitate-stalked)
   - `sessile`: head sitting directly on the surface, no stalk
   - `bulbous`: very small head (a few cells)
   - `non-glandular`: hair without a head (only label these if asked)
4. Check every returned overlay. `remove_heads` for masks on background, on the stalk, or covering two heads, then
   add a better point. `trimmed` = a stalk was cut off (fine). `rejected` = not round, not kept.
5. Skip heads you cannot place within a few pixels (blur, occlusion). Do not judge maturity or cannabinoid content.
6. `save` with a short note: what was hard, which areas you skipped.

Finish with a short report per image: heads labelled per kind, heads you skipped and why.
Your labels are saved as *pending review* - a person approves them (Label Studio or the CTIP review queue)
before they can become training data. Push to Label Studio only when the user asks (`push_to_label_studio`).
