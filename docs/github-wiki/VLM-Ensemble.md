## VLM Ensemble

Instead of relying on a single Vision-Language Model (VLM) for auto-labeling, CTIP can run **multiple VLM backends simultaneously** and aggregate their predictions via majority vote. Ensemble mode increases label reliability and surfaces disagreement as an explicit quality signal.

> **Invariant**: VLM outputs — whether from a single model or an ensemble — are **never** written directly to the training dataset. Every label produced by an ensemble run is placed in the `pending_review` queue and must be approved by a human annotator before it becomes training data.

---

## Supported backends

| Backend | Size | VRAM (4-bit quant) | Notes |
|---|---|---|---|
| Moondream-2B | 2B | ~1.8 GB | Fastest; good for binary decisions |
| Florence-2 | 0.23B–0.77B | ~0.8 GB | Strong grounding and counting |
| Qwen2-VL | 2B–7B | ~2.4 GB (2B) | Best general VQA accuracy |

All models run 4-bit quantized to stay within the 8 GB VRAM budget.

---

## Agreement scoring

After all models return a label, the ensemble computes an **agreement score**:

```
agreement = count(majority_label) / total_models
```

| Score | Level | Action |
|---|---|---|
| ≥ 0.8 | High | Label sent to `pending_review` with high confidence |
| ≥ 0.6 | Medium | Label sent to `pending_review`; reviewer notified |
| < 0.6 | Low | Label flagged; reviewer must adjudicate before approval |

---

## Prompt system

CTIP ships with preset prompt templates optimised for trichome labeling tasks. You can also supply custom `system` and `user` prompts per request.

### Preset IDs

| Preset | Task |
|---|---|
| `morphology_classify` | Classify trichome type (stalked / sessile / bulbous / non-glandular) |
| `maturity_classify` | Classify maturity stage (clear / cloudy / amber) |
| `count_estimate` | Estimate trichome count in a region |
| `quality_check` | Assess image quality (focus, lighting, artifacts) |

---

## API reference

### Run ensemble labeling

```bash
POST /api/v1/vlm/ensemble/label
Content-Type: application/json

{
  "image_id": "img_0047",
  "models": ["moondream", "florence2", "qwen2vl"],
  "preset": "morphology_classify",
  "system_prompt_override": null,
  "user_prompt_override": null
}
```

Response:
```json
{
  "image_id": "img_0047",
  "majority_label": "CAPITATE_STALKED",
  "agreement_score": 0.83,
  "agreement_level": "high",
  "per_model": {
    "moondream":  { "label": "CAPITATE_STALKED", "confidence": 0.91 },
    "florence2":  { "label": "CAPITATE_STALKED", "confidence": 0.87 },
    "qwen2vl":    { "label": "CAPITATE_SESSILE",  "confidence": 0.61 }
  },
  "review_id": "rev_9f3c"
}
```

The `review_id` links to the pending-review queue entry.

### List prompt presets

```bash
GET /api/v1/vlm/prompts
```

```json
{
  "presets": [
    {
      "id": "morphology_classify",
      "description": "Classify trichome morphological type",
      "system_prompt": "You are a cannabis microscopy specialist ...",
      "user_prompt": "Classify the trichome in this image ..."
    }
  ]
}
```

### Create / update a custom prompt

```bash
POST /api/v1/vlm/prompts
Content-Type: application/json

{
  "id": "my_custom_preset",
  "description": "Custom morphology prompt for lab X",
  "system_prompt": "...",
  "user_prompt": "..."
}
```

### Validate a prompt

```bash
POST /api/v1/vlm/prompts/validate
Content-Type: application/json

{
  "system_prompt": "...",
  "user_prompt": "...",
  "test_image_id": "img_0001"
}
```

Returns a dry-run response showing what each enabled backend would return, without writing to the review queue.

---

## Frontend: Annotation page → VLM Config panel

- **Ensemble mode toggle** — enable to select multiple backends; disable to use a single model.
- **Model checkboxes** — select which backends to include; VRAM estimate shown per combination.
- **Preset dropdown** — choose a labeling task; preview of system/user prompts displayed.
- **Custom prompt editor** — override system and/or user prompt for the current run.
- **Agreement threshold slider** — set the minimum score for automatic forwarding to the review queue.
- **Results panel** — per-model labels, individual confidence scores, majority label badge, agreement level indicator.

---

## Agreement visualisation

The annotation page shows a colour-coded badge:

- **Green** (high ≥ 0.8) — models strongly agree.
- **Yellow** (medium ≥ 0.6) — partial agreement; human review recommended.
- **Red** (low < 0.6) — models disagree; adjudication required before the label is usable.
