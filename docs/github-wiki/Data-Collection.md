## Microscope setup

CTIP is designed for **brightfield and fluorescence microscopy** of cannabis trichomes.
Recommended magnification: **40× – 100×** (10× objective + 4× or 10× eyepiece).

### Supported microscope types
- Digital USB microscope (Dino-Lite, Celestron)
- Compound microscope with digital eyepiece
- Stereo microscope with camera attachment

### Lighting
- **Brightfield**: Standard transmitted light. Trichomes appear as round heads on stalks.
- **Incident (epi) light**: Better for surface structure visibility.
- Consistent lighting across all samples is critical for color-based maturity classification.
  → Always use the same light intensity setting per session.

### Image requirements

| Parameter | Recommended |
|-----------|-------------|
| Resolution | ≥ 1920×1080 px (4K preferred) |
| Format | TIFF (lossless) or JPEG quality 95+ |
| Bit depth | 8-bit per channel (24-bit RGB) |
| Color space | sRGB |
| Focus | Sharp trichome heads required |

> **Critical**: Do not mix microscopes within one training dataset without separate calibration files.
> Each microscope has a different µm/px ratio.

---

## Calibration

Before measuring trichome size in µm, calibrate the scale:

```bash
# CLI
trichome calibrate --image stage_micrometer.jpg --known-length 100 --known-unit um

# This creates/updates:
# data/calibrations/{microscope_id}.json
# {
#   "um_per_px": 0.312,
#   "microscope": "DeltaOptical-40x",
#   "date": "2025-01-15",
#   "target": "stage_micrometer"
# }
```

**Stage micrometer** (calibration slide with known scale, e.g. 1mm in 100 divisions):
- Photograph it at the same settings as your samples
- The calibration tool detects the scale bar automatically or accepts manual input

---

## File organization

Recommended directory structure:

```
data/
├── raw/
│   ├── session_2025-01-15_strain-A/
│   │   ├── IMG_0001.tif
│   │   ├── IMG_0002.tif
│   │   └── metadata.json
│   └── session_2025-01-20_strain-B/
│       └── ...
├── calibrations/
│   ├── DeltaOptical-40x.json
│   └── DinoliteDU3131-50x.json
├── models/          # ML weights
├── outputs/         # Detection results
└── exports/         # CSV, PDF reports
```

### Session metadata (metadata.json)

```json
{
  "session_id": "session_2025-01-15_strain-A",
  "date": "2025-01-15",
  "strain": "Strain-A",
  "microscope": "DeltaOptical-40x",
  "objective": "40x",
  "lighting": "brightfield",
  "notes": "Late flowering, day 65",
  "images": ["IMG_0001.tif", "IMG_0002.tif"]
}
```

---

## Image upload to Label Studio

### Bulk upload via API

```python
import httpx
from pathlib import Path

LS_URL = "http://localhost:3005"
API_KEY = "your-api-key"
PROJECT_ID = 1

files = list(Path("data/raw/session_2025-01-15_strain-A").glob("*.tif"))

for img in files:
    with open(img, "rb") as f:
        r = httpx.post(
            f"{LS_URL}/api/projects/{PROJECT_ID}/import",
            headers={"Authorization": f"Token {API_KEY}"},
            files={"file": (img.name, f, "image/tiff")},
        )
        print(img.name, r.status_code)
```

### Via CTIP CLI

```bash
trichome upload --session data/raw/session_2025-01-15_strain-A/ --project 1
```

---

## Data quality rules

| Rule | Why |
|------|-----|
| ≥ 20 images per strain per maturity stage | Minimum for robust training |
| No motion blur | YOLO degrades significantly on blurry images |
| Consistent background | Reduces false positives |
| No overlapping trichomes if possible | Improves segmentation accuracy |
| Label every trichome in frame | Unlabeled trichomes become false negatives |
| No JPG compression artifacts at annotation time | Annotate from TIFF, export JPG only for training |

---

## VLM-assisted pre-labeling

CTIP can use Vision-Language Models to pre-label images:

```bash
trichome vlm-label --session data/raw/session_2025-01-15_strain-A/ --model moondream-2b
```

This creates **pending_review** tasks in Label Studio.
**A human must review and approve every VLM annotation** before it enters training data.
This is a hard architectural constraint — VLM output is never written directly to training data.
