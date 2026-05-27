import type { WikiPage } from '../types';

const en = `
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

\`\`\`bash
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
\`\`\`

**Stage micrometer** (calibration slide with known scale, e.g. 1mm in 100 divisions):
- Photograph it at the same settings as your samples
- The calibration tool detects the scale bar automatically or accepts manual input

---

## File organization

Recommended directory structure:

\`\`\`
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
\`\`\`

### Session metadata (metadata.json)

\`\`\`json
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
\`\`\`

---

## Image upload to Label Studio

### Bulk upload via API

\`\`\`python
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
\`\`\`

### Via CTIP CLI

\`\`\`bash
trichome upload --session data/raw/session_2025-01-15_strain-A/ --project 1
\`\`\`

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

\`\`\`bash
trichome vlm-label --session data/raw/session_2025-01-15_strain-A/ --model moondream-2b
\`\`\`

This creates **pending_review** tasks in Label Studio.
**A human must review and approve every VLM annotation** before it enters training data.
This is a hard architectural constraint — VLM output is never written directly to training data.
`;

const de = `
## Mikroskop-Setup

CTIP ist für **Hellfeld- und Fluoreszenzmikroskopie** von Cannabis-Trichomen ausgelegt.
Empfohlene Vergrößerung: **40× – 100×** (10× Objektiv + 4× oder 10× Okular).

### Beleuchtung
- **Hellfeld**: Standard-Durchlicht. Trichome erscheinen als runde Köpfe auf Stielen.
- **Auflicht**: Besser für Oberflächenstruktur.
- Konsistente Beleuchtung über alle Proben ist entscheidend für die farbbasierte Reifeklassifizierung.
  → Immer gleiche Lichtintensität pro Session verwenden.

### Bildanforderungen

| Parameter | Empfohlen |
|-----------|-----------|
| Auflösung | ≥ 1920×1080 px (4K bevorzugt) |
| Format | TIFF (verlustfrei) oder JPEG Qualität 95+ |
| Farbtiefe | 8-Bit pro Kanal (24-Bit RGB) |
| Fokus | Scharfe Trichomköpfe erforderlich |

> **Wichtig**: Mikroskope nicht innerhalb eines Trainingsdatensatzes mischen ohne separate Kalibrierdateien. Jedes Mikroskop hat einen anderen µm/px-Faktor.

---

## Kalibrierung

\`\`\`bash
trichome calibrate --image stage_micrometer.jpg --known-length 100 --known-unit um
\`\`\`

Erstellt \`data/calibrations/{mikroskop_id}.json\` mit dem µm/px-Faktor.

**Stagemarkierung** (Kalibrierungsobjektträger mit bekannter Skala, z.B. 1mm in 100 Teile):
- Bei gleichen Einstellungen wie die Proben fotografieren
- Kalibrier-Tool erkennt die Skala automatisch

---

## Dateiorganisation

\`\`\`
data/
├── raw/
│   ├── session_2025-01-15_strain-A/
│   │   ├── IMG_0001.tif
│   │   └── metadata.json
│   └── session_2025-01-20_strain-B/
├── calibrations/
├── models/
├── outputs/
└── exports/
\`\`\`

### Session-Metadaten (metadata.json)

\`\`\`json
{
  "session_id": "session_2025-01-15_strain-A",
  "date": "2025-01-15",
  "strain": "Strain-A",
  "microscope": "DeltaOptical-40x",
  "objective": "40x",
  "lighting": "hellfeld",
  "notes": "Späte Blüte, Tag 65"
}
\`\`\`

---

## Upload zu Label Studio

\`\`\`bash
trichome upload --session data/raw/session_2025-01-15_strain-A/ --project 1
\`\`\`

---

## Qualitätsregeln für Daten

| Regel | Warum |
|-------|-------|
| ≥ 20 Bilder je Sorte je Reifestadium | Minimum für robustes Training |
| Kein Bewegungsunschärfe | YOLO degradiert stark bei unscharfen Bildern |
| Konsistenter Hintergrund | Reduziert False Positives |
| Alle Trichome im Bild labeln | Ungelabelte werden zu False Negatives |
| Aus TIFF annotieren | Keine JPEG-Kompressionsartefakte bei Annotation |

---

## VLM-unterstütztes Pre-Labeling

\`\`\`bash
trichome vlm-label --session data/raw/session_2025-01-15_strain-A/ --model moondream-2b
\`\`\`

Erstellt **pending_review**-Tasks in Label Studio. **Jede VLM-Annotation muss von einem Menschen geprüft und genehmigt werden** — VLM-Output wird niemals direkt in Trainingsdaten geschrieben.
`;

const es = `
## Configuración del microscopio

Ampliación recomendada: **40× – 100×**. Iluminación consistente es crítica para la clasificación de madurez basada en color.

### Requisitos de imagen

| Parámetro | Recomendado |
|-----------|-------------|
| Resolución | ≥ 1920×1080 px |
| Formato | TIFF (sin pérdida) o JPEG calidad 95+ |
| Enfoque | Cabezas de tricomas nítidas requeridas |

---

## Calibración

\`\`\`bash
trichome calibrate --image stage_micrometer.jpg --known-length 100 --known-unit um
\`\`\`

Crea \`data/calibrations/{microscopio}.json\` con el factor µm/px.

---

## Organización de archivos

\`\`\`
data/
├── raw/session_FECHA_cepa/
│   ├── IMG_0001.tif
│   └── metadata.json
├── calibrations/
├── models/
└── outputs/
\`\`\`

---

## Reglas de calidad de datos

- ≥ 20 imágenes por cepa por estadio de madurez
- Sin desenfoque de movimiento
- Etiquetar **todos** los tricomas en cada imagen
- Anotar desde TIFF, no desde JPEG

---

## Pre-etiquetado VLM

\`\`\`bash
trichome vlm-label --session data/raw/... --model moondream-2b
\`\`\`

Crea tareas **pending_review** en Label Studio. **Un humano debe revisar y aprobar cada anotación VLM** — el output VLM nunca se escribe directamente en los datos de entrenamiento.
`;

const page: WikiPage = {
  slug: 'data-collection',
  title: { en: 'Data Collection', de: 'Daten sammeln', es: 'Recopilación de datos' },
  description: {
    en: 'Microscope setup, calibration, image requirements, file organization, and VLM pre-labeling.',
    de: 'Mikroskop-Setup, Kalibrierung, Bildanforderungen, Dateiorganisation und VLM-Pre-Labeling.',
    es: 'Configuración del microscopio, calibración, requisitos de imagen y organización de archivos.',
  },
  content: { en, de, es },
  section: 'workflow',
  icon: '📷',
};

export default page;
