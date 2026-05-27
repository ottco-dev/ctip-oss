import type { WikiPage } from '../types';

const en = `
## Inference pipeline

\`\`\`
Input image
    │
    ▼
Tiling (if image > 1280px)
    │  Tile size: 1280px, overlap: 20%
    │
    ▼
YOLO11s detection (per tile)
    │  → confidence calibration (Platt scaling / temperature)
    │
    ▼
NMS merge (tiles → full image coordinates)
    │
    ▼
SAM2-tiny segmentation (prompted by YOLO boxes)
    │  → pixel-accurate masks
    │  → mask refinement (fill holes, smooth edges)
    │
    ▼
Morphology classification (stalked/sessile/bulbous)
    │
    ▼
Maturity classification (clear/cloudy/amber)
    │  HSV + LAB + LBP/GLCM/Gabor features
    │
    ▼
Measurement (px → µm via calibration scale)
    │
    ▼
Report (JSON / CSV / PDF)
\`\`\`

---

## CLI usage

### Analyze a single image

\`\`\`bash
trichome detect \\
  --input data/raw/session_2025-01-15/IMG_0001.tif \\
  --model data/models/yolo11s.pt \\
  --tiled \\
  --tile-size 1280 \\
  --overlap 0.2 \\
  --conf 0.25 \\
  --iou 0.45 \\
  --output data/outputs/IMG_0001/
\`\`\`

Output files:
\`\`\`
data/outputs/IMG_0001/
├── detections.json       # all bounding boxes + confidence + class
├── masks/                # SAM2 segmentation masks (.png per instance)
├── annotated.jpg         # visualization with overlays
├── report.pdf            # summary report
└── measurements.csv      # per-trichome measurements
\`\`\`

### Batch analysis

\`\`\`bash
trichome detect \\
  --input data/raw/session_2025-01-15/ \\
  --pattern "*.tif" \\
  --model data/models/yolo11s.pt \\
  --tiled \\
  --batch-size 4 \\
  --output data/outputs/session_2025-01-15/
\`\`\`

### Segmentation only (from existing detections)

\`\`\`bash
trichome segment \\
  --input data/raw/session_2025-01-15/IMG_0001.tif \\
  --detections data/outputs/IMG_0001/detections.json \\
  --model sam2-tiny
\`\`\`

---

## API usage

### Start inference job

\`\`\`bash
curl -X POST http://localhost:8000/api/v1/detection/analyze \\
  -H "Content-Type: multipart/form-data" \\
  -F "image=@data/raw/IMG_0001.tif" \\
  -F "model=yolo11s" \\
  -F "tiled=true" \\
  -F "tile_size=1280"
\`\`\`

Response:
\`\`\`json
{
  "job_id": "abc123",
  "status": "queued",
  "estimated_seconds": 8
}
\`\`\`

### Poll job status

\`\`\`bash
curl http://localhost:8000/api/v1/detection/jobs/abc123
\`\`\`

\`\`\`json
{
  "job_id": "abc123",
  "status": "done",
  "detections": 47,
  "result_url": "/api/v1/detection/results/abc123"
}
\`\`\`

---

## Tiled inference explained

For images larger than 1280px (common with digital microscopes at 4K):

\`\`\`
Original image (e.g. 3840 × 2160 px)
    │
    ▼
Split into tiles: 1280 × 1280 px, 20% overlap
    │  e.g. 4 × 3 = 12 tiles
    │
    ▼
YOLO inference on each tile (GPU-batched)
    │
    ▼
Map tile coordinates back to full image
    │
    ▼
NMS across all tiles (remove duplicates at tile borders)
    │
    ▼
Merged detection result
\`\`\`

VRAM usage with tiled inference:
- **1 tile at a time**: ~1.8 GB VRAM (safe for 8 GB)
- **4 tiles batched**: ~5.0 GB VRAM (fits RTX 4060)
- **8 tiles batched**: ~9.5 GB VRAM (OOM on 8 GB)

---

## Confidence calibration

Raw YOLO confidence scores are often poorly calibrated (overconfident).
CTIP applies Platt scaling or temperature scaling post-training:

\`\`\`bash
trichome calibrate-confidence \\
  --model data/models/yolo11s.pt \\
  --val-data data/datasets/v1/val/ \\
  --method platt
\`\`\`

This creates \`data/models/yolo11s_calibrated.json\` with calibration parameters.
Calibrated confidence is used in all inference and shown in reports.

---

## Output formats

### JSON (detections.json)

\`\`\`json
{
  "image": "IMG_0001.tif",
  "model": "yolo11s_custom.pt",
  "inference_time_ms": 234,
  "detections": [
    {
      "id": 1,
      "class": "stalked",
      "confidence": 0.87,
      "calibrated_confidence": 0.83,
      "bbox": [124, 89, 198, 163],
      "mask_file": "masks/001.png",
      "maturity": "cloudy",
      "maturity_confidence": 0.79,
      "size_px": 74,
      "size_um": 23.1,
      "calibration": "DeltaOptical-40x.json"
    }
  ],
  "summary": {
    "total": 47,
    "stalked": 31,
    "sessile": 12,
    "bulbous": 3,
    "non_glandular": 1,
    "maturity": {
      "clear": 8,
      "cloudy": 35,
      "amber": 4
    }
  }
}
\`\`\`

### CSV (measurements.csv)

Tabular format with one row per detected trichome — suitable for R/Python analysis.
`;

const de = `
## Inferenz-Pipeline

\`\`\`
Eingabebild
    │
    ▼
Tiling (wenn Bild > 1280px)
    │
    ▼
YOLO11s-Erkennung (je Kachel) + Konfidenz-Kalibrierung
    │
    ▼
NMS-Zusammenführung (Kacheln → vollständige Bildkoordinaten)
    │
    ▼
SAM2-tiny Segmentierung (getriggert durch YOLO-Boxen)
    │
    ▼
Morphologie-Klassifizierung + Reifegrad-Klassifizierung
    │
    ▼
Messung (px → µm via Kalibrierung)
    │
    ▼
Report (JSON / CSV / PDF)
\`\`\`

---

## CLI-Verwendung

### Einzelbild analysieren

\`\`\`bash
trichome detect \\
  --input data/raw/session_2025-01-15/IMG_0001.tif \\
  --model data/models/yolo11s.pt \\
  --tiled \\
  --tile-size 1280 \\
  --output data/outputs/IMG_0001/
\`\`\`

Ausgabedateien: \`detections.json\`, \`masks/\`, \`annotated.jpg\`, \`report.pdf\`, \`measurements.csv\`

### Batch-Analyse

\`\`\`bash
trichome detect \\
  --input data/raw/session_2025-01-15/ \\
  --pattern "*.tif" \\
  --tiled --batch-size 4 \\
  --output data/outputs/session_2025-01-15/
\`\`\`

---

## Tiled Inference erklärt

Für Bilder > 1280px (typisch bei 4K-Digitalmikroskopen):

- Bild wird in 1280×1280px Kacheln mit 20% Überlappung aufgeteilt
- YOLO läuft auf jeder Kachel separat
- Koordinaten werden auf das Originalbild zurückgemappt
- NMS entfernt Duplikate an Kachelgrenzen

VRAM-Verbrauch:
- **1 Kachel gleichzeitig**: ~1,8 GB VRAM (sicher für 8 GB)
- **4 Kacheln gebatcht**: ~5,0 GB VRAM (passt auf RTX 4060)

---

## Ausgabeformate

**JSON** (\`detections.json\`): Vollständige Erkennungsdaten inkl. Klasse, Konfidenz, Maske, Reifegrad, Größe in µm.

**CSV** (\`measurements.csv\`): Tabellarisches Format, eine Zeile je Trichom — geeignet für R/Python-Analyse.

**PDF** (\`report.pdf\`): Zusammenfassungsbericht mit Visualisierungen und Statistiken.
`;

const es = `
## Pipeline de inferencia

\`\`\`
Imagen de entrada
    │
    ▼
Mosaico (si imagen > 1280px, solapamiento 20%)
    │
    ▼
Detección YOLO11s + calibración de confianza
    │
    ▼
Fusión NMS (mosaicos → coordenadas completas)
    │
    ▼
Segmentación SAM2-tiny (impulsada por cajas YOLO)
    │
    ▼
Clasificación morfológica + madurez
    │
    ▼
Medición (px → µm vía calibración)
    │
    ▼
Reporte (JSON / CSV / PDF)
\`\`\`

## Uso CLI

\`\`\`bash
# Analizar imagen individual
trichome detect \\
  --input data/raw/IMG_0001.tif \\
  --model data/models/yolo11s.pt \\
  --tiled --tile-size 1280 \\
  --output data/outputs/IMG_0001/

# Análisis por lotes
trichome detect \\
  --input data/raw/sesion/ \\
  --pattern "*.tif" \\
  --tiled --batch-size 4 \\
  --output data/outputs/sesion/
\`\`\`

## Uso de VRAM (inferencia en mosaico)

- **1 mosaico a la vez**: ~1.8 GB VRAM (seguro para 8 GB)
- **4 mosaicos en lote**: ~5.0 GB VRAM (cabe en RTX 4060)
- **8 mosaicos en lote**: ~9.5 GB VRAM (OOM en 8 GB)
`;

const page: WikiPage = {
  slug: 'inference',
  title: { en: 'Inference & Analysis', de: 'Inferenz & Analyse', es: 'Inferencia & Análisis' },
  description: {
    en: 'Full inference pipeline, CLI usage, tiled inference, output formats, confidence calibration.',
    de: 'Vollständige Inferenz-Pipeline, CLI-Verwendung, Tiled Inference, Ausgabeformate.',
    es: 'Pipeline completo de inferencia, uso CLI, inferencia en mosaico, formatos de salida.',
  },
  content: { en, de, es },
  section: 'workflow',
  icon: '🔍',
};

export default page;
