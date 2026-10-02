# CTIP Handbuch (Deutsch)

> [English](en.md) · [Deutsch](de.md) · [Español](es.md) · [Technology stack](tech-stack.md) · [Back to README](../../README.md)

## Inhaltsverzeichnis

1. [Was ist das?](#de-1)
2. [Hardware-Anforderungen](#de-2)
3. [Installation](#de-3)
4. [Erste Schritte](#de-4)
5. [Datenerfassung — Bildtipps](#de-5)
6. [Labeling-Workflow](#de-6)
7. [Training-Workflow](#de-7)
8. [Verifizierung und Benchmarking](#de-8)
9. [Verbesserungs-Loop](#de-9)
10. [Docker-Deployment](#de-10)
11. [Alle URLs und Seiten](#de-11)
12. [API-Referenz](#de-12)
13. [CLI-Referenz](#de-13)
14. [Konfiguration](#de-14)
15. [Architektur](#de-15)
16. [Wissenschaftliche Methodik](#de-16)
17. [Tests](#de-17)

---

## DE 1. Was ist das? {#de-1}

CTIP ist eine **vollständige, produktionsreife Forschungsplattform** zur automatisierten Trichom-Analyse von *Cannabis sativa L.* unter digitaler Mikroskopie. Kein Demo, kein Spielzeug — ein vollständiges, laufendes System für echte wissenschaftliche Arbeit.

### Was es kann

| Fähigkeit | Methode | Zielwert |
|---|---|---|
| Trichom-Erkennung | YOLO v11s + RTMDet Ensemble | mAP50 > 0.88 |
| Instanz-Segmentierung | SAM2-tiny + Maskenverfeinerung | IoU > 0.82 |
| Reifegradklassifikation | HSV + LAB + Textur (LBP/GLCM/Gabor) | F1 > 0.85 |
| Morphologie-Typisierung | Geometrisch + CNN (gestielt/sitzend/kugelförmig) | Genauigkeit > 0.90 |
| Größenmessung | Kalibrierte px nach µm Umrechnung | ±5% Fehler |
| Fokusbeurteilung | Laplacian + Tenengrad + FFT | — |
| Videoanalyse | Frame-Qualitätsranking + temporale Deduplizierung | — |
| VLM-Vorlabeling | Moondream-2B / Florence-2 / Qwen2-VL (4-bit) | Mensch-in-Loop |
| Aktives Lernen | Unsicherheits- + Dissens-Sampling | — |
| TensorRT-Inferenz | FP16 Engine, async v3 API | RTX 4060 optimiert |

### Was es NICHT ist

- Keine THC/Cannabinoid-Konzentrationsvorhersagen (nur optische Reife)
- Keine Pseudowissenschaft
- VLM-Ausgaben gehen niemals direkt in Trainingsdaten (HITL-Gate ist Pflicht)

---

## DE 2. Hardware-Anforderungen {#de-2}

| Komponente | Minimum | Empfohlen |
|---|---|---|
| GPU | NVIDIA GTX 1080 (8 GB VRAM) | RTX 4060 / 3080 (8+ GB) |
| CPU | 6-Kern modern | i5-13400F oder besser |
| RAM | 16 GB | 32 GB |
| Speicher | 50 GB SSD | 500 GB NVMe |
| CUDA | 11.8+ | 12.6 |

### VRAM-Budget (RTX 4060, 8 GB)

| Komponente | VRAM |
|---|---|
| YOLO v11s Inferenz | ~0,9 GB |
| SAM2-tiny | ~1,8 GB |
| Florence-2 (4-bit) | ~2,1 GB |
| Moondream-2B (4-bit) | ~1,4 GB |
| Qwen2-VL-7B (4-bit) | ~4,8 GB |
| YOLO v11s Training (bs=8) | ~5,5 GB |

> Immer nur **ein GPU-Task gleichzeitig** — asyncio.Semaphore(1). Bewusst so designed für 8-GB-Karten.

---

## DE 3. Installation {#de-3}

### 3.1 Voraussetzungen

```bash
sudo apt update && sudo apt install -y \
    git curl wget build-essential \
    python3.12 python3.12-venv python3.12-dev \
    ffmpeg libgl1 libglib2.0-0 libsm6 libxext6

# uv installieren (schneller Python-Paketmanager)
curl -LsSf https://astral.sh/uv/install.sh | sh && source ~/.bashrc

# Node.js 20 (für Frontend)
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt install -y nodejs

nvcc --version && nvidia-smi
```

### 3.2 Klonen und Installieren

```bash
git clone https://github.com/deinuser/trichome-analysis.git
cd trichome-analysis

python3.12 -m venv .venv && source .venv/bin/activate

uv pip install -e ".[dev]"     # Kern + Dev
uv pip install -e ".[vlm]"     # + VLM-Modelle
uv pip install -e ".[sam]"     # + SAM2-Segmentierung
uv pip install -e ".[all]"     # alles
```

### 3.3 TensorRT (optional, für Produktions-Inferenz)

```bash
sudo apt install -y python3-libnvinfer python3-libnvinfer-dev tensorrt tensorrt-dev

export PATH=/usr/local/cuda-12.6/bin:$PATH && pip install pycuda

SITE=$(python -c "import site; print(site.getsitepackages()[0])")
printf "/usr/lib/python3/dist-packages\n/usr/lib/python3.12/dist-packages\n" > "$SITE/system_trt.pth"
echo 'export PATH=/usr/local/cuda-12.6/bin:$PATH' >> .venv/bin/activate

python -c "import tensorrt; print(tensorrt.__version__)"
```

### 3.4 Frontend

```bash
cd frontend && npm install && cd ..
```

### 3.5 Umgebungskonfiguration

```bash
cp .env.example .env
```

> **Tipp:** Nutze den integrierten **Einrichtungsassistenten** statt die `.env` manuell zu bearbeiten — er führt dich Schritt für Schritt durch alle Einstellungen (→ §4.1).

Für manuelle Konfiguration die wichtigsten Variablen:

```env
DATA_ROOT=/mnt/data/trichome          # oder ./data für lokale Entwicklung
MODELS_ROOT=/mnt/models/trichome
CUDA_VISIBLE_DEVICES=0
VRAM_LIMIT_GB=8.0
MLFLOW_TRACKING_URI=http://localhost:3004
EXPERIMENT_TRACKER=mlflow
LABEL_STUDIO_URL=http://localhost:3005
LABEL_STUDIO_API_KEY=dein_schluessel
```

---

## DE 4. Erste Schritte {#de-4}

### 4.1 Dev-Modus starten & Ersteinrichtung

```bash
# Terminal 1 — Backend-API
source .venv/bin/activate
uvicorn backend.main:app --reload --port 8000

# Terminal 2 — Frontend
cd frontend && npm run dev
```

**http://localhost:3000** öffnen — der **Einrichtungsassistent startet automatisch** beim ersten Start (solange keine `.env` konfiguriert ist).

Der Assistent führt durch 7 Schritte:

| Schritt | Konfiguriert |
|---|---|
| 🌐 Netzwerk | Öffentliche Domain vs. nur localhost, nginx-Port |
| ⚙️ Hardware | CUDA-Gerät, VRAM-Budget |
| 💾 Speicher | Datenwurzel, Modellverzeichnis, Ausgabepfad |
| 🔌 Dienste | Label Studio API-Key, MLflow URI, W&B (optional) |
| 🔒 Sicherheit | Secret Key (Auto-Generator), API-Token |
| ✅ Überprüfung | Zusammenfassung aller Einstellungen |
| 🎉 Fertig | Schreibt `.env`, zeigt Docker-Restart-Befehl |

Nach Abschluss wird die `.env` automatisch geschrieben — kein manuelles Bearbeiten nötig.
Erneut starten jederzeit über die Seitenleiste: **Ersteinrichtung**.

- API-Docs: http://localhost:8000/docs

### 4.2 System prüfen

```bash
pytest tests/ -v --tb=short
# Erwartet: 1694 passed, 4 skipped

python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
curl http://localhost:8000/api/v1/system/health | python -m json.tool
```

### 4.3 Erste Erkennung ausführen

```bash
# Via CLI
trichome detect --input /pfad/zum/bild.jpg --tiled --tile-size 1280

# Via API
curl -X POST http://localhost:8000/api/v1/detection/infer \
  -F "file=@/pfad/zum/bild.jpg" -F "confidence_threshold=0.25" | python -m json.tool

# Via Frontend: http://localhost:3000/inference — Bild hineinziehen
```

---

## DE 5. Datenerfassung — Bildtipps {#de-5}

Gute Daten sind der wichtigste Faktor überhaupt. Was bei Trichom-Mikroskopie wirklich zählt:

### 5.1 Equipment

| Setup | Hinweise |
|---|---|
| Digitalmikroskop | 40x–200x Vergrößerung optimal. USB-Mikroskope (Andonstar, Celestron, Jiusion) reichen für den Einstieg. |
| Smartphone + Clip-Linse | Für größere Trichome geeignet, schlecht für kleine kugelförmige. |
| Stereomikroskop | Beste optische Qualität, schwieriger konsistent zu digitalisieren. |

### 5.2 Aufnahme-Protokoll (Kritisch)

```
MACHEN:
  - Konsistente Vergrößerung für jede Session (z.B. immer 100x)
  - RAW oder maximale JPEG-Qualität aufnehmen
  - Kalibrierungsbild mit bekannter Skala aufnehmen (Objektmikrometer, z.B. 1 mm)
    Ermöglicht px nach µm Kalibration
  - Mindestens 1920x1080, idealerweise 4K
  - Lichtquelle immer in derselben Position
  - Dateiname mit Metadaten: mikroskop01_100x_20260101_probe42_001.jpg
  - Alle Trichom-Typen in einer Session aufnehmen
  - Leere Hintergrundbereiche aufnehmen (keine Trichome) als Negativbeispiele

NICHT MACHEN:
  - Vergrößerungen mischen ohne Notiz (ruiniert Kalibration)
  - Auto-Belichtung verwenden (inkonsistente Helligkeit)
  - Nur perfekte Trichome fotografieren (teilweise sichtbare + überlappende einschließen)
  - Komprimierte Social-Media-Bilder verwenden
```

### 5.3 Reifestadien-Abdeckung

| Stadium | Optik | Ziel-% im Datensatz |
|---|---|---|
| Klar | Glasig, vollständig transparent | ~25% |
| Trüb | Weiß/milchig, opak | ~35% |
| Bernstein | Goldgelb-orange, degradiert | ~25% |
| Gemischt | Übergangspräparate | ~15% |

### 5.4 Datenorganisation

```
data/
├── raw/                    # Originale, unveränderte Bilder
│   ├── session_20260101/
│   └── session_20260115/
├── calibration/            # Objektmikrometer-Bilder pro Mikroskop+Vergrößerung
├── annotated/              # Nach dem Labeling (Label Studio exportiert hierher)
│   ├── images/
│   └── labels/             # YOLO-Format .txt-Dateien
└── splits/                 # train / val / test — niemals Sessions mischen!
    ├── train/
    ├── val/
    └── test/
```

> **Niemals** Bilder aus derselben Mikroskopie-Session in train UND val/test packen — das ist Data Leakage.
> Immer nach **Session** splitten, nicht nach Bild.

### 5.5 Fokusqualitäts-Filter

```bash
trichome video score image.jpg      # focus, exposure and noise score (also shown per image in the UI)
               --output data/gefiltert/ \
               --min-sharpness 80.0 --copy-passing
```

### 5.6 Mindest-Datenmenge

| Phase | Bilder | Annotationen (Boxen) |
|---|---|---|
| Erstes funktionierendes Modell | 150–300 | 2.000–5.000 |
| Gute Generalisierung | 500–1.000 | 10.000–25.000 |
| Produktionsreif | 2.000+ | 50.000+ |

Klein anfangen, schnell trainieren, Schwachstellen identifizieren, gezielt sammeln. Das schlägt 1.000 zufällige Bilder jedes Mal.

---

## DE 6. Labeling-Workflow {#de-6}

### 6.1 Label Studio starten

```bash
# Docker (empfohlen)
cd docker && docker compose --profile annotation up -d label-studio
# Standalone
pip install label-studio && label-studio start --port 3005
```

Zugriff: http://localhost:3005

### 6.2 Projekt erstellen

1. **Create Project** — Namen vergeben
2. **Labeling Setup** → Object Detection with Bounding Boxes
3. Labels anlegen (exakte Schreibweise wichtig für YOLO-Export):

```
capitate-stalked    #FF4444 (Rot)
capitate-sessile    #44FF44 (Grün)
bulbous             #4444FF (Blau)
non-glandular       #FFAA00 (Orange)
```

4. **Bilder importieren**: Settings → Cloud Storage → Add Source Storage → Local Files → Pfad zur Session setzen

### 6.3 VLM-Vorlabeling (3–5x schneller annotieren)

```bash
curl -X POST http://localhost:8000/api/v1/vlm/label \
  -H "Content-Type: application/json" \
  -d '{
    "image_paths": ["data/raw/session_20260101/bild001.jpg"],
    "model": "florence2",
    "confidence_threshold": 0.3
  }'
```

Kandidaten-Boxen landen in der Review-Queue — nie direkt in Trainingsdaten. Im Label Studio sieht man vorgefüllte Boxen, die man korrigieren, ergänzen oder ablehnen kann.

| Modell | VRAM | Geschwindigkeit | Qualität |
|---|---|---|---|
| Moondream-2B (4-bit) | ~1,4 GB | Schnell | Gut für Erkennung |
| Florence-2-large (4-bit) | ~2,1 GB | Mittel | Beste Genauigkeit |
| Qwen2-VL-7B (4-bit) | ~4,8 GB | Langsam | Höchste Qualität |

### 6.4 Annotations-Standards

```
Box-Zeichenregeln:
  JA: Box eng um den Trichom-Kopf (nicht den Stiel)
  JA: Ganzen Kopf einschließen, auch wenn teilweise verdeckt
  JA: Trichome am Bildrand markieren
  JA: ALLE sichtbaren Trichome annotieren — kein selektives Überspringen
  JA: Unsicherheit gestielt/sitzend — sichtbaren Hals suchen

  NEIN: Boxen um nackte Stiele (kein Kopf)
  NEIN: Debris oder Artefakte annotieren
  NEIN: Erkennbare, unscharfe Trichome überspringen
```

### 6.5 Annotierungen exportieren

```bash
# Label Studio UI: Project → Export → YOLO format → Download

# Via API
curl -X POST http://localhost:8000/api/v1/annotation/export \
  -H "Content-Type: application/json" \
  -d '{"project_id": 1, "format": "yolo", "output_dir": "data/annotated/session_20260101"}'
```

### 6.6 Annotierungsqualität prüfen

```bash
curl -X POST http://localhost:8000/api/v1/annotation/stats \
  -H "Content-Type: application/json" \
  -d '{"annotation_dir": "data/annotated/session_20260101"}'
```

Ausgabe: Klassenverteilung, Cohen's κ, Box-Größenverteilung, verdächtige Annotierungen.

---

## DE 7. Training-Workflow {#de-7}

### 7.1 Dataset-Split vorbereiten

```bash
# Export a Label Studio project as a YOLO dataset. Whole imaging sessions go to one split each
# (task field data.session, else the image folder, else the file-name series), so near-identical
# frames never leak between train / val / test. The export log reports images and sessions per split.
curl -X POST http://localhost:8000/api/v1/training/prepare-ls-dataset \
  -H "Content-Type: application/json" \
  -d '{"project_id": 1, "train_ratio": 0.70, "val_ratio": 0.15, "seed": 42}'
```

### 7.2 Training konfigurieren

`configs/training/yolo11s_detection.yaml`:

```yaml
model: yolo11s.pt           # Startgewichte (automatischer Download)
task: detect
data: data/splits/dataset.yaml
imgsz: 1280                 # Für Tiled Inference nötig
batch: 8                    # Optimal für RTX 4060 8GB
workers: 4
epochs: 100
patience: 20
lr0: 0.01
cos_lr: true

# Augmentierung (mikroskopie-spezifisch)
degrees: 90.0               # Volle Rotation — Trichome haben keine Standardausrichtung
flipud: 0.5
fliplr: 0.5
hsv_h: 0.015                # Kleiner Farbtonversatz — Beleuchtung variiert
hsv_s: 0.7
mosaic: 0.3                 # Niedrigeres Mosaik

device: 0
amp: true                   # FP16 Mixed Precision
```

### 7.3 Training starten

```bash
# Via CLI
trichome train start --config configs/training/yolo11s_detection.yaml

# Via API (nicht-blockierend, Progress via WebSocket)
curl -X POST http://localhost:8000/api/v1/training/start \
  -H "Content-Type: application/json" \
  -d '{"config_path": "configs/training/yolo11s_detection.yaml"}'

# Live-Dashboard:  http://localhost:3000/training
# WebSocket:       ws://localhost:8000/ws/training
# MLflow:          http://localhost:3004
```

### 7.4 Trainings-Ausgabe

```
runs/detect/trichome_yolo11s_20260101/
    weights/best.pt      <- dieses für Inferenz verwenden
    weights/last.pt
    results.csv
    confusion_matrix.png
    PR_curve.png
    val_batch0_pred.jpg
```

---

## DE 8. Verifizierung und Benchmarking {#de-8}

### 8.1 Auf Test-Set evaluieren

```bash
trichome benchmark detection \
  --weights runs/detect/trichome_yolo11s_20260101/weights/best.pt \
  --split test --data data/splits/dataset.yaml \
  --conf 0.25 --iou 0.5
```

Erwartete Ausgabe:

```
Klasse              P      R      mAP50  mAP50-95
all                 0.887  0.862  0.883  0.512
capitate-stalked    0.921  0.905  0.918  0.561
capitate-sessile    0.873  0.841  0.864  0.498
bulbous             0.841  0.812  0.832  0.445
non-glandular       0.913  0.890  0.918  0.543
```

### 8.2 Konfidenz-Kalibrierung

YOLO-Konfidenzscores sind unkalibriert. Vor dem Deployment korrigieren:

```bash
curl -X POST http://localhost:8000/api/v1/detection/calibrate \
  -H "Content-Type: application/json" \
  -d '{"weights_path": "runs/.../best.pt", "val_data": "data/splits/val/", "method": "temperature"}'
```

Ziel: ECE < 0,05. Reliability-Diagramme werden automatisch generiert.

### 8.3 TensorRT-Engine bauen (Produktions-Inferenz)

```bash
# YOLO nach ONNX exportieren
python -c "
from ultralytics import YOLO
YOLO('runs/.../best.pt').export(format='onnx', imgsz=1280, dynamic=True, half=True)
"

# TRT FP16 Engine bauen
trichome convert tensorrt runs/.../best.pt \
  --imgsz 1280 --fp16 --workspace-gb 4

# TRT vs. PyTorch benchmarken
trichome benchmark inference \
  --engine models/trichome_yolo11s_fp16.engine \
  --pytorch runs/.../best.pt \
  --image data/splits/test/images/ --n 100
```

### 8.4 Tiled Inference Benchmark

```bash
trichome benchmark tiled \
  --weights runs/.../best.pt \
  --image data/splits/test/images/hochaufloesung_001.jpg \
  --tile-sizes 640 1280 --overlaps 0.1 0.2 0.3
```

---

## DE 9. Verbesserungs-Loop {#de-9}

### 9.1 Aktives Lernen — Schwierige Fälle finden

```bash
curl -X POST http://localhost:8000/api/v1/active_learning/sample \
  -H "Content-Type: application/json" \
  -d '{"strategy": "uncertainty", "n_samples": 50, "unlabeled_dir": "data/raw/neue_session/"}'
```

Die zurückgegebenen Bilder zuerst labeln — sie verbessern das Modell am meisten pro Annotierungsstunde.

### 9.2 Häufige Schwachstellen

| Problem | Ursache | Lösung |
|---|---|---|
| Kugelförmige Trichome werden übersehen | Zu klein in Trainingsdaten | Gezielte Nahaufnahmen sammeln |
| Falsch-Positive auf Debris | Debris sieht aus wie Trichome | Debris als non-glandular annotieren |
| Gestielt/sitzend-Verwechslung | Kurzer Stiel aus schlechtem Winkel | Mehr Winkel-Varianten aufnehmen |
| Schlechte Erkennung an Bildrändern | Padding-Artefakte | Überlappung bei Tiled Inference erhöhen |
| Niedriges mAP bei hohem IoU | Zu lockere Box-Zeichnung | Engeres Annotations-Protokoll durchsetzen |

### 9.3 Checkliste nach jeder Trainingsrunde

```
  [ ] Konfusionsmatrix prüfen — welche Klasse wird verwechselt?
  [ ] val_batch*_pred.jpg ansehen — wo versagt das Modell?
  [ ] Aktives Lernen auf neue unlabeled Daten anwenden
  [ ] Klassenverteilung prüfen (ausgeglichen?)
  [ ] Gezielte Bilder für schwache Klassen hinzufügen
  [ ] Annotierungsqualität prüfen (Cohen kappa >= 0,80)
  [ ] Keine Session-Überschneidungen in Splits
  [ ] Kalibrierung nach jedem neuen Training neu ausführen
```

### 9.4 Retraining-Trigger

```bash
curl http://localhost:8000/api/v1/active_learning/trigger | python -m json.tool
```

Trigger feuern wenn:
- 100+ neue annotierte Bilder seit letztem Training
- Mittlere Unsicherheit > 0,45
- Neue Klassenverteilung > 15% Abweichung von der Trainingsverteilung

---

## DE 10. Docker-Deployment {#de-10}

### Core-Stack (nginx + backend + frontend + MLflow)

```bash
cd docker
docker compose build       # nur beim ersten Mal
docker compose up -d
docker compose logs -f
docker compose down
```

### Mit Annotations-Tools (Label Studio + CVAT + PostgreSQL)

```bash
docker compose --profile annotation up -d
```

### Mit GPU-Training-Stack

```bash
docker compose -f docker-compose.yml -f docker-compose.training.yml up -d
docker exec trichome-backend nvidia-smi   # GPU-Zugriff prüfen
```

### Nur Inferenz (leichtgewichtig, kein Frontend)

```bash
docker compose -f docker-compose.inference.yml up -d
```

### Umgebung für Docker einrichten

```bash
cp .env.example .env
# Container-interne Pfade verwenden:
# DATA_ROOT=/data
# MODELS_ROOT=/models
# MLFLOW_TRACKING_URI=http://mlflow:5000   <- internes Docker-DNS
```

### Daten-Volumes

```bash
docker volume ls | grep trichome
# trichome-models        Modell-Gewichte (geteilt)
# trichome-mlflow        Experiment-Daten
# trichome-db            SQLite-Datenbank
# trichome-label-studio  Label Studio-Daten
```

### Update / Rebuild

```bash
cd docker && git pull
docker compose build --no-cache && docker compose up -d
```

---

## DE 11. Alle URLs und Seiten {#de-11}

### Dev-Modus (kein Docker)

| Dienst | URL | Zweck |
|---|---|---|
| Frontend | http://localhost:3000 | Haupt-Web-UI |
| API Swagger | http://localhost:8000/docs | Interaktive API-Docs |
| API ReDoc | http://localhost:8000/redoc | API-Referenz |
| WS Training | ws://localhost:8000/ws/training | Live-Training-Stream |
| WS System | ws://localhost:8000/ws/system | System-/GPU-Stats |
| WS Jobs | ws://localhost:8000/ws/jobs | Hintergrund-Job-Status |
| WS Logs | ws://localhost:8000/ws/logs | Live-Log-Stream |

### Docker-Modus

| Dienst | Lokal | Öffentlich (via Nginx) |
|---|---|---|
| Nginx-Gateway | http://localhost:3001 | http://your-domain.com:3001 |
| Backend-API | http://localhost:3002/api/v1 | .../api/v1/ |
| API-Docs | http://localhost:3002/docs | .../docs |
| Frontend | http://localhost:3003 | .../ |
| MLflow | http://localhost:3004 | .../mlflow/ |
| Label Studio | http://localhost:3005 | .../annotation/ |
| CVAT | http://localhost:3006 | .../cvat/ |

### Frontend-Seiten

| Seite | Pfad | Funktion |
|---|---|---|
| Dashboard | / | Systemübersicht, GPU-Status, aktuelle Jobs |
| Inferenz | /inference | Bild hochladen, Erkennung/Segmentierung |
| Datensätze | /datasets | Datensätze verwalten |
| Annotierung | /annotation | VLM-Reviews, Label Studio verwalten |
| Label Studio | /labelstudio | Eingebettetes Label Studio |
| Training | /training | Training starten, überwachen, vergleichen |
| Modelle | /models | Modell-Registry, Versionen |
| Experimente | /experiments | MLflow-Vergleich |
| Morphologie | /morphology | Morphologie-Analyseergebnisse |
| Analytics | /analytics | PDF/CSV/JSON-Berichte generieren |
| Video | /video | Video-Pipeline, Frame-Extraktion |
| Berichte | /reports | Vergangene Berichte |
| Benchmarks | /benchmarks | Benchmark-Historie |
| System | /system | Hardware-Stats, Prozessmonitor |
| Prozesse | /processes | Container-Verwaltung, docker compose, Live-Logs |
| Einstellungen | /settings | Konfiguration, Kalibrierung, API-Schlüssel |
| Wiki | /wiki | In-App-Dokumentation (DE/EN/ES, 14 Seiten) |

---

## DE 12. API-Referenz {#de-12}

Basispfad: `/api/v1/` — Vollständige Docs: http://localhost:8000/docs

```bash
# Erkennung
POST /detection/infer               # Einzelbild-Inferenz
POST /detection/infer/tiled         # Tiled Inferenz (4K-Bilder)
POST /detection/infer/batch         # Batch-Inferenz
POST /detection/calibrate           # Konfidenz-Kalibrierung

# Segmentierung
POST /segmentation/segment          # SAM2-Instanz-Segmentierung
POST /segmentation/refine           # Maske verfeinern

# Reifegrad
POST /maturity/classify             # Reifegrad klassifizieren
POST /maturity/classify/batch       # Batch-Klassifikation

# Training
POST /training/start                # Training starten
GET  /training/status               # Trainingsstatus
POST /training/stop                 # Training stoppen
GET  /training/runs                 # Alle Runs auflisten
GET  /training/runs/{run_id}        # Run-Details + Metriken
POST /training/evaluate             # Auf Test-Set evaluieren

# VLM-Vorlabeling
POST /vlm/label                     # Vorlabeling starten (-> Review-Queue)
GET  /vlm/queue                     # Ausstehende Reviews
POST /vlm/queue/{id}/approve        # Vorlabel bestätigen
POST /vlm/queue/{id}/reject         # Vorlabel ablehnen

# Aktives Lernen
POST /active_learning/sample        # Unsichere Samples abrufen
GET  /active_learning/trigger       # Retraining-Trigger prüfen

# Annotierung
POST /annotation/export             # Export (YOLO/COCO/CSV)
POST /annotation/stats              # Qualitätsstatistiken
GET  /annotation/projects           # Label Studio-Projekte

# Berichte
POST /analytics/report              # PDF/CSV/JSON generieren

# System
GET  /system/health                 # Health-Check + GPU-Stats
GET  /system/gpu                    # VRAM, Temperatur, Auslastung
```

---

## DE 13. CLI-Referenz {#de-13}

```bash
trichome status                                   # GPU, models, API
trichome serve                                    # start the FastAPI backend

trichome detect image.jpg                         # YOLO detection (tiled for large images)
trichome segment image.jpg                        # SAM2 instance segmentation
trichome maturity image.jpg                       # optical maturity (colour + texture)
trichome calibrate run                            # pixel -> µm from a stage micrometer
trichome video extract clip.mp4                   # best frames from a microscopy video

trichome train start --data dataset.yaml --model yolo11s --imgsz 1280 --batch 8
trichome train evaluate                           # metrics on a labelled split
trichome train list                               # recent runs

trichome benchmark all                            # focus, maturity, morphology, measurement, video, detection
trichome convert tensorrt best.pt --fp16          # .pt -> ONNX -> TensorRT engine
trichome export run <session> -f pdf,csv,json     # scientific reports
trichome annotate --help                          # VLM pre-labelling (always human-reviewed)
```

Every command has `--help` with all options.

---

## DE 14. Konfiguration {#de-14}

```env
# Pfade
TRICHOME_ROOT=/path/to/trichome-analysis
DATA_ROOT=/mnt/data/trichome
MODELS_ROOT=/mnt/models/trichome

# Hardware
CUDA_VISIBLE_DEVICES=0
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
VRAM_LIMIT_GB=8.0
GPU_INFERENCE_QUEUE_DEPTH=0

# Datenbank
DATABASE_URL=sqlite:///./trichome.db

# Annotierung
LABEL_STUDIO_URL=http://localhost:3005
LABEL_STUDIO_API_KEY=dein_schluessel
CVAT_URL=http://localhost:3006

# Experiment-Tracking
MLFLOW_TRACKING_URI=http://localhost:3004
EXPERIMENT_TRACKER=mlflow

# VLM-Modelle
FLORENCE2_MODEL_ID=microsoft/Florence-2-large
MOONDREAM_MODEL_ID=vikhyatk/moondream2
VLM_CACHE_DIR=/mnt/models/vlm_cache
```

---

## DE 15. Architektur {#de-15}

### Modul-Struktur (Domain-Driven Design)

```
<modul>/
  domain/          # Reine Geschäftslogik (keine Framework-Abhängigkeiten)
  application/     # Orchestriert Domain-Objekte (Pipelines)
  infrastructure/  # Modell-Backends, Datei-I/O, externe APIs
  api/             # FastAPI-Router
  schemas/         # Pydantic Request/Response-Modelle
```

### CV-Pipeline

```
Bild
  -> Fokus-Scorer (unscharfe Frames verwerfen)
  -> Tiled Inferenz (YOLO v11s, 1280px Kacheln, 20% Überlappung)
  -> Konfidenz-Kalibrierung (Temperature Scaling)
  -> [Optional] RTMDet Ensemble
  -> SAM2-tiny (Prompt-basierte Segmentierung mit YOLO-Boxen)
  -> Maskenverfeinerung (Löcher füllen, Konturen glätten)
  -> Morphologie-Klassifikator (gestielt / sitzend / kugelförmig)
  -> Reifegrad-Klassifikator (klar -> trüb -> bernstein)
  -> Messung (px -> µm via CalibrationScale)
  -> Analytics-Engine (Statistiken, Berichtgenerierung)
```

### Backend

- `backend/main.py` — FastAPI App Factory, Lifespan (DB-Init + GPU-Broadcast)
- `backend/config.py` — Settings via pydantic-settings, LRU-gecachter Singleton
- `backend/middleware/gpu_guard.py` — VRAM-Budget-Durchsetzung, HTTP 429 bei Überschreitung
- `asyncio.Semaphore(1)` — ein GPU-Task gleichzeitig, global durchgesetzt

---

## DE 16. Wissenschaftliche Methodik {#de-16}

### Reifegradklassifikation

Reifegrad wird **ausschließlich** aus optischen Eigenschaften beurteilt — keine chemischen Behauptungen:

| Merkmalgruppe | Verwendete Merkmale |
|---|---|
| Farbe (HSV) | Mittlerer Farbton, Sättigung, Hellwert pro Trichom-Region |
| Farbe (LAB) | L* (Helligkeit), a* (Grün-Rot), b* (Blau-Gelb) |
| Textur | LBP (Local Binary Patterns), GLCM (Co-Occurrence-Matrix), Gabor-Filter |
| Morphologie | Kopfdurchmesser, Stiellänge, Kreisförmigkeit |

**Explizite Einschränkungen:**
- Bernstein-Färbung ist kein direkter Proxy für Cannabinoid-Abbau — optische Beobachtung
- Beleuchtungsbedingungen beeinflussen Farbmerkmale erheblich
- Dieses System sagt keine THC-, CBD- oder sonstigen Cannabinoid-Konzentrationen vorher

### Kalibrierung

- Temperature Scaling (bevorzugt) — einzelner Skalar, bewahrt Ranking
- Platt Scaling — Sigmoid-Fit auf Validierungs-Logits
- Reliability-Diagramme bei jeder Evaluierung
- Ziel: ECE < 0,05

### Reproduzierbarkeit

- `GLOBAL_SEED = 42` in allen Trainings-, Sampling- und Augmentierungs-Pipelines
- Datensatz-Splits deterministisch (Session-Hash-basiert)
- Alle Benchmark-Ergebnisse in `docs/progress/benchmark_history.md`

---

## DE 17. Tests {#de-17}

```bash
# Vollständige Test-Suite
pytest tests/ -v

# Schnell (ohne GPU und langsame Integrationstests)
pytest tests/ -m "not gpu and not slow and not integration" -v

# Einzelnes Modul
pytest tests/unit/test_detection_metrics.py -v
pytest tests/unit/test_tensorrt_runner.py -v
pytest tests/unit/test_inference_tiling.py -v

# Mit Coverage
pytest tests/ --cov=. --cov-report=html

# GPU-Tests (erfordert physische GPU + TRICHOME_ENGINE env var)
pytest tests/ -m gpu -v
```

**Aktueller Stand: 1694 passed, 4 skipped (GPU-only + reportlab-Guard)**

| Modul | Tests |
|---|---|
| Erkennungsmetriken | 45 |
| Reifegrad-Klassifikator | 38 |
| Segmentierung | 41 |
| VLM Schema Enforcer | 63 |
| Annotierungsstatistiken | 36 |
| Analytics Export | 61 |
| TensorRT Runner + Builder | 35 |
| Tiled Inference | 57 |
| Alle anderen Module | 584 |
