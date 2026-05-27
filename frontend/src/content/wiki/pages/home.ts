import type { WikiPage } from '../types';

const en = `
## What is CTIP?

CTIP (Cannabis Trichome Intelligence Platform) analyzes cannabis trichomes from microscopy images using computer vision. It detects, classifies, and measures individual trichomes without any manual counting.

**Core pipeline:**
1. YOLO11s detects all trichomes (tiled inference for large images)
2. SAM2-tiny generates pixel-accurate masks per instance
3. Classifier assigns: Stalked / Sessile / Bulbous / Non-glandular
4. Maturity model assigns: Clear → Cloudy → Amber (optical only — no THC claims)
5. Calibrated scale converts pixel size to µm

> **Scientific constraint**: CTIP never predicts cannabinoid concentrations. Maturity = optical observation only.

## Architecture overview

\`\`\`
Browser (:3001)
    │
nginx (Reverse Proxy)
    ├── /api/v1/*  →  FastAPI Backend (:8000)
    └── /*         →  Next.js Frontend (:3000)

Supporting services:
    ├── Label Studio (:3005)  — annotation
    ├── CVAT        (:3006)  — annotation (alternative)
    └── MLflow      (:3004)  — experiment tracking
\`\`\`

## Quick start (5 minutes)

\`\`\`bash
git clone https://github.com/ottco-dev/ctip-oss.git
cd ctip-oss

# Python environment
pip install uv
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e ".[all]"

# Frontend
cd frontend && npm install && cd ..

# Start everything
./scripts/dev-start.sh

# Open browser → setup wizard launches automatically
# http://localhost:3001
\`\`\`

## Hardware requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | NVIDIA 6 GB VRAM | RTX 4060 8 GB |
| RAM | 8 GB | 16 GB |
| CPU | 4 cores | i5-13400F |
| Storage | 20 GB free | 100 GB SSD |
| Python | 3.11 | 3.12 |
| CUDA | 11.8 | 12.1+ |

CPU-only works for inference (slower). Not recommended for training.

## Service ports at a glance

| Service | Port | Purpose |
|---------|------|---------|
| Main UI (nginx) | 3001 | Entry point for all requests |
| Next.js frontend | 3000 | UI server (proxied through nginx) |
| FastAPI backend | 8000 | REST API + WebSockets |
| MLflow | 3004 | Experiment tracking UI |
| Label Studio | 3005 | Annotation platform (Docker) |
| CVAT | 3006 | Alternative annotation (Docker) |
`;

const de = `
## Was ist CTIP?

CTIP (Cannabis Trichome Intelligence Platform) analysiert Cannabis-Trichome aus Mikroskopaufnahmen mit Computer Vision — vollautomatische Erkennung, Klassifizierung und Messung einzelner Trichome ohne manuelles Zählen.

**Core pipeline:**
1. YOLO11s erkennt alle Trichome (tiled inference für große Bilder)
2. SAM2-tiny erzeugt pixelgenaue Masken je Instanz
3. Classifier ordnet zu: Stalked / Sessile / Bulbous / Non-glandular
4. Maturity-Modell: Clear → Cloudy → Amber (nur optisch — keine THC-Aussagen)
5. Kalibrierte Skala rechnet Pixelgröße in µm um

> **Wissenschaftliche Einschränkung**: CTIP macht keine Cannabinoid-Konzentrationsangaben. Reifegrad = rein optische Beobachtung.

## Architektur-Übersicht

\`\`\`
Browser (:3001)
    │
nginx (Reverse Proxy)
    ├── /api/v1/*  →  FastAPI Backend (:8000)
    └── /*         →  Next.js Frontend (:3000)

Weitere Services:
    ├── Label Studio (:3005)  — Annotation
    ├── CVAT        (:3006)  — Annotation (Alternative)
    └── MLflow      (:3004)  — Experiment Tracking
\`\`\`

## Quick Start (5 Minuten)

\`\`\`bash
git clone https://github.com/ottco-dev/ctip-oss.git
cd ctip-oss

# Python-Umgebung
pip install uv
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e ".[all]"

# Frontend
cd frontend && npm install && cd ..

# Alles starten
./scripts/dev-start.sh

# Browser öffnen → Setup-Wizard startet automatisch
# http://localhost:3001
\`\`\`

## Hardware-Anforderungen

| Komponente | Minimum | Empfohlen |
|-----------|---------|-----------|
| GPU | NVIDIA 6 GB VRAM | RTX 4060 8 GB |
| RAM | 8 GB | 16 GB |
| CPU | 4 Kerne | i5-13400F |
| Speicher | 20 GB frei | 100 GB SSD |
| Python | 3.11 | 3.12 |
| CUDA | 11.8 | 12.1+ |

CPU-only funktioniert für Inferenz (langsamer). Für Training nicht empfohlen.

## Service-Ports im Überblick

| Service | Port | Zweck |
|---------|------|-------|
| Main UI (nginx) | 3001 | Einstiegspunkt für alle Anfragen |
| Next.js Frontend | 3000 | UI-Server (über nginx weitergeleitet) |
| FastAPI Backend | 8000 | REST API + WebSockets |
| MLflow | 3004 | Experiment Tracking UI |
| Label Studio | 3005 | Annotation-Plattform (Docker) |
| CVAT | 3006 | Alternative Annotation (Docker) |
`;

const es = `
## ¿Qué es CTIP?

CTIP (Cannabis Trichome Intelligence Platform) analiza tricomas de cannabis en imágenes de microscopía mediante visión por computadora — detección, clasificación y medición automática sin conteo manual.

**Pipeline principal:**
1. YOLO11s detecta todos los tricomas (inferencia en mosaico para imágenes grandes)
2. SAM2-tiny genera máscaras precisas por instancia
3. Clasificador asigna: Stalked / Sessile / Bulbous / Non-glandular
4. Modelo de madurez: Clear → Cloudy → Amber (solo óptico — sin afirmaciones de THC)
5. Escala calibrada convierte píxeles a µm

> **Restricción científica**: CTIP no predice concentraciones de cannabinoides. Madurez = observación óptica únicamente.

## Resumen de arquitectura

\`\`\`
Navegador (:3001)
    │
nginx (Proxy inverso)
    ├── /api/v1/*  →  Backend FastAPI (:8000)
    └── /*         →  Frontend Next.js (:3000)

Servicios adicionales:
    ├── Label Studio (:3005)  — anotación
    ├── CVAT        (:3006)  — anotación (alternativa)
    └── MLflow      (:3004)  — seguimiento de experimentos
\`\`\`

## Inicio rápido (5 minutos)

\`\`\`bash
git clone https://github.com/ottco-dev/ctip-oss.git
cd ctip-oss

pip install uv
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e ".[all]"

cd frontend && npm install && cd ..
./scripts/dev-start.sh
# http://localhost:3001
\`\`\`

## Requisitos de hardware

| Componente | Mínimo | Recomendado |
|-----------|--------|-------------|
| GPU | NVIDIA 6 GB VRAM | RTX 4060 8 GB |
| RAM | 8 GB | 16 GB |
| CPU | 4 núcleos | i5-13400F |
| Almacenamiento | 20 GB libre | 100 GB SSD |
| Python | 3.11 | 3.12 |
| CUDA | 11.8 | 12.1+ |
`;

const page: WikiPage = {
  slug: 'home',
  title: { en: 'CTIP Overview', de: 'CTIP Übersicht', es: 'Visión General' },
  description: {
    en: 'What is CTIP, architecture overview, hardware requirements, and quick start.',
    de: 'Was ist CTIP, Architektur-Übersicht, Hardware-Anforderungen und Quick Start.',
    es: 'Qué es CTIP, arquitectura, requisitos de hardware e inicio rápido.',
  },
  content: { en, de, es },
  section: 'setup',
  icon: '🔬',
};

export default page;
