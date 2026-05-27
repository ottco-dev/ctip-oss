import type { WikiPage } from '../types';

const en = `
> Full interactive API documentation: **http://localhost:8000/docs** (Swagger UI)
> ReDoc alternative: **http://localhost:8000/redoc**

All endpoints are under \`/api/v1/\`.

---

## Setup endpoints

### \`GET /setup/status\`

Check if setup is complete.

\`\`\`json
{ "completed": true, "env_exists": true, "configured_keys": ["CUDA_DEVICE", "..."] }
\`\`\`

### \`GET /setup/config\`

Read current \`.env\` configuration. Sensitive keys are redacted.

### \`POST /setup/configure\`

Write settings to \`.env\`.

\`\`\`json
{
  "settings": {
    "CUDA_DEVICE": "cuda:0",
    "VRAM_LIMIT_GB": "8.0",
    "ENVIRONMENT": "development"
  },
  "mark_setup_complete": true
}
\`\`\`

### \`GET /setup/system-check\`

Run dependency check. Returns list of check items.

### \`GET /setup/docker/status\`

Docker availability, group membership, running containers.

### \`POST /setup/docker/start-annotation\`

Start annotation container stack.

\`\`\`json
{ "profile": "annotation" }
\`\`\`

### \`GET /setup/models/status\`

Model catalog with present/missing status. Returns \`list[ModelInfo]\`.

### \`POST /setup/models/download\`

Start background model download. Returns task ID.

\`\`\`json
{ "model_id": "yolo11s" }
\`\`\`

### \`GET /setup/models/download/{task_id}\`

Poll download progress.

\`\`\`json
{
  "task_id": "abc-123",
  "status": "downloading",
  "progress": 67,
  "filename": "yolo11s.pt",
  "size_mb": 18.4,
  "downloaded_mb": 12.3,
  "detail": "Downloading… 12.3 / 18.4 MB"
}
\`\`\`

### \`POST /setup/label-studio/create-account\`

Create Label Studio account.

\`\`\`json
{ "url": "http://localhost:3005", "email": "admin@example.com", "password": "secret" }
\`\`\`

Response:
\`\`\`json
{ "ok": true, "token": "abc123...", "already_existed": false }
\`\`\`

### \`POST /setup/label-studio/test\`

Test Label Studio connection and authentication.

### \`POST /setup/label-studio/create-project\`

Create trichome annotation project with pre-built label config.

### \`GET /setup/verification\`

Live health check of all configured services. Returns latency + HTTP status per service.

---

## Detection endpoints

### \`POST /detection/analyze\`

Analyze an uploaded image.

\`\`\`bash
curl -X POST http://localhost:8000/api/v1/detection/analyze \\
  -F "image=@IMG_0001.tif" \\
  -F "model=yolo11s" \\
  -F "tiled=true" \\
  -F "conf=0.25"
\`\`\`

Response:
\`\`\`json
{
  "job_id": "xyz-456",
  "status": "queued",
  "estimated_seconds": 8
}
\`\`\`

### \`GET /detection/jobs/{job_id}\`

Poll detection job status.

### \`GET /detection/results/{job_id}\`

Retrieve detection results (JSON).

---

## Training endpoints

### \`POST /training/start\`

\`\`\`json
{ "config": "yolo11s_detection.yaml", "dataset": "v1" }
\`\`\`

### \`GET /training/jobs\`

List all training jobs with status.

### \`DELETE /training/jobs/{job_id}\`

Stop a running training job.

---

## System endpoints

### \`GET /system/health\`

\`\`\`json
{ "status": "ok", "gpu_available": true, "gpu_busy": false }
\`\`\`

### \`GET /system/gpu\`

\`\`\`json
{
  "name": "NVIDIA GeForce RTX 4060",
  "vram_total_gb": 8.0,
  "vram_used_gb": 2.3,
  "vram_free_gb": 5.7,
  "utilization_pct": 0
}
\`\`\`

---

## WebSocket endpoints

All WebSockets at \`ws://localhost:8000/ws/\`:

| Endpoint | Data stream |
|---------|------------|
| \`/ws/training\` | Live training metrics (loss, mAP, epoch) |
| \`/ws/system\` | GPU/RAM usage every 2s |
| \`/ws/jobs\` | Background job progress updates |
| \`/ws/logs\` | Live log stream |

\`\`\`javascript
// Example: live GPU monitoring
const ws = new WebSocket('ws://localhost:8000/ws/system');
ws.onmessage = (e) => {
  const { vram_used_gb, utilization_pct } = JSON.parse(e.data);
  console.log(vram_used_gb, utilization_pct);
};
\`\`\`
`;

const de = `
> Vollständige interaktive API-Dokumentation: **http://localhost:8000/docs** (Swagger UI)

Alle Endpunkte unter \`/api/v1/\`.

## Setup-Endpunkte

| Methode | Pfad | Beschreibung |
|---------|------|--------------|
| GET | \`/setup/status\` | Setup-Status prüfen |
| GET | \`/setup/config\` | Aktuelle .env-Konfiguration lesen |
| POST | \`/setup/configure\` | Einstellungen in .env schreiben |
| GET | \`/setup/system-check\` | Abhängigkeiten prüfen |
| GET | \`/setup/docker/status\` | Docker-Status |
| POST | \`/setup/docker/start-annotation\` | Annotation-Stack starten |
| GET | \`/setup/models/status\` | Modell-Katalog mit Präsenz-Status |
| POST | \`/setup/models/download\` | Hintergrund-Download starten |
| GET | \`/setup/models/download/{task_id}\` | Download-Fortschritt abfragen |
| POST | \`/setup/label-studio/create-account\` | LS-Account erstellen |
| POST | \`/setup/label-studio/test\` | LS-Verbindung testen |
| POST | \`/setup/label-studio/create-project\` | Annotations-Projekt erstellen |
| GET | \`/setup/verification\` | Live-Gesundheitsprüfung |

## Erkennungs-Endpunkte

\`\`\`bash
# Bild analysieren
curl -X POST http://localhost:8000/api/v1/detection/analyze \\
  -F "image=@IMG_0001.tif" -F "model=yolo11s" -F "tiled=true"

# Job-Status abfragen
curl http://localhost:8000/api/v1/detection/jobs/{job_id}
\`\`\`

## WebSocket-Endpunkte

| Endpunkt | Datenstrom |
|---------|-----------|
| \`/ws/training\` | Live-Trainingsmetriken (Loss, mAP, Epoche) |
| \`/ws/system\` | GPU/RAM-Verbrauch alle 2s |
| \`/ws/jobs\` | Hintergrund-Job-Fortschritt |
| \`/ws/logs\` | Live-Log-Stream |
`;

const es = `
> Documentación API interactiva completa: **http://localhost:8000/docs** (Swagger UI)

Todos los endpoints bajo \`/api/v1/\`.

## Endpoints de configuración

| Método | Ruta | Descripción |
|--------|------|-------------|
| GET | \`/setup/status\` | Verificar estado de configuración |
| GET | \`/setup/config\` | Leer configuración .env actual |
| POST | \`/setup/configure\` | Escribir configuración en .env |
| GET | \`/setup/system-check\` | Verificar dependencias |
| GET | \`/setup/docker/status\` | Estado de Docker |
| GET | \`/setup/models/status\` | Catálogo de modelos con estado |
| POST | \`/setup/models/download\` | Iniciar descarga en segundo plano |
| GET | \`/setup/verification\` | Health check en vivo |

## Endpoints de detección

\`\`\`bash
curl -X POST http://localhost:8000/api/v1/detection/analyze \\
  -F "image=@IMG_0001.tif" -F "model=yolo11s" -F "tiled=true"
\`\`\`

## WebSockets

| Endpoint | Flujo de datos |
|---------|---------------|
| \`/ws/training\` | Métricas de entrenamiento en vivo |
| \`/ws/system\` | Uso GPU/RAM cada 2s |
| \`/ws/logs\` | Stream de logs en vivo |
`;

const page: WikiPage = {
  slug: 'api-reference',
  title: { en: 'API Reference', de: 'API-Referenz', es: 'Referencia API' },
  description: {
    en: 'All REST endpoints, request/response formats, WebSocket streams.',
    de: 'Alle REST-Endpunkte, Request/Response-Formate, WebSocket-Streams.',
    es: 'Todos los endpoints REST, formatos de solicitud/respuesta, streams WebSocket.',
  },
  content: { en, de, es },
  section: 'reference',
  icon: '📡',
};

export default page;
