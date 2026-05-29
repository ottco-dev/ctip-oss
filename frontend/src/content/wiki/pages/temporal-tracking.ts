import type { WikiPage } from '../types';

const en = `
## Temporal Tracking

Track individual trichomes across consecutive video frames using SORT (Simple Online and Realtime Tracking): a Kalman filter predicts the next position of each track and the Hungarian algorithm assigns incoming detections to existing tracks.

---

## How it works

\`\`\`
Video frame N
    │
    ▼
YOLO11s detection (tiled, per frame)
    │
    ▼
SORT tracker
    │  Kalman filter: predict next bounding box position
    │  Hungarian algorithm: match detections → active tracks (IoU cost matrix)
    │
    ▼
Track state update
    │  TENTATIVE  → confirmed after min_hits consecutive matches
    │  CONFIRMED  → active track carrying trajectory history
    │  DELETED    → removed after max_age consecutive misses
    │
    ▼
Trajectory data (per-track positions over time)
    │
    ▼
Summary: count, movement stats, development monitoring
\`\`\`

---

## Configuration

| Parameter | Default | Description |
|---|---|---|
| \`max_age\` | 3 | Frames a track survives without a match before deletion |
| \`min_hits\` | 2 | Consecutive matched frames required to confirm a track |
| \`iou_threshold\` | 0.3 | Minimum IoU for detection–track assignment |
| \`min_track_length\` | 3 | Minimum frames a track must span to appear in the summary |

---

## API reference

### Start a tracking session

\`\`\`bash
POST /api/v1/video/tracking/start
Content-Type: application/json

{
  "video_path": "data/raw/videos/session_01.mp4",
  "max_age": 3,
  "min_hits": 2,
  "iou_threshold": 0.3,
  "min_track_length": 3,
  "model": "yolo11s"
}
\`\`\`

Response:
\`\`\`json
{
  "session_id": "trk_abc123",
  "status": "queued",
  "frame_count": 240
}
\`\`\`

### Check session status

\`\`\`bash
GET /api/v1/video/tracking/{session_id}/status
\`\`\`

### Get session summary

\`\`\`bash
GET /api/v1/video/tracking/{session_id}/summary
\`\`\`

### Get trajectory data

\`\`\`bash
GET /api/v1/video/tracking/{session_id}/trajectories
\`\`\`

Returns an array of \`TrichomeTrack\` objects with per-frame bounding box positions for SVG overlay rendering.

### Delete session

\`\`\`bash
DELETE /api/v1/video/tracking/{session_id}
\`\`\`

---

## TrichomeTrack schema

\`\`\`json
{
  "track_id": 7,
  "state": "CONFIRMED",
  "first_frame": 4,
  "last_frame": 198,
  "trajectory_data": [
    { "frame": 4, "bbox": [112, 88, 145, 121], "confidence": 0.91 }
  ],
  "mean_confidence": 0.87,
  "total_displacement_px": 4.3
}
\`\`\`

Track states:

| State | Meaning |
|---|---|
| \`TENTATIVE\` | Fewer than \`min_hits\` consecutive matches; not yet reported |
| \`CONFIRMED\` | Sufficient matches; included in summary and trajectories |
| \`DELETED\` | Exceeded \`max_age\` without a match; archived |

---

## Frontend: Video page → Tracking tab

1. **Session setup** — configure parameters and select video file.
2. **Live progress** — frames processed / total, active track count (WebSocket \`/ws/jobs\`).
3. **Trajectory table** — one row per confirmed track: ID, length (frames), mean confidence, displacement.
4. **SVG bar chart** — track lifetimes visualised as horizontal bars across the frame timeline.
5. **Overlay export** — trajectory data exportable for external annotation tools.

---

## Use cases

- **Development monitoring**: follow the same trichome instances across a time-lapse to observe maturity progression.
- **Movement artefact detection**: high \`mean_displacement_px\` indicates camera shake or sample drift.
- **Population dynamics**: track entry/exit of trichomes during slow panning acquisitions.
`;

const de = `
## Temporales Tracking

Einzelne Trichome werden über aufeinanderfolgende Video-Frames mit dem **SORT-Tracker** (Kalman-Filter + Ungarischer Algorithmus) verfolgt.

---

## Konfiguration

| Parameter | Standard | Beschreibung |
|---|---|---|
| \`max_age\` | 3 | Frames ohne Übereinstimmung bis zur Löschung |
| \`min_hits\` | 2 | Aufeinanderfolgende Treffer zur Bestätigung |
| \`iou_threshold\` | 0.3 | Minimaler IoU für Zuweisung |
| \`min_track_length\` | 3 | Minimale Frame-Länge für die Zusammenfassung |

---

## API-Referenz

\`\`\`bash
# Tracking-Session starten
POST /api/v1/video/tracking/start

# Status abrufen
GET /api/v1/video/tracking/{session_id}/status

# Zusammenfassung abrufen
GET /api/v1/video/tracking/{session_id}/summary

# Trajektoriendaten abrufen
GET /api/v1/video/tracking/{session_id}/trajectories

# Session löschen
DELETE /api/v1/video/tracking/{session_id}
\`\`\`

---

## Track-Zustände

| Zustand | Bedeutung |
|---|---|
| \`TENTATIVE\` | Zu wenige Treffer; noch nicht gemeldet |
| \`CONFIRMED\` | Ausreichend Treffer; in Zusammenfassung enthalten |
| \`DELETED\` | \`max_age\` überschritten; archiviert |

---

## Frontend: Video-Seite → Tracking-Tab

- **Session-Konfiguration** — Parameter und Video-Datei auswählen.
- **Live-Fortschritt** — verarbeitete Frames und aktive Tracks.
- **Trajektorie-Tabelle** — eine Zeile je bestätigtem Track.
- **SVG-Balkendiagramm** — Track-Lebensdauer über den Frame-Zeitstrahl.

---

## Anwendungsfälle

- **Entwicklungsüberwachung**: Gleiche Trichom-Instanzen über Zeitrafferaufnahmen verfolgen.
- **Bewegungsartefakt-Erkennung**: Hoher \`mean_displacement_px\` deutet auf Kameraverwacklung hin.
`;

const es = `
## Seguimiento Temporal

Se realiza el seguimiento de tricomas individuales a lo largo de fotogramas de video usando el tracker **SORT** (filtro de Kalman + algoritmo húngaro).

---

## Configuración

| Parámetro | Defecto | Descripción |
|---|---|---|
| \`max_age\` | 3 | Fotogramas sin coincidencia antes de eliminar el track |
| \`min_hits\` | 2 | Coincidencias consecutivas para confirmar un track |
| \`iou_threshold\` | 0.3 | IoU mínimo para asignación |
| \`min_track_length\` | 3 | Longitud mínima en fotogramas para incluir en el resumen |

---

## Referencia de API

\`\`\`bash
POST   /api/v1/video/tracking/start
GET    /api/v1/video/tracking/{session_id}/status
GET    /api/v1/video/tracking/{session_id}/summary
GET    /api/v1/video/tracking/{session_id}/trajectories
DELETE /api/v1/video/tracking/{session_id}
\`\`\`

---

## Estados del track

| Estado | Significado |
|---|---|
| \`TENTATIVE\` | Pocas coincidencias; aún no reportado |
| \`CONFIRMED\` | Suficientes coincidencias; incluido en el resumen |
| \`DELETED\` | Superó \`max_age\` sin coincidencia; archivado |

---

## Frontend: página de Video → pestaña Tracking

- **Configuración de sesión** — parámetros y selección de archivo de video.
- **Progreso en vivo** — fotogramas procesados y tracks activos.
- **Tabla de trayectorias** — una fila por track confirmado.
- **Gráfico SVG** — duración de tracks visualizada como barras horizontales.
`;

const page: WikiPage = {
  slug: 'temporal-tracking',
  title: {
    en: 'Temporal Tracking',
    de: 'Temporales Tracking',
    es: 'Seguimiento Temporal',
  },
  description: {
    en: 'SORT tracker (Kalman filter + Hungarian assignment) for tracking trichomes across video frames.',
    de: 'SORT-Tracker (Kalman-Filter + Ungarischer Algorithmus) zur Verfolgung von Trichomen über Video-Frames.',
    es: 'Tracker SORT (filtro de Kalman + asignación húngara) para seguimiento de tricomas en video.',
  },
  content: { en, de, es },
  section: 'reference',
  icon: '🎬',
};

export default page;
