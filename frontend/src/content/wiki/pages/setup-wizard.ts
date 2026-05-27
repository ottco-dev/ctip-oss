import type { WikiPage } from '../types';

const en = `
The Setup Wizard is a full-screen, OS-style first-run installer that configures every aspect of CTIP.
It launches automatically on first visit and writes all settings to \`.env\`.

## How it works

1. \`SetupGuard\` checks \`GET /api/v1/setup/status\` on every page load
2. If \`completed: false\` → redirect to \`/setup\`
3. After saving: sets \`SETUP_COMPLETED="true"\` in \`.env\`, sets \`ctip-setup-checked\` in \`sessionStorage\`
4. Re-open anytime from the sidebar: **First-Time Setup**

---

## Step 0 — Welcome

Overview of what the installer does. No user input required.

---

## Step 1 — System Check

Runs \`GET /api/v1/setup/system-check\`. Checks:

**Environment:**
| Check | What it verifies |
|-------|-----------------|
| Python | Version ≥ 3.11 |
| CUDA / nvidia-smi | GPU driver availability |
| Node.js | Version ≥ 18 |
| npm | Package manager |
| nginx | Web server |
| Git | Version control |

**Python packages (14 checked):**
\`torch\`, \`ultralytics\`, \`supervision\`, \`sam2\`, \`fastapi\`, \`uvicorn\`, \`httpx\`,
\`pydantic\`, \`sqlmodel\`, \`mlflow\`, \`cv2\`, \`PIL\`, \`numpy\`, \`sklearn\`

Red = missing required, yellow = missing optional. You can continue with warnings.

Fix missing Python packages:
\`\`\`bash
source .venv/bin/activate
uv pip install -e ".[all]"
\`\`\`

---

## Step 2 — Network

Configure how CTIP is exposed:

| Field | Default | Description |
|-------|---------|-------------|
| Enable public access | off | Toggle to enter a domain |
| Public Domain | (empty = localhost only) | Your DDNS hostname, e.g. \`mylab.ddns.net\` |
| Public Port | 3001 | nginx listens on this port |

**Localhost-only** (default): nginx only binds to \`127.0.0.1:3001\`. Not reachable from outside.

**Public**: Enter your domain → nginx listens on \`0.0.0.0:3001\` and sets \`server_name\`.
Update your router to port-forward to this machine.

---

## Step 3 — Hardware

| Field | Description |
|-------|-------------|
| CUDA Device | \`cuda:0\` / \`mps\` / \`cpu\` |
| CUDA_VISIBLE_DEVICES | GPU index (e.g. \`0\`) |
| Total VRAM (GB) | Your GPU's VRAM (RTX 4060 = 8.0) |
| Inference Reserve (GB) | Budget kept for inference tasks (default 2.0) |
| Environment | \`development\` or \`production\` |

CTIP enforces **one GPU task at a time** via an asyncio semaphore. The VRAM budget prevents OOM errors by refusing tasks that would exceed the limit.

---

## Step 4 — Storage

| Field | Description |
|-------|-------------|
| Data Root | Parent directory for all CTIP data |
| Models Directory | Trained weight files (\`.pt\`, \`.engine\`) |
| Outputs Directory | Detection results, PDFs, CSV exports |

All paths must be **absolute**. Directories are created automatically on startup.

Example for a standard Linux install:
\`\`\`bash
DATA_ROOT="/home/youruser/ctip-oss/data"
MODELS_DIR="/home/youruser/ctip-oss/data/models"
OUTPUTS_DIR="/home/youruser/ctip-oss/data/outputs"
\`\`\`

---

## Step 5 — Docker

Checks Docker availability and manages the annotation container stack.

**Status panel:**
- Docker Engine: reachable / not reachable
- docker group: yes / needs fix

**If not in docker group:**
\`\`\`bash
sudo usermod -aG docker $USER && newgrp docker
\`\`\`
The exact command is shown in the UI with a copy button.

**Container list:** shows all running/stopped annotation containers (Label Studio, CVAT, PostgreSQL).

**Start Annotation Stack** button runs:
\`\`\`bash
docker compose --profile annotation up -d
\`\`\`

---

## Step 6 — ML Models

Downloads YOLO11 and SAM2 model weights. Required models must be present before inference.

| Model | File | Size | Required | Purpose |
|-------|------|------|----------|---------|
| YOLO11n | yolo11n.pt | 5.4 MB | No | Fastest, lowest accuracy |
| YOLO11s | yolo11s.pt | 18.4 MB | **Yes** | Default — best balance for RTX 4060 |
| YOLO11m | yolo11m.pt | 43.0 MB | No | Higher accuracy, needs 12+ GB VRAM |
| SAM2-tiny | sam2_hiera_tiny.pt | 38.9 MB | **Yes** | Instance segmentation |
| SAM2-small | sam2_hiera_small.pt | 46.1 MB | No | Better masks, +20% VRAM |

**Download flow:**
1. Click **Download** → \`POST /api/v1/setup/models/download\` starts a background task
2. Progress bar polls \`GET /api/v1/setup/models/download/{task_id}\` every 500ms
3. On completion: catalog refreshes, model marked as present

**Download All Required** downloads yolo11s + sam2-tiny in parallel.

---

## Step 7 — Label Studio

Full Label Studio setup in 3 sub-sections:

### Account Creation
Enter email + password → \`POST /api/v1/setup/label-studio/create-account\`
- Creates the first admin account on a fresh LS instance
- If account already exists: retrieves the API token
- API key is auto-filled below

### Connection Test
Enter API key → \`POST /api/v1/setup/label-studio/test\`
- Checks reachability
- Authenticates via \`GET /api/current-user/whoami\`
- Shows username and existing project count

### Annotation Project
Enter project name → **Create Annotation Project** → \`POST /api/v1/setup/label-studio/create-project\`

Creates a project with this label config (pre-built):
\`\`\`xml
<RectangleLabels name="label" toName="image">
  <Label value="stalked"        background="#22d3ee" hotkey="1"/>
  <Label value="sessile"        background="#34d399" hotkey="2"/>
  <Label value="bulbous"        background="#a78bfa" hotkey="3"/>
  <Label value="non-glandular"  background="#fb923c" hotkey="4"/>
</RectangleLabels>
<Choices name="quality">good / blurry / poor</Choices>
<TextArea name="notes" .../>
\`\`\`

---

## Step 8 — Services (Experiment Tracking)

### MLflow
| Field | Description |
|-------|-------------|
| Tracking URI | \`http://localhost:3004\` — local MLflow server |
| Default Experiment | Name for training runs (e.g. \`trichome-detection\`) |

### Weights & Biases (optional)
Toggle W&B logging, enter API key + project name.

---

## Step 9 — Security

| Field | Description |
|-------|-------------|
| Secret Key | 64-char random key for session signing — generate with the button |
| API Token | Optional bearer token for API authentication (empty = no auth) |

If CTIP is internet-facing, always set both. The **Generate random** button creates a cryptographically random 64-character key.

---

## Step 10 — Review

Full summary of all configured values. Sensitive fields (keys, tokens) show as \`••••••••\`.
Click **Save & Finish** → writes to \`.env\` → advances to Verification.

---

## Step 11 — Verification

Live health check of all services. Terminal-style log:

\`\`\`
✓ PASS  Backend API          HTTP 200  12ms
✓ PASS  Frontend             HTTP 200  8ms
✓ PASS  MLflow               HTTP 200  34ms
✗ FAIL  Label Studio         Connection refused
✓ PASS  nginx health check   HTTP 200  1ms
\`\`\`

FAIL on optional services (Label Studio, CVAT) is expected if Docker isn't running.
Click **Go to Dashboard** when ready.
`;

const de = `
Der Setup-Wizard ist ein vollbildschirmiger, OS-ähnlicher Ersteinrichtungs-Assistent, der alle Aspekte von CTIP konfiguriert und Einstellungen in \`.env\` schreibt.

## Funktionsweise

1. \`SetupGuard\` prüft \`GET /api/v1/setup/status\` bei jedem Seitenaufruf
2. Bei \`completed: false\` → Weiterleitung zu \`/setup\`
3. Nach dem Speichern: \`SETUP_COMPLETED="true"\` in \`.env\`, \`ctip-setup-checked\` in \`sessionStorage\`
4. Jederzeit wieder aufrufbar über die Seitenleiste: **First-Time Setup**

---

## Schritt 0 — Willkommen

Übersicht was der Installer macht. Keine Eingabe erforderlich.

---

## Schritt 1 — System Check

Führt \`GET /api/v1/setup/system-check\` aus und prüft:

**Umgebung:** Python, CUDA/nvidia-smi, Node.js, npm, nginx, Git

**Python-Pakete (14 geprüft):**
\`torch\`, \`ultralytics\`, \`supervision\`, \`sam2\`, \`fastapi\`, \`uvicorn\` usw.

Rot = fehlendes Required-Paket, Gelb = fehlendes optionales Paket.

Fehlende Pakete installieren:
\`\`\`bash
source .venv/bin/activate
uv pip install -e ".[all]"
\`\`\`

---

## Schritt 2 — Netzwerk

| Feld | Standard | Beschreibung |
|------|---------|--------------|
| Öffentlicher Zugriff | aus | Toggle → Domain eingeben |
| Öffentliche Domain | (leer = nur localhost) | Z.B. \`mylab.ddns.net\` |
| Port | 3001 | nginx hört auf diesem Port |

**Nur localhost** (Standard): nginx bindet nur an \`127.0.0.1:3001\`. Von außen nicht erreichbar.

**Öffentlich**: Domain eingeben → nginx hört auf \`0.0.0.0:3001\`. Router muss Port-Forwarding eingerichtet haben.

---

## Schritt 3 — Hardware

| Feld | Beschreibung |
|------|--------------|
| CUDA Device | \`cuda:0\` / \`mps\` / \`cpu\` |
| CUDA_VISIBLE_DEVICES | GPU-Index (z.B. \`0\`) |
| Total VRAM (GB) | VRAM deiner GPU (RTX 4060 = 8.0) |
| Inference Reserve (GB) | Budget für Inferenz-Tasks (Standard 2.0) |
| Environment | \`development\` oder \`production\` |

CTIP erzwingt **einen GPU-Task gleichzeitig** via asyncio-Semaphore.

---

## Schritt 4 — Speicher

Absolute Pfade für Data Root, Models Directory und Outputs Directory.
Verzeichnisse werden beim Start automatisch erstellt.

---

## Schritt 5 — Docker

Zeigt Docker-Status, docker-Gruppen-Mitgliedschaft, Container-Liste. Button **Start Annotation Stack** startet Label Studio + CVAT.

**Falls nicht in docker-Gruppe:**
\`\`\`bash
sudo usermod -aG docker $USER && newgrp docker
\`\`\`
(Befehl wird mit Kopier-Button angezeigt)

---

## Schritt 6 — ML Models

| Modell | Datei | Größe | Required | Zweck |
|--------|-------|-------|----------|-------|
| YOLO11n | yolo11n.pt | 5,4 MB | Nein | Schnellstes Modell |
| YOLO11s | yolo11s.pt | 18,4 MB | **Ja** | Standard für RTX 4060 |
| YOLO11m | yolo11m.pt | 43,0 MB | Nein | Höhere Genauigkeit, 12+ GB VRAM |
| SAM2-tiny | sam2_hiera_tiny.pt | 38,9 MB | **Ja** | Instanz-Segmentierung |
| SAM2-small | sam2_hiera_small.pt | 46,1 MB | Nein | Bessere Masken |

Download startet im Hintergrund, Fortschrittsbalken aktualisiert alle 500ms.

---

## Schritt 7 — Label Studio

**Account-Erstellung** → API-Key wird automatisch befüllt
**Verbindungstest** → Erreichbarkeit + Authentifizierung
**Annotations-Projekt** erstellen → 4 Klassen + Qualität + Notizen

---

## Schritt 8–9 — Services & Sicherheit

MLflow-URI, optionales W&B, Secret Key (64-Zeichen-Zufallsschlüssel) und API-Token.

---

## Schritt 10 — Review

Vollständige Konfigurationsübersicht. **Save & Finish** → schreibt \`.env\` → startet Verifikation.

---

## Schritt 11 — Verifikation

Live-Gesundheitsprüfung aller Services im Terminal-Stil mit PASS/FAIL, HTTP-Status und Latenz.
`;

const es = `
El Setup Wizard es un instalador de primera ejecución estilo OS que configura todos los aspectos de CTIP y escribe los ajustes en \`.env\`.

## Cómo funciona

1. \`SetupGuard\` comprueba \`GET /api/v1/setup/status\` en cada carga de página
2. Si \`completed: false\` → redirige a \`/setup\`
3. Al guardar: escribe \`SETUP_COMPLETED="true"\` en \`.env\`
4. Re-abrir en cualquier momento desde la barra lateral: **First-Time Setup**

## Pasos resumidos

| Paso | Nombre | Qué configura |
|------|--------|---------------|
| 0 | Bienvenida | Resumen del instalador |
| 1 | System Check | Python, CUDA, Node.js, paquetes |
| 2 | Network | Dominio, puerto nginx |
| 3 | Hardware | GPU, VRAM, entorno |
| 4 | Storage | Rutas de datos absolutas |
| 5 | Docker | Estado de contenedores, inicio |
| 6 | ML Models | Descarga YOLO11s + SAM2-tiny |
| 7 | Label Studio | Cuenta, conexión, proyecto |
| 8 | Services | MLflow, W&B |
| 9 | Security | Secret key, API token |
| 10 | Review | Resumen completo |
| 11 | Verification | Health check en vivo |

**Modelos requeridos:** YOLO11s (18.4 MB) + SAM2-tiny (38.9 MB) — descarga con barra de progreso en tiempo real.

**Label Studio:** Crea cuenta → auto-rellena API key → test de conexión → crea proyecto con 4 clases predefinidas.
`;

const page: WikiPage = {
  slug: 'setup-wizard',
  title: { en: 'Setup Wizard', de: 'Setup-Wizard', es: 'Asistente de Configuración' },
  description: {
    en: 'The 11-step OS-style installer — every step explained in detail.',
    de: 'Der 11-Schritt-OS-Installer — jeder Schritt im Detail erklärt.',
    es: 'El instalador estilo OS de 11 pasos — cada paso explicado en detalle.',
  },
  content: { en, de, es },
  section: 'setup',
  icon: '🧙',
};

export default page;
