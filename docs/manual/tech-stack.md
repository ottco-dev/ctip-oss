# Technology Stack & Rationale

> [English](en.md) · [Deutsch](de.md) · [Español](es.md) · [Technology stack](tech-stack.md) · [Back to README](../../README.md)

---

## 🇬🇧 Why These Technologies?

Every technology choice in CTIP was made deliberately. This section explains what we use and why we picked it over the alternatives.

---

### Python 3.12

**Why:** 3.12 brings significant performance improvements (5–15% faster than 3.10 in CPU-bound scientific workloads) and better error messages. The scientific ML ecosystem (PyTorch, NumPy, OpenCV) fully supports it. We use `uv` as the package manager — it is 10–100x faster than pip for dependency resolution and installs.

---

### PyTorch 2.x + CUDA 12.x

**Why PyTorch over TensorFlow:** PyTorch has dominated research and production CV since 2020. The ecosystem — Ultralytics, SAM2, Hugging Face Transformers — is all PyTorch-native. Dynamic computation graphs make debugging easier. TorchCompile (`torch.compile`) provides significant inference speedups without leaving Python.

**Why CUDA 12.x:** Required for TensorRT 10.x and the latest PyTorch CUDA kernels. CUDA 12.x introduces improved memory management (`expandable_segments:True` in `PYTORCH_CUDA_ALLOC_CONF`) which reduces fragmentation on 8 GB VRAM cards — critical for this hardware target.

---

### YOLO v11s — Detection Backbone

**Why YOLO over Faster-RCNN / DETR / RT-DETR:**
- YOLO v11s achieves comparable mAP to heavier two-stage detectors at 3–5x the inference speed
- Single-stage architecture fits in 8 GB VRAM alongside SAM2 and calibration
- Ultralytics provides production-grade training pipelines, augmentation, and ONNX/TensorRT export out of the box
- `s` (small) variant: best accuracy/VRAM tradeoff on RTX 4060 — `n` underfits on complex microscopy, `m` requires too much VRAM when SAM2 is co-loaded

**Why tiled inference (1280px tiles, 20% overlap):**
Trichomes in 4K microscopy images are tiny relative to image size. Training at 640px means small objects get lost. Tiled inference processes each 1280px crop independently and merges results with WBF (Weighted Boxes Fusion) — this recovers detection of small/edge trichomes that full-image inference misses.

---

### SAM2-tiny — Instance Segmentation

**Why SAM2 over Mask-RCNN / instance-trained YOLO:**
SAM2 (Segment Anything Model 2) is prompt-based — it takes the YOLO bounding boxes as spatial prompts and generates high-quality instance masks without needing mask annotations in the training dataset. This decouples detection labels (cheap to collect) from segmentation masks (expensive to annotate). The `tiny` variant fits in ~1.8 GB VRAM.

**Why not just YOLO segmentation (`yolo11s-seg`):**
YOLO seg produces coarser masks and requires segmentation annotations from the start. SAM2 produces significantly sharper masks for irregular biological shapes like trichome heads and requires zero mask annotation effort.

---

### TensorRT 10.x — Production Inference

**Why TensorRT over plain ONNX Runtime:**
TensorRT compiles the model into an optimized engine for the specific GPU it runs on. On RTX 4060, FP16 TRT engines are typically 2–4x faster than ONNX Runtime FP32 and 1.5–2x faster than ONNX FP16. The TRT 10.x API (`execute_async_v3`, `set_tensor_address`) is fully asynchronous and zero-copy where possible.

**Why FP16:**
FP16 (half precision) halves VRAM usage with negligible accuracy loss (< 0.5% mAP drop in our testing). On RTX 4060, Tensor Cores execute FP16 matmuls at 2x the throughput of FP32.

---

### FastAPI — Backend Framework

**Why FastAPI over Django / Flask / Express:**
- **Native async:** FastAPI is built on Starlette + asyncio. Background GPU tasks, WebSocket streams, and REST requests share one event loop without threading overhead.
- **Automatic OpenAPI docs:** All 128+ endpoints get Swagger UI and ReDoc for free — no manual documentation.
- **Pydantic validation:** Request/response types are validated, serialized, and documented from the same dataclass definitions.
- **Speed:** FastAPI benchmarks faster than Flask and Django in async I/O scenarios (the dominant pattern in this system).

**Why WebSockets for training/system monitoring:**
Training runs for minutes to hours. Polling REST every second is wasteful and laggy. WebSocket streams push metric updates at sub-second latency directly to the dashboard — no client polling needed.

---

### Next.js 14 (App Router) — Frontend

**Why Next.js over plain React / Vue / Svelte:**
- **App Router** (Next.js 14) enables React Server Components — static UI renders on the server, reducing JS bundle size and client load.
- **File-based routing** maps cleanly to the 16 distinct tool pages (inference, training, annotation, etc.)
- **TypeScript native:** The scientific tooling needs reliable type contracts between frontend and backend schemas.

**Why not a simple admin template (Grafana, Retool, etc.):**
This is a scientific instrument interface, not a dashboard. It needs custom annotation review workflows, overlaid bounding box / mask visualization, training loss curves with scientific context, and advanced configuration panels that generic dashboard tools cannot express.

---

### Label Studio + CVAT — Annotation

**Why two annotation tools:**
- **Label Studio:** Best for bounding box + classification annotation. Simple UI, fast keyboard shortcuts, built-in project management, Python SDK for programmatic import/export.
- **CVAT:** Better for polygon / instance mask annotation and multi-annotator workflows with task assignment. Useful when transitioning from box-only to mask annotation.

Both export YOLO-compatible formats. Using both gives flexibility — start with Label Studio for boxes, move to CVAT for masks when needed.

**Why not just use Roboflow / cloud annotation:**
Data sovereignty. Microscopy images may contain proprietary strain or cultivation information. All annotation stays on-premise. Label Studio and CVAT run fully air-gapped.

---

### Florence-2 / Moondream-2B / Qwen2-VL — VLM Pre-labeling

**Why Vision Language Models for pre-labeling:**
Manual annotation of microscopy images is slow and requires expertise. VLMs can generate 60–80% accurate bounding box candidates that a human then corrects in 20–30% of the time of manual annotation. This is the core speed multiplier of the labeling workflow.

**Why three VLMs:**
Different tradeoffs in quality, speed, and VRAM:

| Model | VRAM | Speed | Best for |
|---|---|---|---|
| Moondream-2B (4-bit) | 1.4 GB | Fast | High-volume first pass |
| Florence-2-large (4-bit) | 2.1 GB | Medium | Best detection accuracy |
| Qwen2-VL-7B (4-bit) | 4.8 GB | Slow | Ambiguous or complex images |

**Why 4-bit quantization:**
Full precision Florence-2 requires ~8 GB alone — the entire VRAM budget. 4-bit quantization via bitsandbytes reduces this to ~2 GB with less than 3% accuracy drop, enabling co-loading with other models.

**Why HITL (Human-In-The-Loop) is mandatory:**
VLMs hallucinate. They will confidently produce plausible but wrong bounding boxes. Sending VLM output directly to training data would introduce systematic errors that compound with each retraining cycle. All VLM outputs go to a review queue — a human approves each annotation before it becomes training data. This is a hard architectural constraint, not optional.

---

### Active Learning — Uncertainty Sampling

**Why active learning over random sampling:**
Not all images teach the model equally. Images the model is already confident on add little. Images near the decision boundary (high entropy predictions) improve calibration and generalization the most. Active learning prioritizes labeling effort toward where it matters — typically 30–50% fewer annotations needed to reach the same accuracy.

**Three strategies:**
- **Uncertainty:** High-entropy output distribution (model unsure between classes)
- **Disagreement:** Ensemble disagreement between YOLO and RTMDet predictions
- **Hybrid:** Weighted combination of both

---

### Confidence Calibration (Temperature / Platt Scaling)

**Why calibration matters:**
A YOLO model outputting confidence 0.85 does not mean 85% of such predictions are correct. Raw YOLO scores are often overconfident. Without calibration, downstream decisions (threshold tuning, active learning uncertainty estimates, report confidence intervals) are based on misleading numbers.

**Why temperature scaling:**
Temperature scaling is a single-parameter post-hoc calibration — it does not change the model, only rescales the output logits. It preserves ranking (mAP unchanged) while aligning confidence scores to actual accuracy. ECE (Expected Calibration Error) < 0.05 is the target.

---

### MLflow — Experiment Tracking

**Why MLflow over Weights & Biases:**
- **Self-hosted:** No data leaves the machine. Experiment data stays on-premise.
- **No account required:** W&B requires cloud sign-up even for local use in some modes.
- **ONNX/PyTorch model registry built in:** MLflow's model registry integrates directly with training output.
- W&B is supported too via `EXPERIMENT_TRACKER=wandb` — the choice is configurable.

---

### Docker + Nginx — Deployment

**Why Docker Compose over bare-metal / systemd:**
Docker Compose captures the entire dependency graph (nginx, backend, frontend, MLflow, Label Studio, CVAT, PostgreSQL) in version-controlled YAML. Any machine with Docker can reproduce the full stack in one command. No "works on my machine" problems.

**Why nginx as reverse proxy:**
Single entry point on port 3001. Nginx routes `/api/v1/` to backend, `/mlflow/` to MLflow, `/annotation/` to Label Studio, `/cvat/` to CVAT, and `/` to the Next.js frontend — all behind one port with TLS termination possible. This is production architecture, not a development shortcut.

**Default: localhost only — no public exposure out of the box.**
By default, nginx binds to `127.0.0.1:3001` — only accessible from the local machine. To expose publicly, set your domain in `.env` and the Setup Wizard (Settings → Setup in the web UI) handles nginx config automatically:

```env
# .env
PUBLIC_DOMAIN=your-domain.com     # e.g. mylab.ddns.net — leave empty for localhost only
PUBLIC_PORT=3001
```

To configure nginx manually:

```nginx
# docker/nginx/nginx.conf — find this block and change server_name:

server {
    listen 80;
    server_name localhost;       # ← change to your domain, or leave as localhost
    # server_name mylab.ddns.net; # ← uncomment and set your domain for public access
    ...
}
```

To **disable** public access completely (localhost only, no DDNS):
```env
PUBLIC_DOMAIN=                    # leave empty
```
Nginx will then only respond to `localhost` / `127.0.0.1` — safe for local-only research setups.

> 💡 The Setup Wizard in the web UI (`Settings → First-Time Setup`) configures all of this interactively without editing config files manually.

**Why separate docker-compose files for training:**
The GPU training container needs `runtime: nvidia`, elevated memory limits, and different restart policies. Keeping it separate means the core stack (inference + annotation) can run on machines without NVIDIA Docker runtime.

---

### SQLite → PostgreSQL

**Why SQLite in development:**
Zero configuration. No separate process. Works identically on any OS. For a single-node research platform with one user, SQLite is entirely sufficient — it handles thousands of queries per second.

**Why PostgreSQL for production:**
Multi-user annotation workflows, concurrent API requests, and production annotation databases (Label Studio + CVAT both use PostgreSQL internally) benefit from proper ACID transactions and concurrent write support. Migration is one config line: `DATABASE_URL=postgresql://...`

---

### Domain-Driven Design (DDD)

**Why DDD over flat module structure:**
Each scientific domain (detection, segmentation, maturity, morphology, measurement) has its own distinct rules, entities, and invariants. DDD separates these cleanly:

- `domain/` — pure scientific logic, no I/O, fully testable in isolation
- `application/` — orchestrates domain objects into pipelines
- `infrastructure/` — swappable backends (YOLO → RTMDet, SAM2 → MobileSAM, etc.)
- `api/` — thin FastAPI layer, no business logic

This means the maturity classifier can be tested without starting a server. The YOLO backend can be swapped for TRT without changing domain code. Scientific rules stay isolated from deployment concerns.

---

### asyncio.Semaphore(1) — GPU Guard

**Why a semaphore instead of a task queue:**
The RTX 4060 has 8 GB VRAM shared between inference, training, and VLM tasks. Two simultaneous GPU tasks will OOM-kill each other. A semaphore with depth 1 ensures exactly one GPU task runs at any moment. Requests that arrive during GPU use receive HTTP 429 immediately (fail-fast) rather than queuing indefinitely. This is intentional — the client (frontend) handles retry.

---

### AGPL-3.0 License

**Why AGPL over MIT or GPL:**
MIT and GPL have a "SaaS loophole" — a company can run the software as a cloud service without releasing modifications. AGPL closes this: anyone who deploys CTIP as a network service (API, web app) must release their modifications under the same license. This ensures the scientific community benefits from all improvements, even those made in commercial deployments.

---

---

## 🇩🇪 Warum diese Technologien?

Jede Technologieentscheidung in CTIP wurde bewusst getroffen. Dieser Abschnitt erklärt was wir verwenden und warum wir es gegenüber Alternativen bevorzugt haben.

---

### Python 3.12

**Warum:** Python 3.12 bringt signifikante Performanceverbesserungen (5–15% schneller als 3.10 bei CPU-lastigen wissenschaftlichen Workloads) und bessere Fehlermeldungen. Das gesamte wissenschaftliche ML-Ökosystem (PyTorch, NumPy, OpenCV) unterstützt es vollständig. Als Paketmanager verwenden wir `uv` — 10–100x schneller als pip bei der Abhängigkeitsauflösung.

---

### PyTorch 2.x + CUDA 12.x

**Warum PyTorch und nicht TensorFlow:** PyTorch dominiert seit 2020 Forschung und Produktion im Bereich CV. Das gesamte Ökosystem — Ultralytics, SAM2, Hugging Face Transformers — ist PyTorch-nativ. Dynamische Berechnungsgraphen erleichtern das Debugging erheblich.

**Warum CUDA 12.x:** Notwendig für TensorRT 10.x und die neuesten PyTorch CUDA-Kernel. `expandable_segments:True` in `PYTORCH_CUDA_ALLOC_CONF` reduziert Speicherfragmentierung auf 8-GB-Karten — entscheidend für unser Hardwareziel.

---

### YOLO v11s — Erkennungs-Backbone

**Warum YOLO und nicht Faster-RCNN / DETR:**
- YOLO v11s erreicht vergleichbares mAP wie schwerere Zweistufendetektoren bei 3–5x höherer Inferenzgeschwindigkeit
- Single-Stage-Architektur passt in 8 GB VRAM neben SAM2 und Kalibrierung
- Ultralytics bietet produktionsreife Trainingspipelines, Augmentierung und ONNX/TensorRT-Export
- Variante `s` (small): bestes Genauigkeits-/VRAM-Verhältnis auf RTX 4060

**Warum Tiled Inference (1280px Kacheln, 20% Überlappung):**
Trichome in 4K-Mikroskopiebildern sind winzig relativ zur Bildgröße. Tiled Inference verarbeitet jeden 1280px-Ausschnitt unabhängig und führt Ergebnisse mit WBF (Weighted Boxes Fusion) zusammen — das stellt Erkennung kleiner Trichome am Bildrand sicher.

---

### SAM2-tiny — Instanz-Segmentierung

**Warum SAM2 und nicht Mask-RCNN:**
SAM2 ist Prompt-basiert — es nimmt die YOLO-Bounding-Boxes als räumliche Prompts und erzeugt hochwertige Instanzmasken ohne Maskenannotierungen im Trainingsdatensatz. Dies entkoppelt Erkennungslabels (günstig zu sammeln) von Segmentierungsmasken (teuer zu annotieren). Die `tiny`-Variante benötigt nur ~1,8 GB VRAM.

---

### TensorRT 10.x — Produktions-Inferenz

**Warum TensorRT und nicht plain ONNX Runtime:**
TensorRT kompiliert das Modell in eine für die spezifische GPU optimierte Engine. Auf RTX 4060 sind FP16-TRT-Engines typischerweise 2–4x schneller als ONNX Runtime FP32.

**Warum FP16:**
FP16 halbiert den VRAM-Verbrauch bei vernachlässigbarem Genauigkeitsverlust (< 0,5% mAP-Rückgang). Auf RTX 4060 führen Tensor Cores FP16-Matmuls mit doppeltem Durchsatz gegenüber FP32 aus.

---

### FastAPI — Backend-Framework

**Warum FastAPI und nicht Django / Flask:**
- **Native async:** Gebaut auf Starlette + asyncio. GPU-Tasks, WebSocket-Streams und REST-Anfragen teilen sich einen Event-Loop.
- **Automatische OpenAPI-Docs:** Alle 128+ Endpunkte erhalten Swagger UI und ReDoc kostenlos.
- **Pydantic-Validierung:** Request/Response-Typen werden aus denselben Dataclass-Definitionen validiert, serialisiert und dokumentiert.

**Warum WebSockets für Training/System-Monitoring:**
Training läuft Minuten bis Stunden. REST-Polling jede Sekunde ist verschwenderisch und träge. WebSocket-Streams pushen Metrik-Updates mit Sub-Sekunden-Latenz direkt ins Dashboard.

---

### Next.js 14 (App Router) — Frontend

**Warum Next.js und nicht plain React / Vue:**
- App Router ermöglicht React Server Components — reduziert JS-Bundle-Größe
- Dateibasiertes Routing passt zu den 16 verschiedenen Tool-Seiten
- TypeScript-nativ: Wissenschaftliches Tooling braucht zuverlässige Typverträge

**Warum kein Admin-Template (Grafana, Retool):**
Das ist eine wissenschaftliche Instrumenten-Oberfläche. Sie braucht benutzerdefinierte Annotierungs-Review-Workflows, überlagerte Bounding-Box-Visualisierung, Trainingsverlust-Kurven mit wissenschaftlichem Kontext — das können generische Dashboard-Tools nicht ausdrücken.

---

### Label Studio + CVAT — Annotierung

**Warum zwei Annotierungstools:**
- **Label Studio:** Am besten für Bounding-Box-Annotation. Einfache UI, schnelle Tastenkürzel, Python SDK.
- **CVAT:** Besser für Polygon/Instanzmaske-Annotation und Multi-Annotator-Workflows mit Aufgabenzuweisung.

**Warum nicht Roboflow / Cloud-Annotierung:**
Datensouveränität. Mikroskopiebilder können proprietäre Zucht- oder Anbauinformationen enthalten. Alle Annotierungen bleiben On-Premise.

---

### Florence-2 / Moondream-2B / Qwen2-VL — VLM-Vorlabeling

**Warum VLMs für Vorlabeling:**
Manuelle Annotation von Mikroskopiebildern ist langsam und erfordert Expertise. VLMs können 60–80% genaue Bounding-Box-Kandidaten generieren, die ein Mensch dann in 20–30% der Zeit manueller Annotation korrigiert.

**Warum drei VLMs:**
Unterschiedliche Kompromisse bei Qualität, Geschwindigkeit und VRAM (siehe englische Tabelle oben).

**Warum HITL (Mensch-in-Loop) verpflichtend ist:**
VLMs halluzinieren. Sie produzieren selbstbewusst plausible aber falsche Bounding Boxes. VLM-Ausgaben direkt in Trainingsdaten zu schreiben würde systematische Fehler einführen, die sich mit jedem Retraining-Zyklus verstärken. Alle VLM-Ausgaben gehen in eine Review-Queue — ein Mensch bestätigt jede Annotierung bevor sie Trainingsdaten wird. Das ist eine harte Architekturinvariante.

---

### Aktives Lernen — Unsicherheits-Sampling

**Warum aktives Lernen statt zufälliger Auswahl:**
Nicht alle Bilder trainieren das Modell gleich stark. Bilder nahe der Entscheidungsgrenze (hohe Entropie-Vorhersagen) verbessern Kalibrierung und Generalisierung am meisten. Aktives Lernen priorisiert Annotierungsaufwand dort wo er am wichtigsten ist — typischerweise 30–50% weniger Annotierungen nötig.

---

### Konfidenz-Kalibrierung (Temperature / Platt Scaling)

**Warum Kalibrierung wichtig ist:**
Ein YOLO-Modell mit Konfidenz 0,85 bedeutet nicht, dass 85% solcher Vorhersagen korrekt sind. Roh-YOLO-Scores sind oft überkonfident. Ohne Kalibrierung sind Schwellenwert-Entscheidungen, Unsicherheitsschätzungen und Konfidenzintervalle in Berichten irreführend.

**Warum Temperature Scaling:**
Einzelner Post-hoc-Parameter — verändert das Modell nicht, skaliert nur die Ausgabe-Logits neu. Bewahrt das Ranking (mAP unverändert) bei verbesserter Konfidenzgenauigkeit. Ziel: ECE < 0,05.

---

### MLflow — Experiment-Tracking

**Warum MLflow und nicht Weights & Biases:**
- **Self-hosted:** Keine Daten verlassen die Maschine
- **Kein Account erforderlich:** W&B erfordert Cloud-Anmeldung
- **ONNX/PyTorch Model Registry integriert**
- W&B wird via `EXPERIMENT_TRACKER=wandb` ebenfalls unterstützt

---

### Docker + Nginx — Deployment

**Warum Docker Compose:**
Erfasst den gesamten Abhängigkeitsgraph in versioniertem YAML. Jede Maschine mit Docker kann den vollständigen Stack mit einem Befehl reproduzieren.

**Warum Nginx als Reverse Proxy:**
Einzelner Einstiegspunkt auf Port 3001. Nginx routet `/api/v1/` zum Backend, `/mlflow/` zu MLflow, `/annotation/` zu Label Studio — alles hinter einem Port mit möglicher TLS-Terminierung.

---

### Domain-Driven Design (DDD)

**Warum DDD statt flacher Modulstruktur:**
Jede wissenschaftliche Domäne hat eigene Regeln und Invarianten. DDD trennt diese sauber:
- `domain/` — reine wissenschaftliche Logik, kein I/O, vollständig isoliert testbar
- `infrastructure/` — austauschbare Backends (YOLO → RTMDet, SAM2 → MobileSAM)

Der Reifegrad-Klassifikator kann ohne gestarteten Server getestet werden. Das YOLO-Backend kann gegen TRT ausgetauscht werden ohne Domain-Code zu ändern.

---

### asyncio.Semaphore(1) — GPU-Guard

**Warum Semaphore statt Task-Queue:**
Die RTX 4060 hat 8 GB VRAM geteilt zwischen Inferenz, Training und VLM-Tasks. Zwei gleichzeitige GPU-Tasks würden sich gegenseitig mit OOM-Kill beenden. Semaphore mit Tiefe 1 stellt sicher, dass genau ein GPU-Task gleichzeitig läuft. Anfragen während GPU-Nutzung erhalten sofort HTTP 429 (Fail-Fast).

---

### AGPL-3.0-Lizenz

**Warum AGPL und nicht MIT oder GPL:**
MIT und GPL haben eine "SaaS-Lücke" — ein Unternehmen kann die Software als Cloud-Service betreiben ohne Änderungen zu veröffentlichen. AGPL schließt das: Jeder der CTIP als Netzwerkdienst betreibt muss seine Änderungen unter derselben Lizenz veröffentlichen.

---

---

## 🇪🇸 ¿Por qué estas tecnologías?

Cada decisión tecnológica en CTIP fue tomada deliberadamente. Esta sección explica qué usamos y por qué lo elegimos sobre las alternativas.

---

### Python 3.12

**Por qué:** Python 3.12 trae mejoras de rendimiento significativas (5–15% más rápido que 3.10 en cargas de trabajo científicas) y mejores mensajes de error. Todo el ecosistema ML científico (PyTorch, NumPy, OpenCV) lo soporta completamente. Usamos `uv` como gestor de paquetes — 10–100x más rápido que pip en resolución de dependencias.

---

### PyTorch 2.x + CUDA 12.x

**Por qué PyTorch y no TensorFlow:** PyTorch domina la investigación y producción en CV desde 2020. Todo el ecosistema — Ultralytics, SAM2, Hugging Face Transformers — es nativo de PyTorch. Los grafos de computación dinámicos facilitan la depuración.

**Por qué CUDA 12.x:** Necesario para TensorRT 10.x y los últimos kernels CUDA de PyTorch. `expandable_segments:True` reduce la fragmentación de memoria en tarjetas de 8 GB VRAM — crítico para nuestro hardware objetivo.

---

### YOLO v11s — Backbone de Detección

**Por qué YOLO y no Faster-RCNN / DETR:**
- YOLO v11s alcanza mAP comparable a detectores de dos etapas más pesados a 3–5x mayor velocidad de inferencia
- Arquitectura de una sola etapa cabe en 8 GB VRAM junto con SAM2 y calibración
- Ultralytics proporciona pipelines de entrenamiento de calidad productiva, augmentación y exportación ONNX/TensorRT

**Por qué inferencia tiled (tiles 1280px, 20% solapamiento):**
Los tricomas en imágenes de microscopía 4K son diminutos en relación al tamaño de la imagen. La inferencia tiled procesa cada recorte de 1280px independientemente y fusiona resultados con WBF — recupera la detección de tricomas pequeños y en bordes.

---

### SAM2-tiny — Segmentación de Instancias

**Por qué SAM2 y no Mask-RCNN:**
SAM2 es basado en prompts — toma las bounding boxes de YOLO como prompts espaciales y genera máscaras de instancia de alta calidad sin necesitar anotaciones de máscara en el dataset de entrenamiento. Esto desacopla las etiquetas de detección (baratas de recolectar) de las máscaras de segmentación (caras de anotar). La variante `tiny` usa solo ~1.8 GB VRAM.

---

### TensorRT 10.x — Inferencia en Producción

**Por qué TensorRT y no ONNX Runtime:**
TensorRT compila el modelo en un engine optimizado para la GPU específica. En RTX 4060, los engines TRT FP16 son típicamente 2–4x más rápidos que ONNX Runtime FP32.

**Por qué FP16:**
FP16 reduce a la mitad el uso de VRAM con pérdida de precisión despreciable (< 0.5% de caída en mAP). En RTX 4060, los Tensor Cores ejecutan matmuls FP16 al doble de rendimiento que FP32.

---

### FastAPI — Framework Backend

**Por qué FastAPI y no Django / Flask:**
- **Async nativo:** Construido sobre Starlette + asyncio. Tareas GPU, streams WebSocket y peticiones REST comparten un event loop.
- **Docs OpenAPI automáticas:** Todos los 128+ endpoints obtienen Swagger UI y ReDoc gratis.
- **Validación Pydantic:** Los tipos request/response se validan, serializan y documentan desde las mismas definiciones de dataclass.

**Por qué WebSockets para monitorización:**
El entrenamiento dura minutos a horas. El polling REST cada segundo es ineficiente. Los streams WebSocket empujan actualizaciones de métricas con latencia sub-segundo directamente al dashboard.

---

### Next.js 14 (App Router) — Frontend

**Por qué Next.js y no React puro / Vue:**
- App Router permite React Server Components — reduce el tamaño del bundle JS
- Enrutamiento basado en archivos se mapea limpiamente a las 16 páginas de herramientas
- TypeScript nativo: las herramientas científicas necesitan contratos de tipo fiables

**Por qué no una plantilla admin (Grafana, Retool):**
Esta es una interfaz de instrumento científico. Necesita flujos de trabajo de revisión de anotaciones personalizados, visualización de bounding boxes superpuestas, curvas de pérdida de entrenamiento con contexto científico — herramientas de dashboard genéricas no pueden expresar esto.

---

### Label Studio + CVAT — Anotación

**Por qué dos herramientas de anotación:**
- **Label Studio:** Mejor para anotación de bounding boxes. UI simple, atajos de teclado rápidos, SDK Python.
- **CVAT:** Mejor para anotación de polígonos/máscaras y flujos multi-anotador con asignación de tareas.

**Por qué no Roboflow / anotación en la nube:**
Soberanía de datos. Las imágenes de microscopía pueden contener información propietaria. Toda la anotación permanece on-premise.

---

### Florence-2 / Moondream-2B / Qwen2-VL — Pre-etiquetado VLM

**Por qué VLMs para pre-etiquetado:**
La anotación manual de imágenes de microscopía es lenta y requiere experiencia. Los VLMs pueden generar candidatos de bounding box con 60–80% de precisión que un humano corrige en 20–30% del tiempo de anotación manual.

**Por qué tres VLMs:**
Diferentes compromisos en calidad, velocidad y VRAM (ver tabla en sección inglesa).

**Por qué HITL (Humano en el Loop) es obligatorio:**
Los VLMs alucinan. Enviar salidas VLM directamente a datos de entrenamiento introduciría errores sistemáticos que se agravan con cada ciclo de reentrenamiento. Todas las salidas VLM van a una cola de revisión — un humano aprueba cada anotación antes de que se convierta en dato de entrenamiento. Esta es una invariante arquitectónica estricta.

---

### Aprendizaje Activo — Muestreo por Incertidumbre

**Por qué aprendizaje activo en lugar de muestreo aleatorio:**
No todas las imágenes entrenan al modelo igualmente. Las imágenes cerca del límite de decisión (predicciones de alta entropía) mejoran más la calibración y generalización. El aprendizaje activo prioriza el esfuerzo de etiquetado donde más importa — típicamente 30–50% menos anotaciones necesarias.

---

### Calibración de Confianza (Temperature / Platt Scaling)

**Por qué la calibración importa:**
Un modelo YOLO con confianza 0.85 no significa que el 85% de esas predicciones sean correctas. Las puntuaciones brutas de YOLO suelen ser demasiado confiadas. Sin calibración, las decisiones de umbral y los intervalos de confianza en informes son engañosos.

**Por qué temperature scaling:**
Un único parámetro post-hoc — no cambia el modelo, solo reescala los logits de salida. Preserva el ranking (mAP sin cambios). Objetivo: ECE < 0.05.

---

### MLflow — Seguimiento de Experimentos

**Por qué MLflow y no Weights & Biases:**
- **Auto-hospedado:** Ningún dato sale de la máquina
- **Sin cuenta requerida:** W&B requiere registro en la nube
- **Registro de modelos ONNX/PyTorch integrado**
- W&B también soportado via `EXPERIMENT_TRACKER=wandb`

---

### Docker + Nginx — Despliegue

**Por qué Docker Compose:**
Captura todo el grafo de dependencias en YAML versionado. Cualquier máquina con Docker puede reproducir el stack completo con un comando.

**Por qué Nginx como proxy inverso:**
Punto de entrada único en el puerto 3001. Nginx enruta `/api/v1/` al backend, `/mlflow/` a MLflow, `/annotation/` a Label Studio — todo detrás de un puerto con posible terminación TLS.

---

### Domain-Driven Design (DDD)

**Por qué DDD y no estructura plana de módulos:**
Cada dominio científico tiene sus propias reglas e invariantes. DDD las separa limpiamente:
- `domain/` — lógica científica pura, sin I/O, completamente testable en aislamiento
- `infrastructure/` — backends intercambiables (YOLO → RTMDet, SAM2 → MobileSAM)

El clasificador de madurez puede testearse sin iniciar un servidor. El backend YOLO puede cambiarse por TRT sin modificar código de dominio.

---

### asyncio.Semaphore(1) — GPU Guard

**Por qué un semáforo y no una cola de tareas:**
La RTX 4060 tiene 8 GB VRAM compartidos entre inferencia, entrenamiento y tareas VLM. Dos tareas GPU simultáneas se matarían mutuamente con OOM-kill. El semáforo con profundidad 1 garantiza exactamente una tarea GPU a la vez. Las peticiones durante el uso de GPU reciben HTTP 429 inmediatamente (fail-fast).

---

### Licencia AGPL-3.0

**Por qué AGPL y no MIT o GPL:**
MIT y GPL tienen un "agujero SaaS" — una empresa puede ejecutar el software como servicio en la nube sin publicar modificaciones. AGPL lo cierra: cualquiera que despliegue CTIP como servicio de red debe publicar sus modificaciones bajo la misma licencia.

---

## License

AGPL-3.0 — see [LICENSE](LICENSE)

---

*This platform is for scientific optical analysis only. No claims are made about cannabinoid concentrations, potency, or chemical composition. All maturity assessments are based purely on optical observation under digital microscopy.*
