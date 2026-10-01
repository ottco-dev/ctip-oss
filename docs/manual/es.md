# Manual de CTIP (Español)

> [English](en.md) · [Deutsch](de.md) · [Español](es.md) · [Technology stack](tech-stack.md) · [Back to README](../../README.md)

## Tabla de Contenidos

1. [¿Qué es esto?](#es-1)
2. [Requisitos de Hardware](#es-2)
3. [Instalación](#es-3)
4. [Primeros Pasos](#es-4)
5. [Recolección de Datos — Consejos para Imágenes](#es-5)
6. [Flujo de Trabajo de Etiquetado](#es-6)
7. [Flujo de Trabajo de Entrenamiento](#es-7)
8. [Verificación y Benchmarking](#es-8)
9. [Ciclo de Mejora](#es-9)
10. [Despliegue con Docker](#es-10)
11. [Todas las URLs y Páginas](#es-11)
12. [Referencia de API](#es-12)
13. [Referencia de CLI](#es-13)
14. [Configuración](#es-14)
15. [Arquitectura](#es-15)
16. [Metodología Científica](#es-16)
17. [Pruebas](#es-17)

---

## ES 1. ¿Qué es esto? {#es-1}

CTIP es una **plataforma de investigación completa y de calidad productiva** para análisis automatizado de tricomas de *Cannabis sativa L.* bajo microscopía digital. No es una demo — es un sistema completo y en funcionamiento para trabajo científico real.

### Capacidades

| Capacidad | Método | Objetivo |
|---|---|---|
| Detección de Tricomas | YOLO v11s + ensemble RTMDet | mAP50 > 0.88 |
| Segmentación de Instancias | SAM2-tiny + refinamiento de máscara | IoU > 0.82 |
| Clasificación de Madurez | HSV + LAB + Textura (LBP/GLCM/Gabor) | F1 > 0.85 |
| Tipificación Morfológica | Geométrico + CNN (pedunculado/sésil/bulboso) | Precisión > 0.90 |
| Medición de Tamaño | Conversión calibrada px a µm | ±5% error |
| Evaluación de Enfoque | Laplacian + Tenengrad + FFT | — |
| Análisis de Video | Ranking de calidad + deduplicación temporal | — |
| Pre-etiquetado VLM | Moondream-2B / Florence-2 / Qwen2-VL (4-bit) | Humano en el loop |
| Aprendizaje Activo | Muestreo por incertidumbre + desacuerdo | — |
| Inferencia TensorRT | Engine FP16, API async v3 | Optimizado para RTX 4060 |

### Lo que NO hace

- Sin predicciones de concentración de THC/cannabinoides (solo madurez óptica)
- Sin pseudociencia
- Las salidas VLM nunca van directamente a datos de entrenamiento (HITL obligatorio)

---

## ES 2. Requisitos de Hardware {#es-2}

| Componente | Mínimo | Recomendado |
|---|---|---|
| GPU | NVIDIA GTX 1080 (8 GB VRAM) | RTX 4060 / 3080 (8+ GB) |
| CPU | 6 núcleos modernos | i5-13400F o mejor |
| RAM | 16 GB | 32 GB |
| Almacenamiento | 50 GB SSD | 500 GB NVMe |
| CUDA | 11.8+ | 12.6 |

### Presupuesto VRAM (RTX 4060, 8 GB)

| Componente | VRAM |
|---|---|
| Inferencia YOLO v11s | ~0.9 GB |
| SAM2-tiny | ~1.8 GB |
| Florence-2 (4-bit) | ~2.1 GB |
| Moondream-2B (4-bit) | ~1.4 GB |
| Qwen2-VL-7B (4-bit) | ~4.8 GB |
| Entrenamiento YOLO v11s (bs=8) | ~5.5 GB |

> Solo **una tarea GPU a la vez** — aplicado mediante asyncio.Semaphore(1). Intencional para tarjetas de 8 GB.

---

## ES 3. Instalación {#es-3}

### 3.1 Prerequisitos

```bash
# Ubuntu 22.04 / 24.04
sudo apt update && sudo apt install -y \
    git curl wget build-essential \
    python3.12 python3.12-venv python3.12-dev \
    ffmpeg libgl1 libglib2.0-0 libsm6 libxext6

# Instalar uv (gestor de paquetes Python rápido)
curl -LsSf https://astral.sh/uv/install.sh | sh && source ~/.bashrc

# Node.js 20 (para el frontend)
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt install -y nodejs

# Verificar CUDA
nvcc --version && nvidia-smi
```

### 3.2 Clonar e Instalar

```bash
git clone https://github.com/tuusuario/trichome-analysis.git
cd trichome-analysis

python3.12 -m venv .venv && source .venv/bin/activate

uv pip install -e ".[dev]"       # núcleo + dev
uv pip install -e ".[vlm]"       # + modelos VLM
uv pip install -e ".[sam]"       # + segmentación SAM2
uv pip install -e ".[all]"       # todo
```

### 3.3 TensorRT (opcional, inferencia en producción)

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

### 3.5 Configuración del Entorno

```bash
cp .env.example .env
```

> **Consejo:** Usa el **Asistente de Configuración** integrado en lugar de editar `.env` manualmente — te guía por cada ajuste de forma interactiva (→ §4.1).

Para configuración manual, las variables clave son:

```env
DATA_ROOT=/mnt/data/trichome          # o ./data para desarrollo local
MODELS_ROOT=/mnt/models/trichome
CUDA_VISIBLE_DEVICES=0
VRAM_LIMIT_GB=8.0
MLFLOW_TRACKING_URI=http://localhost:3004
EXPERIMENT_TRACKER=mlflow
LABEL_STUDIO_URL=http://localhost:3005
LABEL_STUDIO_API_KEY=tu_clave_aqui
```

---

## ES 4. Primeros Pasos {#es-4}

### 4.1 Iniciar en Modo Desarrollo y Configuración Inicial

```bash
# Terminal 1 — Backend API
source .venv/bin/activate
uvicorn backend.main:app --reload --port 8000

# Terminal 2 — Frontend
cd frontend && npm run dev
```

Abrir **http://localhost:3000** — el **Asistente de Configuración se inicia automáticamente** en el primer arranque (cuando no hay `.env` configurado).

El asistente recorre 7 pasos:

| Paso | Configura |
|---|---|
| 🌐 Red | Dominio público vs. solo localhost, puerto nginx |
| ⚙️ Hardware | Dispositivo CUDA, presupuesto VRAM |
| 💾 Almacenamiento | Directorio raíz, modelos, salidas |
| 🔌 Servicios | API Key de Label Studio, URI de MLflow, W&B (opcional) |
| 🔒 Seguridad | Clave secreta (generador automático), token API |
| ✅ Revisión | Resumen de todos los ajustes antes de guardar |
| 🎉 Listo | Escribe `.env`, muestra comando de reinicio Docker |

Al terminar, `.env` se escribe automáticamente — sin edición manual.
Volver a ejecutar en cualquier momento desde la barra lateral: **Configuración Inicial**.

- Docs API: http://localhost:8000/docs

### 4.2 Verificar el Sistema

```bash
pytest tests/ -v --tb=short
# Esperado: 1680 passed, 4 skipped

python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
curl http://localhost:8000/api/v1/system/health | python -m json.tool
```

### 4.3 Primera Detección

```bash
# CLI
trichome detect --input /ruta/a/imagen.jpg --tiled --tile-size 1280

# API
curl -X POST http://localhost:8000/api/v1/detection/infer \
  -F "file=@imagen.jpg" -F "confidence_threshold=0.25" | python -m json.tool

# Frontend: http://localhost:3000/inference — arrastrar imagen
```

---

## ES 5. Recolección de Datos — Consejos para Imágenes {#es-5}

Los buenos datos son el factor más importante. Lo que realmente importa en microscopía de tricomas:

### 5.1 Equipamiento

| Configuración | Notas |
|---|---|
| Microscopio digital | 40x–200x de aumento. Los microscopios USB (Andonstar, Celestron, Jiusion) son válidos para empezar. |
| Teléfono + lente clip | Aceptable para tricomas grandes, malo para bulbosos/pequeños. |
| Microscopio estéreo | Mejor claridad óptica, más difícil de digitalizar consistentemente. |

### 5.2 Protocolo de Captura (Crítico)

```
HACER:
  - Magnificación consistente en cada sesión (ej. siempre 100x)
  - Capturar en RAW o máxima calidad JPEG
  - Imagen de calibración con micrómetro ocular (ej. 1 mm)
    Permite calibración px a µm
  - Mínimo 1920x1080, idealmente 4K
  - Misma posición de fuente de luz siempre
  - Nombre de archivo con metadatos:
    microscopio01_100x_20260101_muestra42_001.jpg
  - Todos los tipos de tricomas en una sesión
  - Parches de fondo vacíos como ejemplos negativos

NO HACER:
  - Mezclar magnificaciones sin registrarlas (arruina la calibración)
  - Usar exposición automática (brillo inconsistente)
  - Fotografiar solo tricomas perfectos (incluir parciales, solapados, en bordes)
  - Usar imágenes comprimidas de redes sociales
```

### 5.3 Cobertura de Etapas de Madurez

| Etapa | Visual | Objetivo % del dataset |
|---|---|---|
| Clara | Vítrea, completamente transparente | ~25% |
| Nublada | Blanca/lechosa, opaca | ~35% |
| Ámbar | Dorado-naranja, degradado | ~25% |
| Mixta | Especímenes en transición | ~15% |

### 5.4 Organización de Datos

```
data/
├── raw/                    # Imágenes originales sin modificar
│   ├── sesion_20260101/
│   └── sesion_20260115/
├── calibration/            # Imágenes de micrómetro por microscopio+magnificación
├── annotated/              # Tras etiquetar (Label Studio exporta aquí)
│   ├── images/
│   └── labels/             # Archivos .txt formato YOLO
└── splits/                 # train / val / test — NUNCA mezclar sesiones
    ├── train/
    ├── val/
    └── test/
```

> **Nunca** poner imágenes de la misma sesión en train Y val/test — eso es fuga de datos.
> Dividir siempre por **sesión**, no por imagen.

### 5.5 Filtro de Calidad de Enfoque

```bash
trichome video score image.jpg      # focus, exposure and noise score (also shown per image in the UI)
               --output data/filtrado/ \
               --min-sharpness 80.0 --copy-passing

# O via API
curl -X POST http://localhost:8000/api/v1/focus/score -F "file=@imagen.jpg"
```

### 5.6 Tamaño Mínimo del Dataset

| Fase | Imágenes | Anotaciones (cajas) |
|---|---|---|
| Primer modelo funcional | 150–300 | 2.000–5.000 |
| Buena generalización | 500–1.000 | 10.000–25.000 |
| Listo para producción | 2.000+ | 50.000+ |

Empezar pequeño, entrenar rápido, identificar casos de fallo, recolectar imágenes específicas.

---

## ES 6. Flujo de Trabajo de Etiquetado {#es-6}

### 6.1 Iniciar Label Studio

```bash
# Docker (recomendado)
cd docker && docker compose --profile annotation up -d label-studio
# Independiente
pip install label-studio && label-studio start --port 3005
```

Acceso: http://localhost:3005

### 6.2 Crear Proyecto

1. **Create Project** — asignar nombre (ej. "Tricomas Sesión 20260101")
2. **Labeling Setup** → Object Detection with Bounding Boxes
3. Agregar etiquetas (ortografía exacta requerida para exportación YOLO):

```
capitate-stalked    #FF4444  (rojo)
capitate-sessile    #44FF44  (verde)
bulbous             #4444FF  (azul)
non-glandular       #FFAA00  (naranja)
```

4. **Importar imágenes**: Settings → Cloud Storage → Add Source Storage → Local Files → ruta a `data/raw/sesion_XXXXXXXX/`

### 6.3 Pre-etiquetado VLM (3–5x más rápido)

```bash
curl -X POST http://localhost:8000/api/v1/vlm/label \
  -H "Content-Type: application/json" \
  -d '{
    "image_paths": ["data/raw/sesion_20260101/img001.jpg"],
    "model": "florence2",
    "confidence_threshold": 0.3
  }'
```

Las anotaciones van a una **cola de revisión** — nunca directamente a datos de entrenamiento.

| Modelo | VRAM | Velocidad | Calidad |
|---|---|---|---|
| Moondream-2B (4-bit) | ~1.4 GB | Rápido | Bueno para detección |
| Florence-2-large (4-bit) | ~2.1 GB | Medio | Mejor para escenas complejas |
| Qwen2-VL-7B (4-bit) | ~4.8 GB | Lento | Mayor calidad |

### 6.4 Estándares de Anotación

```
Reglas para dibujar cajas:
  SI: Caja ajustada alrededor de la cabeza del tricoma (no el tallo)
  SI: Incluir cabeza completa aunque esté parcialmente ocluida
  SI: Marcar tricomas en los bordes de la imagen
  SI: Etiquetar TODOS los tricomas visibles — sin saltarse ninguno
  SI: Ante duda pedunculado/sésil: buscar cuello visible

  NO: Cajas alrededor de tallos desnudos (sin cabeza)
  NO: Etiquetar residuos o artefactos
  NO: Saltarse tricomas borrosos si son identificables
```

### 6.5 Exportar y Verificar Anotaciones

```bash
# Label Studio UI: Project → Export → YOLO format → Download

# Via API
curl -X POST http://localhost:8000/api/v1/annotation/export \
  -H "Content-Type: application/json" \
  -d '{"project_id": 1, "format": "yolo", "output_dir": "data/annotated/sesion_20260101"}'

# Verificar calidad
curl -X POST http://localhost:8000/api/v1/annotation/stats \
  -H "Content-Type: application/json" \
  -d '{"annotation_dir": "data/annotated/sesion_20260101"}'
```

---

## ES 7. Flujo de Trabajo de Entrenamiento {#es-7}

### 7.1 Preparar División del Dataset

```bash
# Export a Label Studio project as a YOLO dataset. Whole imaging sessions go to one split each
# (task field data.session, else the image folder, else the file-name series), so near-identical
# frames never leak between train / val / test. The export log reports images and sessions per split.
curl -X POST http://localhost:8000/api/v1/training/prepare-ls-dataset \
  -H "Content-Type: application/json" \
  -d '{"project_id": 1, "train_ratio": 0.70, "val_ratio": 0.15, "seed": 42}'
```

### 7.2 Configurar Entrenamiento

`configs/training/yolo11s_detection.yaml`:

```yaml
model: yolo11s.pt           # Descarga automáticamente
task: detect
data: data/splits/dataset.yaml
imgsz: 1280                 # Requerido para inferencia tiled
batch: 8                    # Óptimo para RTX 4060 8GB
workers: 4
epochs: 100
patience: 20
lr0: 0.01
cos_lr: true

# Augmentación específica para microscopía
degrees: 90.0               # Rotación completa — tricomas sin orientación canónica
flipud: 0.5
fliplr: 0.5
hsv_h: 0.015                # Pequeño cambio de tono — iluminación variable
hsv_s: 0.7
mosaic: 0.3                 # Mosaico más bajo — contexto microscópico importa

device: 0
amp: true                   # Precisión mixta FP16
```

### 7.3 Iniciar Entrenamiento

```bash
# CLI
trichome train start --config configs/training/yolo11s_detection.yaml

# API (no bloqueante, progreso via WebSocket)
curl -X POST http://localhost:8000/api/v1/training/start \
  -H "Content-Type: application/json" \
  -d '{"config_path": "configs/training/yolo11s_detection.yaml"}'

# Dashboard en vivo: http://localhost:3000/training
# Stream WebSocket:  ws://localhost:8000/ws/training
# MLflow UI:         http://localhost:3004
```

### 7.4 Salida del Entrenamiento

```
runs/detect/trichome_yolo11s_20260101/
    weights/best.pt      <- usar este para inferencia
    weights/last.pt
    results.csv
    confusion_matrix.png
    PR_curve.png
    val_batch0_pred.jpg
```

---

## ES 8. Verificación y Benchmarking {#es-8}

### 8.1 Evaluar en Conjunto de Prueba

```bash
trichome benchmark detection \
  --weights runs/detect/trichome_yolo11s_20260101/weights/best.pt \
  --split test --data data/splits/dataset.yaml \
  --conf 0.25 --iou 0.5
```

Salida esperada:

```
Clase               P      R      mAP50  mAP50-95
all                 0.887  0.862  0.883  0.512
capitate-stalked    0.921  0.905  0.918  0.561
capitate-sessile    0.873  0.841  0.864  0.498
bulbous             0.841  0.812  0.832  0.445
non-glandular       0.913  0.890  0.918  0.543
```

### 8.2 Calibración de Confianza

Las puntuaciones de confianza de YOLO no están calibradas. Corregir antes del despliegue:

```bash
curl -X POST http://localhost:8000/api/v1/detection/calibrate \
  -H "Content-Type: application/json" \
  -d '{
    "weights_path": "runs/.../best.pt",
    "val_data": "data/splits/val/",
    "method": "temperature"
  }'
```

Objetivo: ECE < 0.05. Diagramas de fiabilidad generados automáticamente.

### 8.3 Construir Engine TensorRT

```bash
# Exportar YOLO a ONNX
python -c "
from ultralytics import YOLO
YOLO('runs/.../best.pt').export(format='onnx', imgsz=1280, dynamic=True, half=True)
"

# Construir engine FP16
trichome convert tensorrt runs/.../best.pt \
  --imgsz 1280 --fp16 --workspace-gb 4

# Benchmark TRT vs PyTorch
trichome benchmark inference \
  --engine models/trichome_yolo11s_fp16.engine \
  --pytorch runs/.../best.pt \
  --image data/splits/test/images/ --n 100
```

### 8.4 Benchmark Inferencia Tiled

```bash
trichome benchmark tiled \
  --weights runs/.../best.pt \
  --image data/splits/test/images/alta_resolucion_001.jpg \
  --tile-sizes 640 1280 --overlaps 0.1 0.2 0.3
```

---

## ES 9. Ciclo de Mejora {#es-9}

### 9.1 Aprendizaje Activo — Encontrar Casos Difíciles

```bash
curl -X POST http://localhost:8000/api/v1/active_learning/sample \
  -H "Content-Type: application/json" \
  -d '{"strategy": "uncertainty", "n_samples": 50, "unlabeled_dir": "data/raw/nueva_sesion/"}'
```

Etiquetar primero las imágenes devueltas — enseñan más al modelo por hora de anotación.

### 9.2 Casos de Fallo Comunes

| Fallo | Causa | Solución |
|---|---|---|
| Tricomas bulbosos no detectados | Muy pequeños en training | Recolectar primeros planos específicos |
| Falsos positivos en residuos | Se parecen a tricomas | Etiquetar como non-glandular |
| Confusión pedunculado/sésil | Tallo corto en mal ángulo | Añadir más variaciones de ángulo |
| Detección pobre en bordes | Artefactos de relleno | Aumentar solapamiento en tiled inference |
| mAP bajo con IoU alto | Cajas dibujadas muy holgadas | Reforzar protocolo de anotación |

### 9.3 Lista de Verificación Post-entrenamiento

```
  [ ] Revisar matriz de confusión — ¿qué clase se confunde con cuál?
  [ ] Examinar val_batch*_pred.jpg — ¿dónde falla visualmente el modelo?
  [ ] Aplicar muestreo activo a nuevos datos sin etiquetar
  [ ] Verificar distribución de clases (¿equilibrada?)
  [ ] Añadir imágenes específicas para clases con bajo rendimiento
  [ ] Re-verificar calidad de anotación (Cohen kappa >= 0.80)
  [ ] Sin solapamiento de sesiones en splits
  [ ] Ejecutar calibración tras cada nuevo entrenamiento
```

### 9.4 Triggers de Reentrenamiento

```bash
curl http://localhost:8000/api/v1/active_learning/trigger | python -m json.tool
```

Se activan cuando:
- 100+ nuevas imágenes anotadas desde último entrenamiento
- Incertidumbre media de predicciones recientes > 0.45
- Nueva distribución de clases diverge > 15% de la distribución de entrenamiento

---

## ES 10. Despliegue con Docker {#es-10}

### Stack Principal (nginx + backend + frontend + MLflow)

```bash
cd docker
docker compose build       # solo la primera vez
docker compose up -d
docker compose logs -f
docker compose down
```

### Con Herramientas de Anotación (Label Studio + CVAT + PostgreSQL)

```bash
docker compose --profile annotation up -d
```

### Con Stack de Entrenamiento GPU

```bash
docker compose -f docker-compose.yml -f docker-compose.training.yml up -d
docker exec trichome-backend nvidia-smi   # verificar acceso GPU
```

### Solo Inferencia (ligero, sin frontend)

```bash
docker compose -f docker-compose.inference.yml up -d
```

### Configuración del Entorno para Docker

```bash
cp .env.example .env
# Usar rutas internas del contenedor:
# DATA_ROOT=/data
# MODELS_ROOT=/models
# MLFLOW_TRACKING_URI=http://mlflow:5000   <- DNS interno Docker
```

### Volúmenes de Datos

```bash
docker volume ls | grep trichome
# trichome-models        pesos de modelos (compartidos)
# trichome-mlflow        datos de experimentos
# trichome-db            base de datos SQLite
# trichome-label-studio  datos de Label Studio
```

### Actualizar / Reconstruir

```bash
cd docker && git pull
docker compose build --no-cache && docker compose up -d
```

---

## ES 11. Todas las URLs y Páginas {#es-11}

### Modo Desarrollo (sin Docker)

| Servicio | URL | Propósito |
|---|---|---|
| Frontend | http://localhost:3000 | Interfaz web principal |
| API Swagger | http://localhost:8000/docs | Docs API interactivos |
| API ReDoc | http://localhost:8000/redoc | Referencia API |
| WS Entrenamiento | ws://localhost:8000/ws/training | Stream entrenamiento en vivo |
| WS Sistema | ws://localhost:8000/ws/system | Stats sistema/GPU |
| WS Jobs | ws://localhost:8000/ws/jobs | Estado de trabajos en segundo plano |
| WS Logs | ws://localhost:8000/ws/logs | Stream de logs en vivo |

### Modo Docker

| Servicio | URL Local | URL Pública (via nginx) |
|---|---|---|
| Gateway Nginx | http://localhost:3001 | http://your-domain.com:3001 |
| Backend API | http://localhost:3002/api/v1 | .../api/v1/ |
| Docs API | http://localhost:3002/docs | .../docs |
| Frontend | http://localhost:3003 | .../ |
| MLflow | http://localhost:3004 | .../mlflow/ |
| Label Studio | http://localhost:3005 | .../annotation/ |
| CVAT | http://localhost:3006 | .../cvat/ |

### Páginas del Frontend

| Página | Ruta | Función |
|---|---|---|
| Dashboard | / | Resumen del sistema, estado GPU, trabajos recientes |
| Inferencia | /inference | Subir imagen, ejecutar detección/segmentación |
| Datasets | /datasets | Explorar, importar, validar datasets |
| Anotación | /annotation | Revisar pre-etiquetas VLM, gestionar Label Studio |
| Label Studio | /labelstudio | Label Studio integrado |
| Entrenamiento | /training | Iniciar, monitorizar, comparar entrenamientos |
| Modelos | /models | Registro de modelos, gestión de versiones |
| Experimentos | /experiments | Comparación de experimentos MLflow |
| Morfología | /morphology | Resultados de análisis morfológico |
| Analytics | /analytics | Generar informes PDF/CSV/JSON |
| Video | /video | Pipeline de video, extracción de frames |
| Informes | /reports | Archivo de informes pasados |
| Benchmarks | /benchmarks | Historial y comparación de benchmarks |
| Sistema | /system | Stats de hardware, monitor de procesos |
| Procesos | /processes | Gestión de contenedores Docker, compose, logs en vivo |
| Configuración | /settings | Config, calibración, claves API |
| Wiki | /wiki | Documentación integrada (ES/EN/DE, 14 páginas) |

---

## ES 12. Referencia de API {#es-12}

Ruta base: `/api/v1/` — Docs completos: http://localhost:8000/docs

```bash
# Detección
POST /detection/infer               # Inferencia imagen única
POST /detection/infer/tiled         # Inferencia tiled (imágenes 4K)
POST /detection/infer/batch         # Inferencia por lotes
POST /detection/calibrate           # Calibrar puntuaciones de confianza

# Segmentación
POST /segmentation/segment          # Segmentación instancias SAM2
POST /segmentation/refine           # Refinar máscara existente

# Madurez
POST /maturity/classify             # Clasificar madurez de región
POST /maturity/classify/batch       # Clasificación por lotes
GET  /maturity/thresholds           # Umbrales actuales
PUT  /maturity/thresholds           # Actualizar umbrales

# Entrenamiento
POST /training/start                # Iniciar trabajo de entrenamiento
GET  /training/status               # Estado actual del entrenamiento
POST /training/stop                 # Detener entrenamiento
GET  /training/runs                 # Listar todos los runs
GET  /training/runs/{run_id}        # Detalles del run + métricas
POST /training/evaluate             # Evaluar en conjunto de prueba

# Pre-etiquetado VLM
POST /vlm/label                     # Pre-etiquetado VLM (a cola de revisión)
GET  /vlm/queue                     # Elementos pendientes de revisión
POST /vlm/queue/{id}/approve        # Aprobar pre-etiqueta
POST /vlm/queue/{id}/reject         # Rechazar pre-etiqueta
GET  /vlm/models                    # Modelos VLM disponibles

# Aprendizaje Activo
POST /active_learning/sample        # Muestras inciertas para etiquetar
GET  /active_learning/trigger       # Verificar trigger de reentrenamiento
POST /active_learning/priority      # Definir cola de prioridad

# Anotación
POST /annotation/export             # Exportar desde Label Studio (YOLO/COCO/CSV)
POST /annotation/stats              # Estadísticas de calidad
POST /annotation/import             # Importar lote de anotaciones
GET  /annotation/projects           # Listar proyectos de Label Studio

# Analytics e Informes
POST /analytics/report              # Generar informe (PDF/CSV/JSON)
GET  /analytics/reports             # Listar informes pasados
GET  /analytics/reports/{id}        # Descargar informe específico

# Sistema
GET  /system/health                 # Health check completo + stats GPU
GET  /system/gpu                    # VRAM, temperatura, utilización
GET  /system/version                # Versiones de todos los componentes
GET  /models                        # Registro de modelos cargados
```

---

## ES 13. Referencia de CLI {#es-13}

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

## ES 14. Configuración {#es-14}

```env
# Rutas
TRICHOME_ROOT=/path/to/trichome-analysis
DATA_ROOT=/mnt/data/trichome
MODELS_ROOT=/mnt/models/trichome

# Hardware
CUDA_VISIBLE_DEVICES=0
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
VRAM_LIMIT_GB=8.0
GPU_INFERENCE_QUEUE_DEPTH=0        # 0 = fail-fast, sin cola

# Base de datos
DATABASE_URL=sqlite:///./trichome.db

# Anotación
LABEL_STUDIO_URL=http://localhost:3005
LABEL_STUDIO_API_KEY=tu_clave
CVAT_URL=http://localhost:3006

# Seguimiento de experimentos
MLFLOW_TRACKING_URI=http://localhost:3004
EXPERIMENT_TRACKER=mlflow          # mlflow | wandb | both | none
WANDB_API_KEY=tu_clave

# Modelos VLM
FLORENCE2_MODEL_ID=microsoft/Florence-2-large
MOONDREAM_MODEL_ID=vikhyatk/moondream2
VLM_CACHE_DIR=/mnt/models/vlm_cache
```

---

## ES 15. Arquitectura {#es-15}

### Estructura de Módulos (Domain-Driven Design)

```
<modulo>/
  domain/          # Lógica de negocio pura (sin dependencias de framework)
  application/     # Orquesta objetos de dominio (pipelines)
  infrastructure/  # Backends de modelos, I/O de archivos, APIs externas
  api/             # Router FastAPI para este módulo
  schemas/         # Modelos Pydantic request/response
```

Módulos: `detection/`, `segmentation/`, `maturity/`, `morphology/`, `measurement/`,
`focus/`, `vlm_labeling/`, `annotation/`, `active_learning/`, `training/`, `inference/`,
`video_pipeline/`, `analytics/`

### Pipeline CV

```
Imagen
  -> Evaluador de enfoque (descartar frames borrosos)
  -> Inferencia tiled (YOLO v11s, tiles 1280px, 20% solapamiento)
  -> Calibración de confianza (temperature scaling)
  -> [Opcional] Ensemble RTMDet
  -> SAM2-tiny (segmentación por prompts con cajas YOLO)
  -> Refinamiento de máscaras (rellenar huecos, suavizar contornos)
  -> Clasificador morfológico (pedunculado / sésil / bulboso)
  -> Clasificador de madurez (clara -> nublada -> ámbar)
  -> Medición (px -> µm via CalibrationScale)
  -> Motor de analytics (estadísticas, generación de informes)
```

### Restricciones Arquitectónicas

- `asyncio.Semaphore(1)` — solo una tarea GPU a la vez
- Salidas VLM nunca escritas directamente en datos de entrenamiento
- `GLOBAL_SEED = 42` en todos los pipelines de entrenamiento y muestreo
- `backend/middleware/gpu_guard.py` — HTTP 429 cuando se supera el presupuesto VRAM

---

## ES 16. Metodología Científica {#es-16}

### Clasificación de Madurez

La madurez se evalúa únicamente a partir de **características ópticas** — sin afirmaciones químicas:

| Grupo de Características | Características Usadas |
|---|---|
| Color (HSV) | Tono medio, saturación, valor por región de tricoma |
| Color (LAB) | L* (luminosidad), a* (verde-rojo), b* (azul-amarillo) |
| Textura | LBP (Local Binary Patterns), GLCM (matriz de co-ocurrencia), filtros Gabor |
| Morfología | Diámetro de cabeza, longitud de tallo, circularidad |

**Limitaciones explícitas:**
- La coloración ámbar NO es un proxy directo de degradación de cannabinoides — es una observación óptica
- Las condiciones de iluminación afectan significativamente las características de color — la imagen consistente es crítica
- Este sistema **no** predice concentraciones de THC, CBD ni ningún otro cannabinoide

### Calibración

- Temperature scaling (preferido) — parámetro escalar único, preserva el ranking
- Platt scaling — ajuste sigmoide en logits de validación
- Diagramas de fiabilidad generados en cada evaluación
- ECE (Error de Calibración Esperado) < 0.05 como objetivo

### Reproducibilidad

- `GLOBAL_SEED = 42` en todos los pipelines
- Divisiones de dataset deterministas (hash de sesión)
- Todos los resultados de benchmark almacenados en `docs/progress/benchmark_history.md`

---

## ES 17. Pruebas {#es-17}

```bash
# Suite completa
pytest tests/ -v

# Rápido (sin GPU ni integración lenta)
pytest tests/ -m "not gpu and not slow and not integration" -v

# Módulo específico
pytest tests/unit/test_detection_metrics.py -v
pytest tests/unit/test_tensorrt_runner.py -v
pytest tests/unit/test_inference_tiling.py -v

# Con cobertura
pytest tests/ --cov=. --cov-report=html

# Pruebas GPU (requiere TRICHOME_ENGINE env var)
pytest tests/ -m gpu -v
```

**Estado actual: 1680 passed, 4 skipped (GPU-only + guard de reportlab)**

| Módulo | Pruebas |
|---|---|
| Métricas de detección | 45 |
| Clasificador de madurez | 38 |
| Segmentación | 41 |
| VLM schema enforcer | 63 |
| Estadísticas de anotación | 36 |
| Exportación analytics | 61 |
| TensorRT runner + builder | 35 |
| Inferencia tiled | 57 |
| Resto de módulos | 584 |
