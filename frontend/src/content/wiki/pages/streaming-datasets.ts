import type { WikiPage } from '../types';

const en = `
## Streaming Datasets

Large microscopy image collections frequently exceed available RAM. CTIP provides two memory-efficient dataset backends — **Zarr** (chunked, streamable) and **HDF5** (random-access) — that integrate directly with PyTorch DataLoaders without loading the entire dataset into memory.

---

## Format comparison

| Feature | Zarr | HDF5 |
|---|---|---|
| Primary use case | Streaming / cloud-friendly | Random-access / classic HPC |
| On-disk layout | Directory tree of chunk files | Single \`.h5\` file |
| Access pattern | Sequential or chunked streaming | Direct index lookup |
| Compression | Blosc (default), Zlib, LZ4 | GZIP, LZF |
| PyTorch DataLoader compat | Yes (\`ZarrDataset\`) | Yes (\`HDF5Dataset\`) |
| Recommended for | Incremental acquisition, S3/NFS | Fixed dataset, fast local SSD |

---

## Python usage

### ZarrDataset

\`\`\`python
from backend.training.datasets.zarr_dataset import ZarrDataset, ZarrDatasetConfig

config = ZarrDatasetConfig(
    source_root="data/datasets/v2/",
    zarr_root="data/zarr/v2.zarr",
    chunk_size=64,
    compression="blosc",
)
ZarrDataset.create_from_images(config)

dataset = ZarrDataset(config)
sample = dataset[0]   # {"image": Tensor[3,H,W], "labels": Tensor[N,5]}
\`\`\`

### HDF5Dataset

\`\`\`python
from backend.training.datasets.hdf5_dataset import HDF5Dataset, HDF5DatasetConfig

config = HDF5DatasetConfig(
    h5_path="data/hdf5/v2.h5",
    split="train",
)
dataset = HDF5Dataset(config)
sample = dataset[0]   # {"image": Tensor[3,H,W], "labels": Tensor[N,5]}
\`\`\`

Both classes are \`torch.utils.data.Dataset\` subclasses and can be wrapped in a standard \`DataLoader\`.

---

## DatasetConverter

\`DatasetConverter\` supports four conversion paths:

| Source | Target |
|---|---|
| YOLO dataset | Zarr |
| YOLO dataset | HDF5 |
| Zarr | HDF5 |
| HDF5 | Zarr |

---

## API reference

### Start a conversion task

\`\`\`bash
POST /api/v1/datasets/convert
Content-Type: application/json

{
  "source_format": "yolo",
  "target_format": "zarr",
  "source_root": "data/datasets/v2/",
  "output_path": "data/zarr/v2.zarr",
  "chunk_size": 64,
  "compression": "blosc"
}
\`\`\`

Response:
\`\`\`json
{ "task_id": "conv_xyz789", "status": "queued" }
\`\`\`

### Check conversion status

\`\`\`bash
GET /api/v1/datasets/convert/{task_id}
\`\`\`

\`\`\`json
{
  "task_id": "conv_xyz789",
  "status": "running",
  "images_converted": 840,
  "images_total": 1200
}
\`\`\`

### Streaming stats

\`\`\`bash
GET /api/v1/datasets/streaming/stats
\`\`\`

\`\`\`json
{
  "zarr_datasets": [
    { "path": "data/zarr/v2.zarr", "images": 1200, "size_gb": 3.4, "chunk_size": 64 }
  ],
  "hdf5_datasets": [
    { "path": "data/hdf5/v2.h5", "images": 1200, "size_gb": 3.1 }
  ]
}
\`\`\`

---

## Frontend: Datasets page → Streaming Formats tab

- **Format selector** — choose Zarr or HDF5 as the conversion target.
- **Conversion progress** — real-time images converted / total.
- **Stats table** — registered datasets with path, image count, and disk size.
- **Format comparison** — inline table for quick reference.

---

## Memory efficiency

Neither backend loads the full dataset into RAM. Memory usage during training is bounded by:

\`\`\`
peak_RAM ≈ batch_size × image_size_bytes × num_workers
\`\`\`

For 640×640 RGB images, \`batch_size=16\`, \`num_workers=4\`: approximately 300 MB — well within the 16 GB target system RAM.
`;

const de = `
## Streaming-Datensätze

Große Mikroskopie-Bilddatensätze übersteigen häufig den verfügbaren RAM. CTIP bietet zwei speichereffiziente Dataset-Backends: **Zarr** (chunk-basiert, streamingfähig) und **HDF5** (direkter Zugriff).

---

## Format-Vergleich

| Merkmal | Zarr | HDF5 |
|---|---|---|
| Haupteinsatz | Streaming / cloud-freundlich | Direktzugriff / klassisches HPC |
| Speicherlayout | Verzeichnisbaum aus Chunk-Dateien | Einzelne \`.h5\`-Datei |
| Zugriffsmuster | Sequenziell oder chunk-weise | Direkter Index-Zugriff |
| Kompression | Blosc (Standard), Zlib, LZ4 | GZIP, LZF |
| PyTorch DataLoader | Ja (\`ZarrDataset\`) | Ja (\`HDF5Dataset\`) |

---

## Python-Verwendung

\`\`\`python
# ZarrDataset erstellen
ZarrDataset.create_from_images(config)
dataset = ZarrDataset(config)

# HDF5Dataset verwenden
dataset = HDF5Dataset(config)
sample = dataset[0]  # {"image": Tensor[3,H,W], "labels": Tensor[N,5]}
\`\`\`

Beide Klassen sind Unterklassen von \`torch.utils.data.Dataset\`.

---

## API-Referenz

\`\`\`bash
# Konvertierung starten
POST /api/v1/datasets/convert

# Status prüfen
GET /api/v1/datasets/convert/{task_id}

# Streaming-Statistiken
GET /api/v1/datasets/streaming/stats
\`\`\`

---

## Speichereffizienz

Kein Backend lädt den gesamten Datensatz in den RAM. Der Speicherverbrauch während des Trainings ist auf \`batch_size × Bildgröße × num_workers\` begrenzt.
`;

const es = `
## Datasets de Streaming

Las colecciones grandes de imágenes de microscopía frecuentemente superan la RAM disponible. CTIP ofrece dos backends eficientes: **Zarr** (streaming por chunks) y **HDF5** (acceso aleatorio).

---

## Comparación de formatos

| Característica | Zarr | HDF5 |
|---|---|---|
| Caso de uso | Streaming / compatible con la nube | Acceso aleatorio / HPC clásico |
| Layout en disco | Árbol de archivos de chunk | Archivo único \`.h5\` |
| Compresión | Blosc (defecto), Zlib, LZ4 | GZIP, LZF |
| Compat. DataLoader | Sí (\`ZarrDataset\`) | Sí (\`HDF5Dataset\`) |

---

## Uso en Python

\`\`\`python
ZarrDataset.create_from_images(config)
dataset = ZarrDataset(config)
sample = dataset[0]  # {"image": Tensor[3,H,W], "labels": Tensor[N,5]}

dataset = HDF5Dataset(config)
\`\`\`

---

## Referencia de API

\`\`\`bash
POST /api/v1/datasets/convert
GET  /api/v1/datasets/convert/{task_id}
GET  /api/v1/datasets/streaming/stats
\`\`\`
`;

const page: WikiPage = {
  slug: 'streaming-datasets',
  title: {
    en: 'Streaming Datasets',
    de: 'Streaming-Datensätze',
    es: 'Datasets de Streaming',
  },
  description: {
    en: 'Zarr and HDF5 dataset backends for large microscopy datasets that exceed available RAM.',
    de: 'Zarr- und HDF5-Dataset-Backends für große Mikroskopie-Datensätze, die den RAM übersteigen.',
    es: 'Backends Zarr y HDF5 para datasets de microscopía grandes que superan la RAM disponible.',
  },
  content: { en, de, es },
  section: 'reference',
  icon: '🗄️',
};

export default page;
