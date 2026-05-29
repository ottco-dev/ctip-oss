## Streaming Datasets

Large microscopy image collections frequently exceed available RAM. CTIP provides two memory-efficient dataset backends — **Zarr** (chunked, streamable) and **HDF5** (random-access) — that integrate directly with PyTorch DataLoaders without requiring the entire dataset to be loaded at once.

---

## Format comparison

| Feature | Zarr | HDF5 |
|---|---|---|
| Primary use case | Streaming / cloud-friendly | Random-access / classic HPC |
| On-disk layout | Directory tree of chunk files | Single `.h5` file |
| Access pattern | Sequential or chunked streaming | Direct index lookup |
| Compression | Blosc (default), Zlib, LZ4 | GZIP, LZF |
| PyTorch DataLoader compat | Yes (`ZarrDataset`) | Yes (`HDF5Dataset`) |
| Partial update | Append new chunks | Re-write array |
| Recommended for | Incremental acquisition, S3/NFS | Fixed dataset, fast local SSD |

---

## Python usage

### ZarrDataset

```python
from backend.training.datasets.zarr_dataset import ZarrDataset, ZarrDatasetConfig

# Convert an existing YOLO dataset to zarr
config = ZarrDatasetConfig(
    source_root="data/datasets/v2/",
    zarr_root="data/zarr/v2.zarr",
    chunk_size=64,
    compression="blosc",
)
ZarrDataset.create_from_images(config)

# Use in training
dataset = ZarrDataset(config)
sample = dataset[0]   # {"image": Tensor[3,H,W], "labels": Tensor[N,5]}
```

### HDF5Dataset

```python
from backend.training.datasets.hdf5_dataset import HDF5Dataset, HDF5DatasetConfig

config = HDF5DatasetConfig(
    h5_path="data/hdf5/v2.h5",
    split="train",
)
dataset = HDF5Dataset(config)
sample = dataset[0]   # {"image": Tensor[3,H,W], "labels": Tensor[N,5]}
```

Both classes are `torch.utils.data.Dataset` subclasses and can be wrapped in a standard `DataLoader`.

---

## DatasetConverter

`DatasetConverter` supports four conversion paths:

| Source | Target | CLI flag |
|---|---|---|
| YOLO dataset | Zarr | `yolo_to_zarr` |
| YOLO dataset | HDF5 | `yolo_to_hdf5` |
| Zarr | HDF5 | `zarr_to_hdf5` |
| HDF5 | Zarr | `hdf5_to_zarr` |

---

## API reference

### Start a conversion task

```bash
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
```

Response:
```json
{
  "task_id": "conv_xyz789",
  "status": "queued"
}
```

### Check conversion status

```bash
GET /api/v1/datasets/convert/{task_id}
```

```json
{
  "task_id": "conv_xyz789",
  "status": "running",
  "images_converted": 840,
  "images_total": 1200
}
```

### Streaming stats

```bash
GET /api/v1/datasets/streaming/stats
```

```json
{
  "zarr_datasets": [
    { "path": "data/zarr/v2.zarr", "images": 1200, "size_gb": 3.4, "chunk_size": 64 }
  ],
  "hdf5_datasets": [
    { "path": "data/hdf5/v2.h5", "images": 1200, "size_gb": 3.1 }
  ]
}
```

---

## Frontend: Datasets page → Streaming Formats tab

- **Format selector** — choose Zarr or HDF5 as the conversion target.
- **Conversion progress** — real-time images converted / total.
- **Stats table** — registered datasets with path, image count, and disk size.
- **Format comparison** — inline table (reproduced from above) for quick reference.

---

## Memory efficiency

Neither backend loads the full dataset into RAM. Memory usage during training is bounded by:

```
peak_RAM ≈ batch_size × image_size_bytes × num_workers
```

For 640×640 RGB images, `batch_size=16`, `num_workers=4`:

```
peak_RAM ≈ 16 × (640×640×3) × 4 ≈ 300 MB
```

This fits comfortably within the 16 GB target system RAM, even when the YOLO trainer is also holding model weights in VRAM.
