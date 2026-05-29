"""
shared.datasets.streaming — Memory-efficient streaming dataset backends.

Provides two complementary backends for large microscopy image datasets
that exceed available RAM:

ZarrDataset (IterableDataset):
    - Chunk-based streaming reads via zarr v3
    - Optimal for sequential throughput (training loops)
    - Supports DataLoader multi-worker via chunk-range sharding
    - Minimal random-access overhead

HDF5Dataset (Dataset):
    - Random-access reads via h5py with SWMR concurrency
    - Optimal for indexed access, validation, evaluation
    - Per-worker lazy file open avoids descriptor conflicts
    - Supports gzip/lzf compression per-chunk

DatasetConverter:
    - YOLO directory layout → zarr or HDF5
    - zarr ↔ HDF5 bidirectional conversion
    - Deterministic train/val/test splitting with seed control

Both formats use the same logical layout:
    /images   (N, H, W, 3)  uint8
    /labels   (N, max_boxes, 5)  float32  [cls, cx, cy, w, h]  normalised
    /meta     (N,)  string  JSON-encoded per-sample metadata
"""

from shared.datasets.streaming.zarr_dataset import ZarrDataset, ZarrDatasetConfig
from shared.datasets.streaming.hdf5_dataset import HDF5Dataset, HDF5DatasetConfig
from shared.datasets.streaming.dataset_converter import DatasetConverter

__all__ = [
    "ZarrDataset",
    "ZarrDatasetConfig",
    "HDF5Dataset",
    "HDF5DatasetConfig",
    "DatasetConverter",
]
