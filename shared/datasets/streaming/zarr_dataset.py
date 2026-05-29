"""
shared.datasets.streaming.zarr_dataset — Zarr-backed streaming dataset.

Design rationale:
    - zarr v3 chunk-based I/O gives sequential throughput without loading
      the full dataset into RAM (critical for 16 GB RAM / RTX 4060 targets).
    - IterableDataset is used instead of map-style Dataset because the
      optimal access pattern is sequential chunk reads, not random indexing.
    - Worker sharding is chunk-granular: each DataLoader worker owns a
      contiguous slice of chunks, avoiding interleaved I/O across workers.
    - Images are written once at creation time at the target image_size to
      eliminate per-sample resize overhead during training.

Store layout:
    /images   shape=(N, H, W, 3)           dtype=uint8   chunks=(chunk_size, H, W, 3)
    /labels   shape=(N, max_boxes, 5)       dtype=float32 chunks=(chunk_size, max_boxes, 5)
    /meta     shape=(N,)                    dtype=str     chunks=(chunk_size,)

Label encoding: [class_id, cx, cy, w, h]  all values normalised to [0, 1].
                Padding rows are [−1, 0, 0, 0, 0] (class_id < 0 → ignore).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np
import torch
import torch.utils.data
import zarr
import zarr.storage
from PIL import Image

from shared.logging.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_PADDING_LABEL: list[float] = [-1.0, 0.0, 0.0, 0.0, 0.0]  # sentinel for empty boxes


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class ZarrDatasetConfig:
    """Configuration for a ZarrDataset instance.

    Args:
        zarr_path: Path to zarr store directory or .zip archive.
        split: Dataset split identifier (train|val|test).  Used as an
            informational label only — the store is expected to already
            contain only the relevant split's data.
        image_size: Square resize target in pixels.  0 = keep original size.
        chunk_size: Number of samples per zarr chunk along the batch axis.
            Larger values improve sequential throughput at the cost of higher
            per-chunk I/O latency.
        prefetch_size: Number of chunks to load ahead of the current
            iteration position (implemented via a background thread).
        augment: Whether to apply training augmentations during iteration.
        seed: Random seed used for shuffling and augmentation.
    """

    zarr_path: str
    split: str = "train"
    image_size: int = 640
    chunk_size: int = 64
    prefetch_size: int = 4
    augment: bool = False
    seed: int = 42


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class ZarrDataset(torch.utils.data.IterableDataset):
    """Streaming IterableDataset backed by a zarr store.

    Each sample yields:
        image   : torch.Tensor  shape=(3, H, W)  dtype=float32  range=[0, 1]
        annots  : list[dict]    [{cls, cx, cy, w, h}]  normalised coords
        meta    : dict          parsed JSON metadata for the sample

    Multi-worker DataLoader support:
        Pass ``dataset.get_worker_init_fn()`` as the ``worker_init_fn``
        argument to DataLoader.  The function splits chunk ranges evenly
        across workers so each worker reads a disjoint subset of chunks.

    Example::

        cfg = ZarrDatasetConfig(zarr_path="data/train.zarr", split="train")
        ds = ZarrDataset(cfg)
        loader = DataLoader(ds, batch_size=8, num_workers=4,
                            worker_init_fn=ds.get_worker_init_fn(),
                            collate_fn=zarr_collate_fn)
    """

    def __init__(self, config: ZarrDatasetConfig) -> None:
        super().__init__()
        self.config = config
        self._store_path = str(config.zarr_path)

        # Open store to read metadata (shape, dtype).
        # We do NOT keep this handle open across workers.
        root = self._open_store(mode="r")
        images_arr = root["images"]
        self._total = images_arr.shape[0]
        self._img_h = images_arr.shape[1]
        self._img_w = images_arr.shape[2]
        self._max_boxes = root["labels"].shape[1]
        self._chunk_size = config.chunk_size

        # Compute number of chunks
        self._chunk_count = math.ceil(self._total / self._chunk_size)

        logger.info(
            "ZarrDataset initialised",
            path=self._store_path,
            total=self._total,
            h=self._img_h,
            w=self._img_w,
            chunks=self._chunk_count,
        )

        # Worker chunk range — set by worker_init_fn at DataLoader startup.
        # Default: all chunks (single-worker or non-sharded usage).
        self._worker_chunk_start: int = 0
        self._worker_chunk_end: int = self._chunk_count

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _open_store(self, mode: str = "r") -> zarr.Group:
        """Open the zarr store and return the root Group."""
        store = zarr.storage.LocalStore(self._store_path)
        return zarr.open_group(store=store, mode=mode)

    def _load_chunk(self, chunk_idx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load a single chunk from the zarr store.

        Returns:
            images  : (chunk_size, H, W, 3)  uint8
            labels  : (chunk_size, max_boxes, 5)  float32
            meta    : (chunk_size,)  str
        """
        start = chunk_idx * self._chunk_size
        end = min(start + self._chunk_size, self._total)

        root = self._open_store(mode="r")
        images = root["images"][start:end]  # (n, H, W, 3) uint8
        labels = root["labels"][start:end]  # (n, max_boxes, 5) float32
        meta_raw = root["meta"][start:end]  # (n,) str
        return images, labels, meta_raw

    @staticmethod
    def _decode_sample(
        image_hwc_u8: np.ndarray,
        label_row: np.ndarray,
        meta_str: str,
        augment: bool,
        rng: np.random.Generator,
    ) -> tuple[torch.Tensor, list[dict[str, Any]], dict[str, Any]]:
        """Convert a single raw sample into the canonical output format.

        Args:
            image_hwc_u8: (H, W, 3) uint8 numpy array.
            label_row:    (max_boxes, 5) float32 — [cls, cx, cy, w, h].
            meta_str:     JSON-encoded metadata string.
            augment:      Apply horizontal-flip augmentation when True.
            rng:          Seeded numpy Generator for reproducible augmentation.

        Returns:
            image_tensor: (3, H, W) float32 in [0, 1].
            annotations:  list of non-padding label dicts.
            meta:         parsed metadata dict.
        """
        # Augmentation: random horizontal flip
        if augment and rng.random() > 0.5:
            image_hwc_u8 = np.ascontiguousarray(image_hwc_u8[:, ::-1, :])
            # Flip cx: cx_new = 1 - cx_old
            valid = label_row[:, 0] >= 0
            label_row = label_row.copy()
            label_row[valid, 1] = 1.0 - label_row[valid, 1]

        # HWC uint8 → CHW float32 [0, 1]
        image_tensor = torch.from_numpy(
            np.ascontiguousarray(image_hwc_u8).astype(np.float32) / 255.0
        ).permute(2, 0, 1)  # (3, H, W)

        # Decode annotations (skip padding rows where cls < 0)
        annotations: list[dict[str, Any]] = []
        for row in label_row:
            if row[0] < 0:
                continue
            annotations.append(
                {
                    "cls": int(row[0]),
                    "cx": float(row[1]),
                    "cy": float(row[2]),
                    "w": float(row[3]),
                    "h": float(row[4]),
                }
            )

        # Decode metadata
        try:
            meta: dict[str, Any] = json.loads(meta_str) if meta_str else {}
        except (json.JSONDecodeError, TypeError):
            meta = {"raw": str(meta_str)}

        return image_tensor, annotations, meta

    # ------------------------------------------------------------------
    # IterableDataset interface
    # ------------------------------------------------------------------

    def __iter__(self) -> Iterator[tuple[torch.Tensor, list[dict[str, Any]], dict[str, Any]]]:
        rng = np.random.default_rng(self.config.seed)

        for chunk_idx in range(self._worker_chunk_start, self._worker_chunk_end):
            images_chunk, labels_chunk, meta_chunk = self._load_chunk(chunk_idx)

            for i in range(len(images_chunk)):
                yield self._decode_sample(
                    images_chunk[i],
                    labels_chunk[i],
                    str(meta_chunk[i]),
                    self.config.augment,
                    rng,
                )

    def __len__(self) -> int:
        """Total number of samples in the dataset (across all workers)."""
        return self._total

    # ------------------------------------------------------------------
    # Worker sharding
    # ------------------------------------------------------------------

    def get_worker_init_fn(self) -> Callable[[int], None]:
        """Return a ``worker_init_fn`` that shards chunks across DataLoader workers.

        Each worker receives a contiguous, non-overlapping range of chunk
        indices.  Uneven chunk counts are distributed round-robin so that
        worker 0 receives any extra chunk.

        Usage::

            loader = DataLoader(
                dataset,
                num_workers=4,
                worker_init_fn=dataset.get_worker_init_fn(),
            )
        """
        total_chunks = self._chunk_count
        dataset_ref = self  # capture for closure

        def _worker_init(worker_id: int) -> None:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is None:
                return  # single-process mode — no sharding needed

            num_workers = worker_info.num_workers
            chunks_per_worker = total_chunks // num_workers
            remainder = total_chunks % num_workers

            # Distribute remainder to the first `remainder` workers
            if worker_id < remainder:
                start = worker_id * (chunks_per_worker + 1)
                end = start + chunks_per_worker + 1
            else:
                start = worker_id * chunks_per_worker + remainder
                end = start + chunks_per_worker

            dataset_ref._worker_chunk_start = start
            dataset_ref._worker_chunk_end = end

        return _worker_init

    # ------------------------------------------------------------------
    # Factory: create from raw images
    # ------------------------------------------------------------------

    @staticmethod
    def create_from_images(
        image_paths: list[str],
        annotations: list[list[dict[str, Any]]],
        output_path: str,
        chunk_size: int = 64,
        image_size: int = 640,
    ) -> "ZarrDataset":
        """Build a zarr store from image files and annotation dicts.

        Writes data in batches of ``chunk_size`` images to avoid OOM on
        systems with limited RAM.  Images are resized to ``image_size`` ×
        ``image_size`` at write time.

        Args:
            image_paths:  Ordered list of absolute or relative image paths.
            annotations:  Per-image annotation lists.  Each annotation is a
                dict with keys: ``cls`` (int), ``x_min``, ``y_min``,
                ``x_max``, ``y_max`` (all float, pixel coordinates).
                Empty lists are valid (unannotated images).
            output_path:  Directory path for the new zarr store.
            chunk_size:   Number of samples per zarr chunk.
            image_size:   Square resize target in pixels (0 = keep original).

        Returns:
            ZarrDataset instance pointing at the new store.

        Raises:
            ValueError: If ``image_paths`` and ``annotations`` have
                different lengths.
            FileNotFoundError: If an image file cannot be read.
        """
        if len(image_paths) != len(annotations):
            raise ValueError(
                f"image_paths length ({len(image_paths)}) != "
                f"annotations length ({len(annotations)})"
            )
        if len(image_paths) == 0:
            raise ValueError("image_paths must not be empty")

        n_total = len(image_paths)

        # Determine output image dimensions by reading the first image.
        try:
            _first_pil = Image.open(str(image_paths[0])).convert("RGB")
        except (FileNotFoundError, OSError) as exc:
            raise FileNotFoundError(f"Cannot read image: {image_paths[0]}") from exc
        if image_size > 0:
            out_h = out_w = image_size
        else:
            out_w, out_h = _first_pil.size  # PIL: (width, height)

        # Find max_boxes across all annotation lists for label padding.
        max_boxes = max((len(a) for a in annotations), default=1)
        max_boxes = max(max_boxes, 1)  # always at least 1

        # Create zarr store
        store_path = str(output_path)
        Path(store_path).mkdir(parents=True, exist_ok=True)
        store = zarr.storage.LocalStore(store_path)
        root = zarr.open_group(store=store, mode="w")

        images_arr = root.create_array(
            "images",
            shape=(n_total, out_h, out_w, 3),
            dtype="u1",
            chunks=(chunk_size, out_h, out_w, 3),
        )
        labels_arr = root.create_array(
            "labels",
            shape=(n_total, max_boxes, 5),
            dtype="f4",
            chunks=(chunk_size, max_boxes, 5),
        )
        meta_arr = root.create_array(
            "meta",
            shape=(n_total,),
            dtype=str,
            chunks=(chunk_size,),
        )

        logger.info(
            "Creating zarr store",
            path=store_path,
            n=n_total,
            img_size=image_size,
            chunk_size=chunk_size,
            max_boxes=max_boxes,
        )

        # Write in chunk-sized batches to bound peak RAM usage.
        n_chunks = math.ceil(n_total / chunk_size)
        for chunk_idx in range(n_chunks):
            start = chunk_idx * chunk_size
            end = min(start + chunk_size, n_total)
            batch_size = end - start

            imgs_batch = np.zeros((batch_size, out_h, out_w, 3), dtype=np.uint8)
            labels_batch = np.full(
                (batch_size, max_boxes, 5), _PADDING_LABEL, dtype=np.float32
            )
            meta_batch: list[str] = []

            for local_i, global_i in enumerate(range(start, end)):
                path = str(image_paths[global_i])
                try:
                    pil_img = Image.open(path).convert("RGB")
                    if image_size > 0 and (pil_img.height != out_h or pil_img.width != out_w):
                        pil_img = pil_img.resize((out_w, out_h), Image.BILINEAR)
                    img = np.asarray(pil_img, dtype=np.uint8)
                except (FileNotFoundError, OSError):
                    logger.warning("Cannot read image, using blank", path=path)
                    img = np.zeros((out_h, out_w, 3), dtype=np.uint8)

                imgs_batch[local_i] = img

                ann_list = annotations[global_i]
                img_h_orig, img_w_orig = img.shape[:2]
                for box_i, ann in enumerate(ann_list[:max_boxes]):
                    # Convert xyxy pixel coords → normalised cxcywh
                    x_min = float(ann.get("x_min", ann.get("xmin", 0.0)))
                    y_min = float(ann.get("y_min", ann.get("ymin", 0.0)))
                    x_max = float(ann.get("x_max", ann.get("xmax", float(out_w))))
                    y_max = float(ann.get("y_max", ann.get("ymax", float(out_h))))
                    cls_id = int(ann.get("cls", ann.get("class_id", 0)))

                    cx = (x_min + x_max) / 2.0 / out_w
                    cy = (y_min + y_max) / 2.0 / out_h
                    bw = (x_max - x_min) / out_w
                    bh = (y_max - y_min) / out_h

                    labels_batch[local_i, box_i] = [cls_id, cx, cy, bw, bh]

                meta_batch.append(
                    json.dumps(
                        {
                            "path": str(path),
                            "frame_idx": global_i,
                            "original_index": global_i,
                        }
                    )
                )

            images_arr[start:end] = imgs_batch
            labels_arr[start:end] = labels_batch
            meta_arr[start:end] = meta_batch

            logger.debug(
                "Wrote chunk",
                chunk=chunk_idx + 1,
                total_chunks=n_chunks,
                samples=batch_size,
            )

        logger.info("Zarr store creation complete", path=store_path, total=n_total)

        return ZarrDataset(
            ZarrDatasetConfig(
                zarr_path=store_path,
                chunk_size=chunk_size,
                image_size=0,  # already resized
            )
        )

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    @property
    def stats(self) -> dict[str, Any]:
        """Return summary statistics for the store.

        Returns:
            dict with keys:
                ``total_images``  (int)  — number of samples
                ``chunk_count``   (int)  — number of zarr chunks
                ``store_size_mb`` (float) — on-disk size in megabytes
        """
        store_path = Path(self._store_path)
        size_bytes = sum(
            f.stat().st_size
            for f in store_path.rglob("*")
            if f.is_file()
        )
        return {
            "total_images": self._total,
            "chunk_count": self._chunk_count,
            "store_size_mb": round(size_bytes / (1024 ** 2), 3),
        }
