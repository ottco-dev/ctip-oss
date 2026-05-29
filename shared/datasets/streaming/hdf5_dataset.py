"""
shared.datasets.streaming.hdf5_dataset — HDF5-backed random-access dataset.

Design rationale:
    - Random-access map-style Dataset allows indexed sampling for validation
      and evaluation loops where sequential ordering is not required.
    - SWMR (Single Writer Multiple Reader) mode enables concurrent reads
      from multiple DataLoader worker processes without file-descriptor
      conflicts or data corruption.
    - Lazy per-worker file open: the h5py.File handle is NOT opened in
      __init__ to avoid pickling issues across multiprocessing workers.
      Instead, __getitem__ opens (and caches) the handle on first access
      within each worker process.
    - Compression (gzip/lzf) reduces disk footprint at the cost of slightly
      higher per-sample decompression overhead.

Store layout (mirrored across /train, /val, /test groups):
    /{split}/images   shape=(N, H, W, 3)       dtype=uint8
    /{split}/labels   shape=(N, max_boxes, 5)  dtype=float32
    /{split}/meta     shape=(N,)               dtype=h5py.string_dtype()

Label encoding: [class_id, cx, cy, w, h]  normalised to [0, 1].
                Padding rows: [−1, 0, 0, 0, 0] (class_id < 0 → ignore).
"""

from __future__ import annotations

import json
import math
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
import torch.utils.data
from PIL import Image

from shared.logging.logger import get_logger

logger = get_logger(__name__)

# Per-thread file handle cache — avoids redundant open() calls per sample
# while remaining safe across multiprocessing workers (each worker has its
# own thread-local storage).
_thread_local = threading.local()

_PADDING_LABEL: list[float] = [-1.0, 0.0, 0.0, 0.0, 0.0]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class HDF5DatasetConfig:
    """Configuration for an HDF5Dataset instance.

    Args:
        hdf5_path:       Path to the .h5 file.
        split:           Dataset split to read (train|val|test).
        image_size:      Square resize target in pixels.  0 = keep stored size.
        prefetch_size:   Prefetch buffer size hint (not currently used in this
                         implementation — h5py handles its own chunk cache).
        augment:         Apply training augmentations during iteration.
        seed:            Random seed for augmentation reproducibility.
    """

    hdf5_path: str
    split: str = "train"
    image_size: int = 640
    prefetch_size: int = 8
    augment: bool = False
    seed: int = 42


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class HDF5Dataset(torch.utils.data.Dataset):
    """Random-access Dataset backed by HDF5 with SWMR support.

    Each sample yields:
        image   : torch.Tensor  shape=(3, H, W)  dtype=float32  range=[0, 1]
        annots  : list[dict]    [{cls, cx, cy, w, h}]  normalised coords
        meta    : dict          parsed JSON metadata

    Multiprocessing safety:
        h5py file handles are opened lazily per-worker (on the first call
        to __getitem__) and cached in thread-local storage.  Pass
        ``num_workers >= 1`` to DataLoader freely.

    Example::

        cfg = HDF5DatasetConfig(hdf5_path="data/dataset.h5", split="train")
        ds = HDF5Dataset(cfg)
        loader = DataLoader(ds, batch_size=16, num_workers=4,
                            collate_fn=hdf5_collate_fn)
    """

    def __init__(self, config: HDF5DatasetConfig) -> None:
        super().__init__()
        self.config = config
        self._hdf5_path = str(config.hdf5_path)
        self._split = config.split

        # Read metadata using a short-lived handle in the constructor
        # (main process only — workers will reopen lazily).
        with h5py.File(self._hdf5_path, "r", swmr=True) as f:
            split_grp = f[self._split]
            self._total: int = split_grp["images"].shape[0]
            self._img_h: int = split_grp["images"].shape[1]
            self._img_w: int = split_grp["images"].shape[2]
            self._max_boxes: int = split_grp["labels"].shape[1]

        self._rng = np.random.default_rng(config.seed)

        logger.info(
            "HDF5Dataset initialised",
            path=self._hdf5_path,
            split=self._split,
            total=self._total,
            h=self._img_h,
            w=self._img_w,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_file_handle(self) -> h5py.File:
        """Return a thread-local h5py file handle keyed by path.

        The cache is a dict mapping hdf5_path → h5py.File so that multiple
        HDF5Dataset instances in the same thread each maintain their own
        independent handle rather than sharing one global slot.
        """
        cache: dict[str, h5py.File] = getattr(_thread_local, "h5_handles", None)
        if cache is None:
            cache = {}
            _thread_local.h5_handles = cache
        handle = cache.get(self._hdf5_path)
        if handle is None or not handle.id.valid:
            handle = h5py.File(self._hdf5_path, "r", swmr=True)
            cache[self._hdf5_path] = handle
        return handle

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self._total

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, list[dict[str, Any]], dict[str, Any]]:
        if idx < 0 or idx >= self._total:
            raise IndexError(
                f"Index {idx} out of range for dataset of size {self._total}"
            )

        f = self._get_file_handle()
        split_grp = f[self._split]

        image_hwc_u8: np.ndarray = split_grp["images"][idx]  # (H, W, 3) uint8
        label_row: np.ndarray = split_grp["labels"][idx]     # (max_boxes, 5) float32
        meta_bytes = split_grp["meta"][idx]
        meta_str: str = (
            meta_bytes.decode("utf-8")
            if isinstance(meta_bytes, bytes)
            else str(meta_bytes)
        )

        # Optional augmentation (horizontal flip)
        if self.config.augment and self._rng.random() > 0.5:
            image_hwc_u8 = np.ascontiguousarray(image_hwc_u8[:, ::-1, :])
            valid = label_row[:, 0] >= 0
            label_row = label_row.copy()
            label_row[valid, 1] = 1.0 - label_row[valid, 1]

        # HWC uint8 → CHW float32 [0, 1]
        image_tensor = torch.from_numpy(
            np.ascontiguousarray(image_hwc_u8).astype(np.float32) / 255.0
        ).permute(2, 0, 1)

        # Decode annotations
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
            meta = {"raw": meta_str}

        return image_tensor, annotations, meta

    # ------------------------------------------------------------------
    # Factory: create from raw images
    # ------------------------------------------------------------------

    @staticmethod
    def create_from_images(
        image_paths: list[str],
        annotations: list[list[dict[str, Any]]],
        output_path: str,
        split: str = "train",
        image_size: int = 640,
        compression: str = "gzip",
        compression_opts: int = 4,
    ) -> "HDF5Dataset":
        """Build an HDF5 file from image files and annotation dicts.

        Args:
            image_paths:      Ordered list of image file paths.
            annotations:      Per-image annotation lists.  Each annotation
                              is a dict with keys: ``cls``, ``x_min``,
                              ``y_min``, ``x_max``, ``y_max``.
            output_path:      Path for the output .h5 file.
            split:            Group name inside the HDF5 file (train|val|test).
            image_size:       Square resize target (0 = keep original).
            compression:      h5py compression filter: "gzip", "lzf", or None.
            compression_opts: gzip compression level (1–9).  Ignored for lzf.

        Returns:
            HDF5Dataset instance pointing at the new file.

        Raises:
            ValueError: Mismatched lengths or empty input.
            FileNotFoundError: Image file cannot be read.
        """
        if len(image_paths) != len(annotations):
            raise ValueError(
                f"image_paths length ({len(image_paths)}) != "
                f"annotations length ({len(annotations)})"
            )
        if len(image_paths) == 0:
            raise ValueError("image_paths must not be empty")

        n_total = len(image_paths)

        # Determine output image dimensions from first image
        try:
            _first_pil = Image.open(str(image_paths[0])).convert("RGB")
        except (FileNotFoundError, OSError) as exc:
            raise FileNotFoundError(f"Cannot read image: {image_paths[0]}") from exc
        if image_size > 0:
            out_h = out_w = image_size
        else:
            out_w, out_h = _first_pil.size  # PIL: (width, height)

        max_boxes = max((len(a) for a in annotations), default=1)
        max_boxes = max(max_boxes, 1)

        # Resolve compression kwargs
        compress_kwargs: dict[str, Any] = {}
        if compression and compression.lower() != "none":
            compress_kwargs["compression"] = compression.lower()
            if compression.lower() == "gzip":
                compress_kwargs["compression_opts"] = compression_opts

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        logger.info(
            "Creating HDF5 store",
            path=output_path,
            split=split,
            n=n_total,
            img_size=image_size,
            compression=compression,
        )

        chunk_size = min(64, n_total)

        with h5py.File(output_path, "w") as f:
            grp = f.create_group(split)

            img_ds = grp.create_dataset(
                "images",
                shape=(n_total, out_h, out_w, 3),
                dtype="uint8",
                chunks=(1, out_h, out_w, 3),
                **compress_kwargs,
            )
            lbl_ds = grp.create_dataset(
                "labels",
                shape=(n_total, max_boxes, 5),
                dtype="float32",
                chunks=(1, max_boxes, 5),
                **compress_kwargs,
            )
            meta_ds = grp.create_dataset(
                "meta",
                shape=(n_total,),
                dtype=h5py.string_dtype(),
            )

            # Write in batches to keep RAM usage bounded
            n_batches = math.ceil(n_total / chunk_size)
            for batch_idx in range(n_batches):
                start = batch_idx * chunk_size
                end = min(start + chunk_size, n_total)
                batch_n = end - start

                imgs_batch = np.zeros((batch_n, out_h, out_w, 3), dtype=np.uint8)
                labels_batch = np.full(
                    (batch_n, max_boxes, 5), _PADDING_LABEL, dtype=np.float32
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
                    for box_i, ann in enumerate(ann_list[:max_boxes]):
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
                                "path": path,
                                "frame_idx": global_i,
                                "original_index": global_i,
                                "split": split,
                            }
                        )
                    )

                img_ds[start:end] = imgs_batch
                lbl_ds[start:end] = labels_batch
                meta_ds[start:end] = meta_batch

                logger.debug(
                    "Wrote HDF5 batch",
                    batch=batch_idx + 1,
                    total=n_batches,
                    samples=batch_n,
                )

        logger.info("HDF5 store creation complete", path=output_path, split=split, total=n_total)

        return HDF5Dataset(
            HDF5DatasetConfig(
                hdf5_path=output_path,
                split=split,
                image_size=0,  # already resized at write time
            )
        )

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    @property
    def stats(self) -> dict[str, Any]:
        """Return summary statistics for the HDF5 file.

        Returns:
            dict with keys:
                ``total_images``  (int)   — number of samples in the split
                ``split``         (str)   — active split name
                ``store_size_mb`` (float) — on-disk file size in megabytes
                ``max_boxes``     (int)   — max annotation rows per sample
        """
        size_mb = round(Path(self._hdf5_path).stat().st_size / (1024 ** 2), 3)
        return {
            "total_images": self._total,
            "split": self._split,
            "store_size_mb": size_mb,
            "max_boxes": self._max_boxes,
        }
