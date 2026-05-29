"""
shared.datasets.streaming.dataset_converter — Cross-format dataset conversion.

Supported conversions:
    YOLO directory  → zarr stores (train/val/test)
    YOLO directory  → HDF5 file   (train/val/test splits as groups)
    zarr store      → HDF5 file
    HDF5 file       → zarr store

YOLO directory layout expected:
    {root}/
        images/   *.jpg | *.png | *.jpeg | *.tiff | *.tif | *.bmp
        labels/   *.txt  (one file per image, YOLO normalised format)

YOLO label format (per line):
    class_id  cx  cy  w  h   (all normalised 0-1, space-separated)

Split reproducibility:
    Splits are determined by seeded shuffle of sorted file lists.
    Same seed + same file list → identical splits.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from shared.datasets.streaming.hdf5_dataset import HDF5Dataset, HDF5DatasetConfig
from shared.datasets.streaming.zarr_dataset import ZarrDataset, ZarrDatasetConfig
from shared.logging.logger import get_logger

logger = get_logger(__name__)

# Image file extensions accepted by YOLO loaders
_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".webp"}


def _read_yolo_labels(label_path: Path) -> list[dict[str, Any]]:
    """Parse a YOLO-format .txt label file.

    Returns a list of annotation dicts with keys:
        ``cls`` (int), ``x_min``, ``y_min``, ``x_max``, ``y_max`` (float).

    Coordinates are returned as normalised pixel coordinates in the range
    [0, 1] scaled to a 1×1 canvas (i.e., ``x_min = cx - w/2``).
    The downstream create_from_images methods expect pixel coordinates
    relative to the *stored* image size; since images are resized uniformly,
    passing normalised values multiplied by 1.0 preserves validity when
    image_size is passed as the denominator in the converter calls.

    Actually for simplicity: we return unnormalised coords assuming a 1×1
    image so that the downstream methods correctly re-normalise them when
    image_size is the denominator.
    """
    annotations: list[dict[str, Any]] = []
    if not label_path.exists():
        return annotations

    with open(label_path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                cls_id = int(parts[0])
                cx = float(parts[1])
                cy = float(parts[2])
                bw = float(parts[3])
                bh = float(parts[4])
            except ValueError:
                continue

            # Convert normalised cxcywh → normalised xyxy on [0,1] canvas.
            # Downstream create_from_images will re-normalise by out_h/out_w.
            x_min = cx - bw / 2.0
            y_min = cy - bh / 2.0
            x_max = cx + bw / 2.0
            y_max = cy + bh / 2.0

            annotations.append(
                {
                    "cls": cls_id,
                    "x_min": x_min,
                    "y_min": y_min,
                    "x_max": x_max,
                    "y_max": y_max,
                }
            )
    return annotations


def _collect_yolo_pairs(
    yolo_root: str,
) -> tuple[list[str], list[list[dict[str, Any]]]]:
    """Collect (image_path, annotations) pairs from a YOLO directory.

    Returns sorted by image path for deterministic ordering.
    """
    root = Path(yolo_root)
    images_dir = root / "images"
    labels_dir = root / "labels"

    if not images_dir.exists():
        raise FileNotFoundError(f"YOLO images directory not found: {images_dir}")

    image_paths = sorted(
        p for p in images_dir.rglob("*") if p.suffix.lower() in _IMAGE_EXTENSIONS
    )

    if not image_paths:
        raise FileNotFoundError(f"No images found under: {images_dir}")

    all_image_paths: list[str] = []
    all_annotations: list[list[dict[str, Any]]] = []

    for img_path in image_paths:
        label_path = labels_dir / img_path.with_suffix(".txt").name
        all_image_paths.append(str(img_path))
        all_annotations.append(_read_yolo_labels(label_path))

    return all_image_paths, all_annotations


def _split_indices(
    n: int,
    val_split: float,
    test_split: float,
    seed: int,
) -> tuple[list[int], list[int], list[int]]:
    """Deterministically split n indices into train/val/test subsets.

    Args:
        n:          Total number of samples.
        val_split:  Fraction for validation (e.g. 0.15).
        test_split: Fraction for test (e.g. 0.10).
        seed:       Random seed for reproducibility.

    Returns:
        (train_indices, val_indices, test_indices)
    """
    if val_split + test_split >= 1.0:
        raise ValueError(
            f"val_split ({val_split}) + test_split ({test_split}) must be < 1.0"
        )

    rng = np.random.default_rng(seed)
    indices = list(range(n))
    rng.shuffle(indices)

    n_test = max(1, math.floor(n * test_split))
    n_val = max(1, math.floor(n * val_split))
    n_train = n - n_val - n_test

    if n_train <= 0:
        raise ValueError(
            f"Not enough samples ({n}) for requested splits "
            f"(val={val_split}, test={test_split})"
        )

    train_idx = sorted(indices[:n_train])
    val_idx = sorted(indices[n_train : n_train + n_val])
    test_idx = sorted(indices[n_train + n_val :])
    return train_idx, val_idx, test_idx


def _subset(lst: list, indices: list[int]) -> list:
    return [lst[i] for i in indices]


class DatasetConverter:
    """Convert between dataset formats for the trichome-analysis pipeline.

    All methods are static — no instance state is required.

    Conversion matrix:
        YOLO dir  → zarr (per-split stores)
        YOLO dir  → HDF5 (splits as groups in one file)
        zarr      → HDF5
        HDF5      → zarr
    """

    # ------------------------------------------------------------------
    # YOLO → zarr
    # ------------------------------------------------------------------

    @staticmethod
    def yolo_to_zarr(
        yolo_root: str,
        output_path: str,
        val_split: float = 0.15,
        test_split: float = 0.10,
        image_size: int = 640,
        seed: int = 42,
    ) -> dict[str, ZarrDataset]:
        """Convert a YOLO dataset directory to per-split zarr stores.

        Args:
            yolo_root:   Path to the YOLO root directory (contains images/
                         and labels/).
            output_path: Parent directory for the output zarr stores.
                         Three sub-stores will be created:
                         ``{output_path}/train.zarr``,
                         ``{output_path}/val.zarr``,
                         ``{output_path}/test.zarr``.
            val_split:   Fraction of data to use for validation.
            test_split:  Fraction of data to use for test.
            image_size:  Square resize target in pixels.
            seed:        Random seed for reproducible splitting.

        Returns:
            dict with keys ``"train"``, ``"val"``, ``"test"`` mapping to
            the corresponding ZarrDataset instances.
        """
        logger.info(
            "Converting YOLO → zarr",
            source=yolo_root,
            dest=output_path,
            val_split=val_split,
            test_split=test_split,
        )

        all_paths, all_annotations = _collect_yolo_pairs(yolo_root)
        train_idx, val_idx, test_idx = _split_indices(
            len(all_paths), val_split, test_split, seed
        )

        out_root = Path(output_path)
        out_root.mkdir(parents=True, exist_ok=True)

        results: dict[str, ZarrDataset] = {}
        for split_name, indices in [
            ("train", train_idx),
            ("val", val_idx),
            ("test", test_idx),
        ]:
            subset_paths = _subset(all_paths, indices)
            subset_anns = _subset(all_annotations, indices)
            store_path = str(out_root / f"{split_name}.zarr")

            logger.info(
                "Building zarr split",
                split=split_name,
                n=len(subset_paths),
                path=store_path,
            )

            ds = ZarrDataset.create_from_images(
                image_paths=subset_paths,
                annotations=subset_anns,
                output_path=store_path,
                image_size=image_size,
            )
            results[split_name] = ds

        logger.info("YOLO → zarr conversion complete", splits=list(results.keys()))
        return results

    # ------------------------------------------------------------------
    # YOLO → HDF5
    # ------------------------------------------------------------------

    @staticmethod
    def yolo_to_hdf5(
        yolo_root: str,
        output_path: str,
        val_split: float = 0.15,
        test_split: float = 0.10,
        image_size: int = 640,
        seed: int = 42,
        compression: str = "gzip",
        compression_opts: int = 4,
    ) -> dict[str, HDF5Dataset]:
        """Convert a YOLO dataset directory to a multi-split HDF5 file.

        All three splits (train/val/test) are written as groups into a
        single HDF5 file at ``output_path``.

        Args:
            yolo_root:        YOLO root directory.
            output_path:      Output .h5 file path.
            val_split:        Validation fraction.
            test_split:       Test fraction.
            image_size:       Square resize target.
            seed:             Random seed.
            compression:      HDF5 compression filter (gzip|lzf|None).
            compression_opts: gzip level.

        Returns:
            dict with keys ``"train"``, ``"val"``, ``"test"`` mapping to
            HDF5Dataset instances for each split.
        """
        logger.info(
            "Converting YOLO → HDF5",
            source=yolo_root,
            dest=output_path,
            val_split=val_split,
            test_split=test_split,
        )

        all_paths, all_annotations = _collect_yolo_pairs(yolo_root)
        train_idx, val_idx, test_idx = _split_indices(
            len(all_paths), val_split, test_split, seed
        )

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        results: dict[str, HDF5Dataset] = {}
        for split_name, indices in [
            ("train", train_idx),
            ("val", val_idx),
            ("test", test_idx),
        ]:
            subset_paths = _subset(all_paths, indices)
            subset_anns = _subset(all_annotations, indices)

            logger.info(
                "Building HDF5 split",
                split=split_name,
                n=len(subset_paths),
                path=output_path,
            )

            ds = HDF5Dataset.create_from_images(
                image_paths=subset_paths,
                annotations=subset_anns,
                output_path=output_path,
                split=split_name,
                image_size=image_size,
                compression=compression,
                compression_opts=compression_opts,
            )
            results[split_name] = ds

        logger.info("YOLO → HDF5 conversion complete", path=output_path)
        return results

    # ------------------------------------------------------------------
    # zarr → HDF5
    # ------------------------------------------------------------------

    @staticmethod
    def zarr_to_hdf5(zarr_path: str, hdf5_path: str) -> None:
        """Convert a zarr store to an HDF5 file.

        The zarr store is treated as a single split ("train").  The output
        HDF5 file will have a single group ``/train`` containing the
        ``images``, ``labels``, and ``meta`` datasets.

        Args:
            zarr_path:  Path to the source zarr store directory.
            hdf5_path:  Path for the output HDF5 file.
        """
        import zarr
        import zarr.storage
        import h5py

        logger.info("Converting zarr → HDF5", source=zarr_path, dest=hdf5_path)

        store = zarr.storage.LocalStore(str(zarr_path))
        root = zarr.open_group(store=store, mode="r")

        images_z = root["images"]
        labels_z = root["labels"]
        meta_z = root["meta"]

        n = images_z.shape[0]
        h, w = images_z.shape[1], images_z.shape[2]
        max_boxes = labels_z.shape[1]
        chunk_size = images_z.chunks[0]

        Path(hdf5_path).parent.mkdir(parents=True, exist_ok=True)

        with h5py.File(hdf5_path, "w") as f:
            grp = f.create_group("train")
            img_ds = grp.create_dataset(
                "images",
                shape=(n, h, w, 3),
                dtype="uint8",
                chunks=(1, h, w, 3),
            )
            lbl_ds = grp.create_dataset(
                "labels",
                shape=(n, max_boxes, 5),
                dtype="float32",
                chunks=(1, max_boxes, 5),
            )
            meta_ds = grp.create_dataset(
                "meta",
                shape=(n,),
                dtype=h5py.string_dtype(),
            )

            n_chunks = math.ceil(n / chunk_size)
            for chunk_idx in range(n_chunks):
                start = chunk_idx * chunk_size
                end = min(start + chunk_size, n)
                img_ds[start:end] = images_z[start:end]
                lbl_ds[start:end] = labels_z[start:end]
                # Convert str array to list for h5py
                meta_chunk = meta_z[start:end]
                meta_ds[start:end] = [
                    str(m) if not isinstance(m, (str, bytes)) else m
                    for m in meta_chunk
                ]

                logger.debug(
                    "zarr→HDF5 chunk",
                    chunk=chunk_idx + 1,
                    total=n_chunks,
                )

        logger.info("zarr → HDF5 conversion complete", n=n, dest=hdf5_path)

    # ------------------------------------------------------------------
    # HDF5 → zarr
    # ------------------------------------------------------------------

    @staticmethod
    def hdf5_to_zarr(
        hdf5_path: str,
        zarr_path: str,
        split: str = "train",
        chunk_size: int = 64,
    ) -> None:
        """Convert an HDF5 split to a zarr store.

        Args:
            hdf5_path:   Path to the source HDF5 file.
            zarr_path:   Path for the output zarr store directory.
            split:       HDF5 group name to read (train|val|test).
            chunk_size:  Chunk size for the output zarr arrays.
        """
        import zarr
        import zarr.storage
        import h5py

        logger.info(
            "Converting HDF5 → zarr",
            source=hdf5_path,
            split=split,
            dest=zarr_path,
        )

        with h5py.File(hdf5_path, "r", swmr=True) as f:
            grp = f[split]
            n = grp["images"].shape[0]
            h, w = grp["images"].shape[1], grp["images"].shape[2]
            max_boxes = grp["labels"].shape[1]

            Path(zarr_path).mkdir(parents=True, exist_ok=True)
            store = zarr.storage.LocalStore(str(zarr_path))
            root = zarr.open_group(store=store, mode="w")

            images_z = root.create_array(
                "images",
                shape=(n, h, w, 3),
                dtype="u1",
                chunks=(chunk_size, h, w, 3),
            )
            labels_z = root.create_array(
                "labels",
                shape=(n, max_boxes, 5),
                dtype="f4",
                chunks=(chunk_size, max_boxes, 5),
            )
            meta_z = root.create_array(
                "meta",
                shape=(n,),
                dtype=str,
                chunks=(chunk_size,),
            )

            n_batches = math.ceil(n / chunk_size)
            for batch_idx in range(n_batches):
                start = batch_idx * chunk_size
                end = min(start + chunk_size, n)

                images_z[start:end] = grp["images"][start:end]
                labels_z[start:end] = grp["labels"][start:end]

                # h5py string dataset returns bytes or str depending on version
                meta_raw = grp["meta"][start:end]
                meta_strs = []
                for m in meta_raw:
                    if isinstance(m, bytes):
                        meta_strs.append(m.decode("utf-8"))
                    else:
                        meta_strs.append(str(m))
                meta_z[start:end] = meta_strs

                logger.debug(
                    "HDF5→zarr batch",
                    batch=batch_idx + 1,
                    total=n_batches,
                )

        logger.info("HDF5 → zarr conversion complete", n=n, dest=zarr_path)
