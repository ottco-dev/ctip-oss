"""
backend.api.v1.datasets — Dataset management endpoints.

Endpoints:
    GET    /datasets              — List datasets
    POST   /datasets              — Create dataset
    GET    /datasets/{id}         — Get dataset details
    DELETE /datasets/{id}         — Delete dataset
    GET    /datasets/{id}/samples — List samples
    POST   /datasets/{id}/upload  — Upload images to dataset
    GET    /datasets/{id}/stats   — Dataset statistics
"""

from __future__ import annotations

import hashlib
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sqlmodel import Session, select, func

from backend.config import get_settings
from backend.database import get_session
from backend.models.dataset import Dataset, Sample, Annotation
from shared.logging.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/datasets", tags=["datasets"])

settings = get_settings()


class DatasetCreate(BaseModel):
    name: str
    description: str = ""
    class_names: list[str] = ["capitate_stalked", "capitate_sessile", "bulbous", "non_glandular"]
    split_config: dict[str, float] = {"train": 0.7, "val": 0.2, "test": 0.1}


class DatasetSummary(BaseModel):
    id: int
    name: str
    description: str
    num_samples: int
    num_annotated: int
    num_reviewed: int
    class_names: list[str]
    version: str
    status: str
    created_at: float


class SampleSummary(BaseModel):
    id: int
    filename: str
    width: int
    height: int
    quality_score: float
    focus_score: float
    split: str
    num_annotations: int
    annotation_source: str
    created_at: float


@router.get("", response_model=list[DatasetSummary])
async def list_datasets(
    status: str | None = None,
    db: Session = Depends(get_session),
) -> list[DatasetSummary]:
    """List all datasets."""
    query = select(Dataset)
    if status:
        query = query.where(Dataset.status == status)
    datasets = db.exec(query.order_by(Dataset.id.desc())).all()  # type: ignore

    return [
        DatasetSummary(
            id=d.id,
            name=d.name,
            description=d.description,
            num_samples=d.num_samples,
            num_annotated=d.num_annotated,
            num_reviewed=d.num_reviewed,
            class_names=d.class_names,
            version=d.version,
            status=d.status,
            created_at=d.created_at,
        )
        for d in datasets
    ]


@router.post("", response_model=dict)
async def create_dataset(
    request: DatasetCreate,
    db: Session = Depends(get_session),
) -> dict[str, Any]:
    """Create a new dataset container."""
    import json

    # Create storage directory
    dataset_dir = Path(settings.images_dir) / request.name.replace(" ", "_")
    dataset_dir.mkdir(parents=True, exist_ok=True)

    dataset = Dataset(
        name=request.name,
        description=request.description,
        root_path=str(dataset_dir),
        class_names_json=json.dumps(request.class_names),
        split_config_json=json.dumps(request.split_config),
    )
    db.add(dataset)
    db.commit()
    db.refresh(dataset)

    logger.info("Dataset created", dataset_id=dataset.id, name=request.name)
    return dataset.to_dict()


@router.get("/{dataset_id}", response_model=dict)
async def get_dataset(
    dataset_id: int,
    db: Session = Depends(get_session),
) -> dict[str, Any]:
    """Get dataset details."""
    dataset = db.get(Dataset, dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail=f"Dataset {dataset_id} not found")
    return dataset.to_dict()


@router.delete("/{dataset_id}")
async def delete_dataset(
    dataset_id: int,
    delete_files: bool = False,
    db: Session = Depends(get_session),
) -> dict[str, Any]:
    """
    Delete dataset record (and optionally its files).

    Args:
        delete_files: If True, also delete the image files from disk.
    """
    dataset = db.get(Dataset, dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail=f"Dataset {dataset_id} not found")

    if delete_files and dataset.root_path:
        root = Path(dataset.root_path)
        if root.exists():
            shutil.rmtree(root)

    dataset.status = "archived"
    db.add(dataset)
    db.commit()

    return {"deleted": True, "dataset_id": dataset_id}


@router.get("/{dataset_id}/samples", response_model=list[SampleSummary])
async def list_samples(
    dataset_id: int,
    split: str | None = None,
    min_quality: float = 0.0,
    limit: int = 100,
    offset: int = 0,
    db: Session = Depends(get_session),
) -> list[SampleSummary]:
    """List samples in a dataset with optional filters."""
    dataset = db.get(Dataset, dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail=f"Dataset {dataset_id} not found")

    query = select(Sample).where(Sample.dataset_id == dataset_id)
    if split:
        query = query.where(Sample.split == split)
    if min_quality > 0:
        query = query.where(Sample.quality_score >= min_quality)

    query = query.offset(offset).limit(limit)
    samples = db.exec(query).all()

    return [
        SampleSummary(
            id=s.id,
            filename=s.filename,
            width=s.width,
            height=s.height,
            quality_score=s.quality_score,
            focus_score=s.focus_score,
            split=s.split,
            num_annotations=s.num_annotations,
            annotation_source=s.annotation_source,
            created_at=s.created_at,
        )
        for s in samples
    ]


@router.post("/{dataset_id}/upload")
async def upload_images(
    dataset_id: int,
    files: list[UploadFile] = File(...),
    compute_quality: bool = Form(default=True),
    db: Session = Depends(get_session),
) -> dict[str, Any]:
    """
    Upload images to a dataset.

    Streams files to disk, computes quality scores, creates Sample records.
    """
    dataset = db.get(Dataset, dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail=f"Dataset {dataset_id} not found")

    uploaded = 0
    skipped = 0
    errors = []
    dataset_dir = Path(dataset.root_path)

    for upload_file in files:
        if not upload_file.filename:
            continue

        filename = Path(upload_file.filename).name
        dest_path = dataset_dir / filename

        try:
            content = await upload_file.read()
            file_hash = hashlib.sha256(content).hexdigest()

            # Check for duplicate
            existing = db.exec(
                select(Sample).where(
                    Sample.dataset_id == dataset_id,
                    Sample.file_hash == file_hash,
                )
            ).first()

            if existing:
                skipped += 1
                continue

            # Save file
            with open(dest_path, "wb") as f:
                f.write(content)

            # Get image dimensions
            import cv2
            import numpy as np
            arr = np.frombuffer(content, np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            h, w = (img.shape[:2] if img is not None else (0, 0))

            # Compute quality if requested
            focus_score = 0.0
            quality_score = 0.0
            if compute_quality and img is not None:
                try:
                    from focus.metrics.composite import compute_focus_score
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    focus_result = compute_focus_score(img_rgb)
                    focus_score = focus_result.composite
                    quality_score = focus_score
                except Exception:
                    pass

            # Create Sample record
            sample = Sample(
                dataset_id=dataset_id,
                filename=filename,
                file_path=str(dest_path),
                file_hash=file_hash,
                width=w,
                height=h,
                focus_score=focus_score,
                quality_score=quality_score,
                is_usable=quality_score >= 0.2,
            )
            db.add(sample)
            uploaded += 1

        except Exception as e:
            errors.append(f"{filename}: {str(e)}")
            if dest_path.exists():
                dest_path.unlink()

    # Update dataset sample count
    dataset.num_samples = int(
        db.exec(
            select(func.count(Sample.id)).where(Sample.dataset_id == dataset_id)
        ).first() or 0
    )
    dataset.updated_at = time.time()
    db.add(dataset)
    db.commit()

    logger.info(
        "Images uploaded",
        dataset_id=dataset_id,
        uploaded=uploaded,
        skipped=skipped,
    )

    return {
        "uploaded": uploaded,
        "skipped_duplicates": skipped,
        "errors": errors,
        "total_samples": dataset.num_samples,
    }


@router.get("/{dataset_id}/stats")
async def dataset_stats(
    dataset_id: int,
    db: Session = Depends(get_session),
) -> dict[str, Any]:
    """Get detailed dataset statistics (class balance, quality distribution, etc.)."""
    dataset = db.get(Dataset, dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail=f"Dataset {dataset_id} not found")

    samples = db.exec(
        select(Sample).where(Sample.dataset_id == dataset_id)
    ).all()

    if not samples:
        return {"dataset_id": dataset_id, "num_samples": 0}

    quality_scores = [s.quality_score for s in samples]
    split_counts = {}
    for s in samples:
        split_counts[s.split] = split_counts.get(s.split, 0) + 1

    annotations = db.exec(
        select(Annotation).where(
            Annotation.sample_id.in_([s.id for s in samples if s.id])
        )
    ).all()

    class_counts: dict[str, int] = {}
    for ann in annotations:
        cn = ann.class_name or str(ann.class_id)
        class_counts[cn] = class_counts.get(cn, 0) + 1

    return {
        "dataset_id": dataset_id,
        "num_samples": len(samples),
        "num_annotated": sum(1 for s in samples if s.num_annotations > 0),
        "num_reviewed": dataset.num_reviewed,
        "split_distribution": split_counts,
        "quality": {
            "mean": float(np.mean(quality_scores)),
            "std": float(np.std(quality_scores)),
            "min": float(np.min(quality_scores)),
            "max": float(np.max(quality_scores)),
            "usable": sum(1 for s in samples if s.is_usable),
        },
        "class_distribution": class_counts,
        "total_annotations": len(annotations),
        "annotation_sources": {
            src: sum(1 for a in annotations if a.source == src)
            for src in set(a.source for a in annotations)
        },
    }
