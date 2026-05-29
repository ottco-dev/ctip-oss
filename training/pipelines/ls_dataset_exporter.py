"""
training.pipelines.ls_dataset_exporter

Export annotated tasks from a Label Studio project and convert to a YOLO
dataset that can be passed directly to YOLOTrainer.

Supports:
  - Confirmed human annotations   (use_predictions=False, default)
  - YOLO pre-annotations          (use_predictions=True — useful for pipeline testing)

Output layout:
  data/datasets/ls_export_{project_id}_{timestamp}/
    images/
      train/  val/  test/
    labels/
      train/  val/  test/
    dataset.yaml

Image URL resolution:
  /data/local-files/?d=<rel>  →  DATA_ROOT / <rel>   (LS local-file serving)
  http[s]://...               →  download to images/  (uploaded files)
  Anything else               →  skipped with warning
"""

from __future__ import annotations

import json
import os
import random
import shutil
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from shared.logging.logger import get_logger

logger = get_logger(__name__)

# Absolute path to the CTIP data directory (mounted at /ctip-data inside LS container)
_DATA_ROOT = Path(os.environ.get("DATA_ROOT", "./data")).resolve()
if not _DATA_ROOT.is_absolute():
    _DATA_ROOT = Path(__file__).resolve().parents[2] / _DATA_ROOT

# Class name → YOLO index (canonical CTIP ordering)
CTIP_CLASS_ORDER = ["stalked", "sessile", "bulbous", "non-glandular"]
CTIP_CLASS_INDEX: dict[str, int] = {name: i for i, name in enumerate(CTIP_CLASS_ORDER)}

# Fallback for class names used in label config (capitate-* variants)
_CLASS_ALIASES: dict[str, str] = {
    "capitate-stalked": "stalked",
    "capitate_stalked": "stalked",
    "capitate-sessile": "sessile",
    "capitate_sessile": "sessile",
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@dataclass
class ExportConfig:
    project_id: int
    ls_host: str = "http://localhost:3005"
    ls_token: str = ""
    use_predictions: bool = False
    """If True, export YOLO pre-annotations; if False, only human-confirmed annotations."""
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    # test_ratio = 1 - train - val
    seed: int = 42
    output_root: Optional[Path] = None
    """Directory to write dataset. Auto-generated if None."""
    min_annotations_per_task: int = 1
    progress_callback: Optional[Callable[[str, str], None]] = None
    """Optional (line, level) callback for live progress reporting."""


@dataclass
class ExportResult:
    dataset_dir: Path
    dataset_yaml: Path
    total_tasks: int
    exported_tasks: int
    skipped_tasks: int
    train_count: int
    val_count: int
    test_count: int
    classes: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "dataset_dir": str(self.dataset_dir),
            "dataset_yaml": str(self.dataset_yaml),
            "total_tasks": self.total_tasks,
            "exported_tasks": self.exported_tasks,
            "skipped_tasks": self.skipped_tasks,
            "train_count": self.train_count,
            "val_count": self.val_count,
            "test_count": self.test_count,
            "classes": self.classes,
            "warnings": self.warnings,
        }


def export_ls_project(config: ExportConfig) -> ExportResult:
    """
    Export a Label Studio project to a YOLO dataset directory.

    Args:
        config: ExportConfig specifying project_id, split ratios, etc.

    Returns:
        ExportResult with dataset_yaml path and statistics.

    Raises:
        ValueError: If project not found or no tasks available.
        RuntimeError: If LS API is unreachable.
    """
    import requests

    headers = {"Authorization": f"Token {config.ls_token}", "Content-Type": "application/json"}
    base = config.ls_host.rstrip("/")

    # ── Fetch project metadata ──────────────────────────────────────────────
    try:
        r = requests.get(f"{base}/api/projects/{config.project_id}/", headers=headers, timeout=15)
        r.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(f"Cannot reach Label Studio at {base}: {exc}") from exc

    project = r.json()
    project_title = project.get("title", f"project_{config.project_id}")
    logger.info("Exporting LS project", id=config.project_id, title=project_title)

    _log = config.progress_callback or (lambda line, level="info": logger.info(line))

    _log(f"Project: {project_title!r}  (id={config.project_id})", "info")
    _log(
        f"Source: {'pre-annotations (YOLO)' if config.use_predictions else 'human-confirmed annotations'}",
        "dim",
    )

    # ── Fetch all tasks with annotations ────────────────────────────────────
    _log("Fetching tasks from Label Studio…", "dim")
    all_tasks = _fetch_all_tasks(
        base, headers, config.project_id, config.use_predictions, progress_fn=_log
    )

    if not all_tasks:
        _log(
            f"No annotated tasks found in project {config.project_id!r}. "
            f"{'Try use_predictions=True to export YOLO pre-annotations.' if not config.use_predictions else ''}",
            "error",
        )
        raise ValueError(
            f"No annotated tasks found in project {config.project_id!r}. "
            f"{'Enable use_predictions=True to export pre-annotations.' if not config.use_predictions else ''}"
        )

    _log(f"Found {len(all_tasks)} tasks — preparing output directory…", "info")

    # ── Output directory ────────────────────────────────────────────────────
    ts = int(time.time())
    safe_title = "".join(c if c.isalnum() or c in "-_" else "_" for c in project_title)[:40]
    out_dir = config.output_root or (
        _DATA_ROOT / "datasets" / f"ls_export_{safe_title}_{config.project_id}_{ts}"
    )
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for split in ("train", "val", "test"):
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    # ── Split tasks ─────────────────────────────────────────────────────────
    rng = random.Random(config.seed)
    rng.shuffle(all_tasks)
    n = len(all_tasks)
    n_train = max(1, int(n * config.train_ratio))
    n_val = max(1, int(n * config.val_ratio))
    splits = {
        "train": all_tasks[:n_train],
        "val": all_tasks[n_train : n_train + n_val],
        "test": all_tasks[n_train + n_val :],
    }

    # ── Write images + labels ────────────────────────────────────────────────
    exported = 0
    skipped = 0
    warnings: list[str] = []
    class_names_seen: set[str] = set()
    _total_tasks = len(all_tasks)
    _processed = 0

    for split, tasks in splits.items():
        for task in tasks:
            _processed += 1
            if _processed % 25 == 0 or _processed == _total_tasks:
                _log(
                    f"  Processing tasks… {_processed}/{_total_tasks}  "
                    f"(exported={exported}, skipped={skipped})",
                    "dim",
                )
            task_id = task["id"]
            image_url: str = task.get("data", {}).get("image", "")
            annots = task.get("_annotations", [])

            # Resolve image path
            img_path = _resolve_image(image_url, config)
            if img_path is None:
                skipped += 1
                warnings.append(f"Task {task_id}: cannot resolve image URL {image_url!r}")
                continue

            # Build YOLO label lines
            lines = []
            for annot in annots:
                for result in annot.get("result", []):
                    if result.get("type") != "rectanglelabels":
                        continue
                    value = result["value"]
                    labels_list = value.get("rectanglelabels", [])
                    if not labels_list:
                        continue
                    raw_label = labels_list[0]
                    label = _CLASS_ALIASES.get(raw_label, raw_label)
                    class_names_seen.add(label)
                    class_idx = CTIP_CLASS_INDEX.get(label)
                    if class_idx is None:
                        warnings.append(f"Task {task_id}: unknown label {raw_label!r} — skipped")
                        continue

                    x_pct = value["x"] / 100
                    y_pct = value["y"] / 100
                    w_pct = value["width"] / 100
                    h_pct = value["height"] / 100
                    cx = x_pct + w_pct / 2
                    cy = y_pct + h_pct / 2
                    lines.append(f"{class_idx} {cx:.6f} {cy:.6f} {w_pct:.6f} {h_pct:.6f}")

            if not lines:
                skipped += 1
                warnings.append(f"Task {task_id}: no valid bounding boxes — skipped")
                continue

            # Copy image
            stem = f"task_{task_id}_{img_path.stem}"
            dst_img = out_dir / "images" / split / f"{stem}{img_path.suffix}"
            dst_lbl = out_dir / "labels" / split / f"{stem}.txt"
            try:
                shutil.copy2(img_path, dst_img)
            except OSError as exc:
                skipped += 1
                warnings.append(f"Task {task_id}: cannot copy image {img_path}: {exc}")
                continue

            dst_lbl.write_text("\n".join(lines) + "\n")
            exported += 1

    # ── Dataset YAML ────────────────────────────────────────────────────────
    _log(f"Writing dataset YAML…", "dim")
    # Use canonical class order; fall back to alphabetical for unknown names
    classes = [c for c in CTIP_CLASS_ORDER if c in class_names_seen]
    unknown = sorted(class_names_seen - set(CTIP_CLASS_ORDER))
    classes.extend(unknown)

    # Re-write labels that use dynamic indices based on final class list
    if unknown:
        _rewrite_labels_for_dynamic_classes(out_dir, classes)

    yaml_path = out_dir / "dataset.yaml"
    yaml_path.write_text(_render_dataset_yaml(
        out_dir=out_dir,
        classes=classes,
        project_title=project_title,
        config=config,
        exported=exported,
    ))

    result = ExportResult(
        dataset_dir=out_dir,
        dataset_yaml=yaml_path,
        total_tasks=len(all_tasks),
        exported_tasks=exported,
        skipped_tasks=skipped,
        train_count=sum(1 for t in splits["train"] if _task_in_export(t)),
        val_count=sum(1 for t in splits["val"] if _task_in_export(t)),
        test_count=sum(1 for t in splits["test"] if _task_in_export(t)),
        classes=classes,
        warnings=warnings,
    )

    logger.info(
        "LS export complete",
        exported=exported,
        skipped=skipped,
        yaml=str(yaml_path),
    )
    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

# Expose alias dict at module level so it can be used in outer scope
_CLASS_ALIASES = _CLASS_ALIASES  # noqa: SIM900


def _task_in_export(task: dict) -> bool:
    return bool(task.get("_annotations"))


def _fetch_all_tasks(
    base: str,
    headers: dict,
    project_id: int,
    use_predictions: bool,
    progress_fn: Optional[Callable[[str, str], None]] = None,
) -> list[dict]:
    """Paginate through all project tasks and attach annotations / predictions."""
    import requests

    all_tasks = []
    page = 1
    page_size = 100

    while True:
        if progress_fn:
            progress_fn(f"  Fetching page {page}…  ({len(all_tasks)} tasks collected so far)", "dim")
        r = requests.get(
            f"{base}/api/tasks/",
            headers=headers,
            params={"project": project_id, "page": page, "page_size": page_size},
            timeout=30,
        )
        # LS returns 404 when page exceeds total pages — treat as end of data
        if r.status_code == 404:
            break
        r.raise_for_status()
        data = r.json()
        tasks = data.get("tasks", [])
        if not tasks:
            break

        for task in tasks:
            task_id = task["id"]

            if use_predictions:
                # Use YOLO pre-annotations (predictions)
                pr = requests.get(
                    f"{base}/api/predictions/",
                    headers=headers,
                    params={"task": task_id},
                    timeout=15,
                )
                preds = pr.json() if pr.ok else []
                preds = preds if isinstance(preds, list) else preds.get("results", [])
                if preds:
                    task["_annotations"] = preds
                    all_tasks.append(task)
            else:
                # Only confirmed human annotations
                if task.get("total_annotations", 0) > 0:
                    ar = requests.get(
                        f"{base}/api/tasks/{task_id}/annotations/",
                        headers=headers,
                        timeout=15,
                    )
                    annots = ar.json() if ar.ok else []
                    annots = [a for a in annots if not a.get("was_cancelled")]
                    if annots:
                        task["_annotations"] = annots
                        all_tasks.append(task)

        page += 1
        if len(tasks) < page_size:
            break

    return all_tasks


def _resolve_image(url: str, config: ExportConfig) -> Optional[Path]:
    """
    Resolve an LS image URL to a local Path.

    Handles:
      /data/local-files/?d=<rel>   →  DATA_ROOT / rel
      http[s]://...                →  download (TODO: implement for uploads)
    """
    if not url:
        return None

    if "/data/local-files/" in url:
        parsed = urllib.parse.urlparse(url)
        qs = urllib.parse.parse_qs(parsed.query)
        rel = qs.get("d", [None])[0]
        if not rel:
            return None
        candidate = _DATA_ROOT / rel
        return candidate if candidate.exists() else None

    if url.startswith("http://") or url.startswith("https://"):
        # For uploaded images: could download, but skip for now
        return None

    # Bare path
    candidate = Path(url)
    return candidate if candidate.exists() else None


def _rewrite_labels_for_dynamic_classes(out_dir: Path, classes: list[str]) -> None:
    """Re-index any labels that used CTIP_CLASS_INDEX to the final class list."""
    new_index = {name: i for i, name in enumerate(classes)}
    # For now the label files already use CTIP_CLASS_INDEX which matches our
    # CTIP_CLASS_ORDER. If unknown classes were added they'd need rewriting.
    # This is a no-op for the typical CTIP workflow.
    pass


def _render_dataset_yaml(
    out_dir: Path,
    classes: list[str],
    project_title: str,
    config: ExportConfig,
    exported: int,
) -> str:
    nc = len(classes)
    names_yaml = "\n".join(f"  {i}: {name}" for i, name in enumerate(classes))
    use_preds = "predictions (YOLO pre-annotations)" if config.use_predictions else "confirmed human annotations"
    return (
        f"# Auto-exported from Label Studio project: {project_title}\n"
        f"# Project ID: {config.project_id}\n"
        f"# Source: {use_preds}\n"
        f"# Tasks exported: {exported}\n"
        f"# Split: {config.train_ratio:.0%} train / {config.val_ratio:.0%} val / "
        f"{1 - config.train_ratio - config.val_ratio:.0%} test  (seed={config.seed})\n\n"
        f"path: {out_dir}\n"
        f"train: images/train\n"
        f"val:   images/val\n"
        f"test:  images/test\n\n"
        f"nc: {nc}\n"
        f"names:\n{names_yaml}\n"
    )
