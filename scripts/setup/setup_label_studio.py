"""
scripts/setup/setup_label_studio.py

Full Label Studio setup for CTIP:
  1. Creates/updates 3 projects (ctip_combined, ctip_synthetic, cystolith_cannabis)
  2. Imports all images as tasks with local-file URLs
  3. Converts YOLO annotations → Label Studio pre-annotations
  4. Reports task/annotation counts per project

Usage:
    python3 scripts/setup/setup_label_studio.py
    python3 scripts/setup/setup_label_studio.py --dry-run
    python3 scripts/setup/setup_label_studio.py --clear-existing
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

import requests

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_ROOT / "data"
DATASETS_ROOT = DATA_ROOT / "datasets"

LS_HOST = os.environ.get("LABEL_STUDIO_URL", "http://localhost:3005").rstrip("/")
LS_TOKEN = os.environ.get("LABEL_STUDIO_API_KEY", "")

# CTIP class names — must match dataset.yaml order
CTIP_CLASSES = {0: "stalked", 1: "sessile", 2: "bulbous", 3: "non-glandular"}
CYSTOLITH_CLASSES = {0: "non-glandular"}  # cystolith_cannabis only has class 0 = non-glandular

# Label Studio XML config with hotkeys, colors, maturity tags and notes
LABEL_CONFIG = """<View>
  <Image name="image" value="$image" maxWidth="100%" zoomControl="true" zoom="true" rotateControl="true"/>
  <Header value="Trichome Type"/>
  <RectangleLabels name="label" toName="image" showInline="true">
    <Label value="stalked" background="#60a5fa" hotkey="1"/>
    <Label value="sessile" background="#a78bfa" hotkey="2"/>
    <Label value="bulbous" background="#34d399" hotkey="3"/>
    <Label value="non-glandular" background="#f87171" hotkey="4"/>
  </RectangleLabels>
  <Header value="Maturity (optional)"/>
  <Choices name="maturity" toName="image" choice="single" showInline="true">
    <Choice value="clear"/>
    <Choice value="cloudy"/>
    <Choice value="amber"/>
    <Choice value="mixed"/>
  </Choices>
  <Header value="Image Quality"/>
  <Choices name="quality" toName="image" choice="single" showInline="true">
    <Choice value="good"/>
    <Choice value="blurry"/>
    <Choice value="overexposed"/>
    <Choice value="underexposed"/>
  </Choices>
  <TextArea name="notes" toName="image" placeholder="Optional notes (focus, magnification, batch...)"
            editable="true" maxSubmissions="1" rows="2"/>
</View>"""

# Datasets to import
DATASET_CONFIGS = [
    {
        "name": "CTIP Combined",
        "description": "270 images — 200 synthetic + 70 real cannabis microscopy (Zvirin 2025). All 4 CTIP classes. Primary training dataset.",
        "dataset_dir": DATASETS_ROOT / "ctip_combined",
        "class_map": CTIP_CLASSES,
        "splits": ["train", "val", "test"],
    },
    {
        "name": "CTIP Synthetic",
        "description": "200 procedurally generated microscopy images. All 4 CTIP classes. Seed=42, uniform class distribution.",
        "dataset_dir": DATASETS_ROOT / "ctip_synthetic",
        "class_map": CTIP_CLASSES,
        "splits": ["train", "val", "test"],
    },
    {
        "name": "Cystolith Cannabis (Real)",
        "description": "70 real cannabis microscopy images from Zvirin et al. 2025. Class 0 only (non-glandular trichome hairs).",
        "dataset_dir": DATASETS_ROOT / "cystolith_cannabis",
        "class_map": CYSTOLITH_CLASSES,
        "splits": ["train", "val", "test"],
    },
]


# ---------------------------------------------------------------------------
# Label Studio REST helpers
# ---------------------------------------------------------------------------

def _headers() -> dict:
    return {"Authorization": f"Token {LS_TOKEN}", "Content-Type": "application/json"}


def _get(path: str, **kwargs) -> dict | list:
    r = requests.get(f"{LS_HOST}{path}", headers=_headers(), timeout=30, **kwargs)
    r.raise_for_status()
    return r.json()


def _post(path: str, data=None, **kwargs) -> dict | list:
    r = requests.post(f"{LS_HOST}{path}", headers=_headers(), json=data, timeout=120, **kwargs)
    if not r.ok:
        print(f"  POST {path} failed {r.status_code}: {r.text[:300]}")
        r.raise_for_status()
    return r.json() if r.content else {}


def _patch(path: str, data=None) -> dict:
    r = requests.patch(f"{LS_HOST}{path}", headers=_headers(), json=data, timeout=30)
    r.raise_for_status()
    return r.json()


def _delete(path: str) -> None:
    r = requests.delete(f"{LS_HOST}{path}", headers=_headers(), timeout=30)
    r.raise_for_status()


# ---------------------------------------------------------------------------
# YOLO → Label Studio conversion
# ---------------------------------------------------------------------------

def yolo_to_ls_result(label_file: Path, class_map: dict[int, str]) -> list[dict]:
    """
    Convert a YOLO .txt annotation file to Label Studio result format.

    YOLO format: class_id cx cy w h  (normalized 0-1)
    LS format:   x y width height in percentage (0-100), from top-left corner
    """
    if not label_file.exists():
        return []

    results = []
    for line in label_file.read_text().strip().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        class_id = int(parts[0])
        cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])

        class_name = class_map.get(class_id, f"class_{class_id}")

        # Convert center format to top-left format, scale to percentage
        x_pct = (cx - w / 2) * 100
        y_pct = (cy - h / 2) * 100
        w_pct = w * 100
        h_pct = h * 100

        results.append({
            "from_name": "label",
            "to_name": "image",
            "type": "rectanglelabels",
            "value": {
                "x": round(x_pct, 4),
                "y": round(y_pct, 4),
                "width": round(w_pct, 4),
                "height": round(h_pct, 4),
                "rotation": 0,
                "rectanglelabels": [class_name],
            },
        })
    return results


# ---------------------------------------------------------------------------
# Image URL builder
# ---------------------------------------------------------------------------

def image_to_ls_url(image_path: Path) -> str:
    """
    Build a Label Studio local-file URL for an image.

    The container mounts DATA_ROOT at /ctip-data (read-only).
    Label Studio document root = /ctip-data.
    URL format: /data/local-files/?d=<relative path from DATA_ROOT>
    """
    try:
        rel = image_path.relative_to(DATA_ROOT)
    except ValueError:
        # fallback: use full path relative to repo root
        rel = image_path.relative_to(REPO_ROOT / "data")
    return f"/data/local-files/?d={rel.as_posix()}"


# ---------------------------------------------------------------------------
# Project management
# ---------------------------------------------------------------------------

def list_projects() -> dict[str, int]:
    """Return {title: id} for all existing projects."""
    data = _get("/api/projects/")
    return {p["title"]: p["id"] for p in data.get("results", [])}


def create_or_update_project(name: str, description: str) -> int:
    """Create project if it doesn't exist; update label_config if it does."""
    existing = list_projects()
    if name in existing:
        proj_id = existing[name]
        _patch(f"/api/projects/{proj_id}/", {
            "label_config": LABEL_CONFIG,
            "description": description,
        })
        print(f"  Updated existing project [{proj_id}] {name!r}")
        return proj_id

    data = _post("/api/projects/", {
        "title": name,
        "description": description,
        "label_config": LABEL_CONFIG,
    })
    proj_id = int(data["id"])
    print(f"  Created project [{proj_id}] {name!r}")
    return proj_id


def clear_project_tasks(project_id: int) -> None:
    """Delete all tasks in a project (used with --clear-existing)."""
    while True:
        data = _get(f"/api/tasks/?project={project_id}&page_size=100")
        tasks = data.get("tasks", [])
        if not tasks:
            break
        for task in tasks:
            _delete(f"/api/tasks/{task['id']}/")
    print(f"  Cleared all tasks from project {project_id}")


# ---------------------------------------------------------------------------
# Task import
# ---------------------------------------------------------------------------

def build_tasks(
    dataset_dir: Path,
    class_map: dict[int, str],
    splits: list[str],
) -> list[dict]:
    """
    Build Label Studio task payloads from a YOLO dataset directory.

    Each task = {
        "data": {"image": "<ls_url>", "split": "train|val|test", "filename": "<name>"},
        "predictions": [{"model_version": "yolo-preann", "result": [...]}]
    }
    """
    tasks = []
    for split in splits:
        img_dir = dataset_dir / "images" / split
        lbl_dir = dataset_dir / "labels" / split

        if not img_dir.exists():
            continue

        for img_path in sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png")):
            lbl_path = lbl_dir / (img_path.stem + ".txt")
            ls_result = yolo_to_ls_result(lbl_path, class_map)

            task: dict = {
                "data": {
                    "image": image_to_ls_url(img_path),
                    "split": split,
                    "filename": img_path.name,
                },
            }

            if ls_result:
                task["predictions"] = [{
                    "model_version": "yolo-preann-v1",
                    "score": 0.9,
                    "result": ls_result,
                }]

            tasks.append(task)

    return tasks


def import_tasks_batched(project_id: int, tasks: list[dict], batch_size: int = 50) -> int:
    """Import tasks in batches; return total imported count."""
    total = 0
    for i in range(0, len(tasks), batch_size):
        batch = tasks[i:i + batch_size]
        result = _post(f"/api/projects/{project_id}/import", batch)
        imported = result.get("task_count", len(batch))
        total += imported
        print(f"    batch {i//batch_size + 1}: +{imported} tasks (total {total}/{len(tasks)})")
        time.sleep(0.3)  # avoid hammering the API
    return total


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Setup Label Studio for CTIP")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be imported without doing it")
    parser.add_argument("--clear-existing", action="store_true", help="Delete existing tasks before import")
    parser.add_argument("--dataset", help="Only import this dataset (by name fragment, e.g. 'combined')")
    args = parser.parse_args()

    print(f"\nCTIP Label Studio Setup")
    print(f"  Host:     {LS_HOST}")
    print(f"  Data:     {DATA_ROOT}")
    print(f"  Dry-run:  {args.dry_run}")
    print()

    # Health check
    try:
        _get("/health/")
    except Exception:
        try:
            r = requests.get(f"{LS_HOST}/health", timeout=5)
            if r.status_code != 200:
                raise RuntimeError(f"unhealthy: {r.status_code}")
        except Exception as e:
            print(f"ERROR: Label Studio not reachable at {LS_HOST}: {e}")
            sys.exit(1)
    print("  Label Studio: OK\n")

    results_summary = []

    for cfg in DATASET_CONFIGS:
        if args.dataset and args.dataset.lower() not in cfg["name"].lower():
            continue

        print(f"─── {cfg['name']} ───")
        dataset_dir: Path = cfg["dataset_dir"]

        if not dataset_dir.exists():
            print(f"  SKIP: {dataset_dir} not found")
            continue

        # Build tasks
        tasks = build_tasks(dataset_dir, cfg["class_map"], cfg["splits"])
        ann_count = sum(1 for t in tasks if t.get("predictions"))
        print(f"  Tasks: {len(tasks)} | With pre-annotations: {ann_count}")

        if args.dry_run:
            for split in cfg["splits"]:
                n = sum(1 for t in tasks if t["data"].get("split") == split)
                print(f"    {split}: {n} tasks")
            print()
            continue

        # Create / update project
        proj_id = create_or_update_project(cfg["name"], cfg["description"])

        # Optionally clear existing tasks
        if args.clear_existing:
            clear_project_tasks(proj_id)

        # Import
        if tasks:
            print(f"  Importing {len(tasks)} tasks...")
            imported = import_tasks_batched(proj_id, tasks)
            print(f"  Done: {imported} tasks imported")
        else:
            print("  No images found — skipping import")

        results_summary.append({
            "project": cfg["name"],
            "project_id": proj_id,
            "tasks": len(tasks),
            "pre_annotated": ann_count,
        })
        print()

    # Summary
    if not args.dry_run and results_summary:
        print("=" * 50)
        print("SUMMARY")
        print("=" * 50)
        for r in results_summary:
            print(f"  [{r['project_id']}] {r['project']}")
            print(f"       Tasks: {r['tasks']} | Pre-annotated: {r['pre_annotated']}")
        print()
        print(f"  Open Label Studio: {LS_HOST}")
        print()


if __name__ == "__main__":
    main()
