"""Inputs for jobs: safe unpacking of dataset archives into a content-addressed cache."""

from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

import yaml

ALLOWED = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp", ".txt", ".yaml", ".yml", ".json", ".csv", ".md"}
MAX_UNPACKED = 8 * 1024**3


def safe_extract(archive: Path, dest: Path) -> Path:
    """Unzip only image/label/config files, refusing paths that leave `dest` (zip slip) and zip bombs."""
    dest = dest.resolve()
    tmp = dest.with_name(dest.name + ".part")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    total = 0
    with zipfile.ZipFile(archive) as z:
        for info in z.infolist():
            if info.is_dir():
                continue
            name = Path(info.filename)
            if name.is_absolute() or ".." in name.parts:
                raise ValueError(f"unsafe path in archive: {info.filename}")
            if name.suffix.lower() not in ALLOWED or name.name.startswith("."):
                continue
            total += info.file_size
            if total > MAX_UNPACKED:
                raise ValueError("archive unpacks to more than 8 GB")
            target = (tmp / name).resolve()
            if tmp not in target.parents:
                raise ValueError(f"unsafe path in archive: {info.filename}")
            target.parent.mkdir(parents=True, exist_ok=True)
            with z.open(info) as src, target.open("wb") as out:
                shutil.copyfileobj(src, out)
    shutil.rmtree(dest, ignore_errors=True)
    tmp.rename(dest)
    return dest


def dataset_yaml(root: Path, out: Path) -> Path:
    """Find dataset.yaml (or data.yaml) in an unpacked dataset and write a copy whose `path` points at `root`."""
    found = next((p for name in ("dataset.yaml", "data.yaml") for p in sorted(root.rglob(name))), None)
    if found is None:
        raise FileNotFoundError("dataset archive has no dataset.yaml / data.yaml")
    data = yaml.safe_load(found.read_text()) or {}
    data["path"] = str(found.parent.resolve())
    for key in ("train", "val", "test"):                 # keep relative entries relative to the new root
        if isinstance(data.get(key), str) and Path(data[key]).is_absolute():
            data[key] = Path(data[key]).name if (found.parent / Path(data[key]).name).exists() else data[key]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(data))
    return out


def cache_size(cache: Path) -> int:
    return sum(p.stat().st_size for p in cache.rglob("*") if p.is_file()) if cache.exists() else 0


def trim_cache(cache: Path, max_bytes: int) -> None:
    """Drop the least recently used entries until the cache fits."""
    if not cache.exists():
        return
    entries = sorted((p for p in cache.iterdir()), key=lambda p: p.stat().st_atime)
    while cache_size(cache) > max_bytes and entries:
        victim = entries.pop(0)
        shutil.rmtree(victim, ignore_errors=True) if victim.is_dir() else victim.unlink(missing_ok=True)
