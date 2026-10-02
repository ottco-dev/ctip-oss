"""Agent settings and limits, stored in the user's config dir with owner-only permissions."""

from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path


def config_dir() -> Path:
    if sys.platform == "win32":
        base = Path(os.environ.get("APPDATA", Path.home()))
    else:
        base = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    return base / "ctip-worker"


@dataclass
class Limits:
    max_vram_fraction: float = 0.80   # share of the GPU memory a job may use (AutoBatch target)
    reserve_vram_gb: float = 1.0      # always left free for the desktop / other programs
    max_ram_gb: float = 0.0           # 0 = no extra cap beyond the job's own needs
    min_free_ram_gb: float = 1.0      # stop a job when the machine's free RAM stays below this (no swap storms)
    cpu_threads: int = 0              # 0 = half of the cores
    yield_when_busy: bool = True      # hand a job back (with checkpoint) when someone else needs the GPU
    busy_gpu_util: int = 30           # % GPU utilisation by OTHER processes that counts as busy
    busy_grace_s: int = 60            # how long it must stay busy before yielding
    cache_gb: float = 20.0            # max size of downloaded datasets/models
    kinds: list[str] = field(default_factory=lambda: ["yolo_train", "yolo_eval", "detection_benchmark"])

    def threads(self) -> int:
        return self.cpu_threads or max(1, (os.cpu_count() or 2) // 2)


@dataclass
class WorkerConfig:
    server: str = ""
    worker_id: str = ""
    token: str = ""
    name: str = ""
    ca_bundle: str = ""               # optional: pin a CA / self-signed certificate
    work_dir: str = ""
    limits: Limits = field(default_factory=Limits)

    @property
    def workdir(self) -> Path:
        return Path(self.work_dir) if self.work_dir else config_dir() / "work"

    @classmethod
    def load(cls, path: Path | None = None) -> WorkerConfig:
        path = path or config_dir() / "config.json"
        if not path.exists():
            return cls()
        data = json.loads(path.read_text())
        lim = data.pop("limits", {})
        known = {f.name for f in fields(Limits)}
        cfg = cls(**{k: v for k, v in data.items() if k in {f.name for f in fields(cls)}})
        cfg.limits = Limits(**{k: v for k, v in lim.items() if k in known})
        return cfg

    def save(self, path: Path | None = None) -> Path:
        path = path or config_dir() / "config.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)   # the token is a secret
        with os.fdopen(fd, "w") as fh:
            json.dump(asdict(self), fh, indent=1)
        os.replace(tmp, path)
        return path
