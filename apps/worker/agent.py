"""The agent loop: lease → fetch inputs → run in a subprocess → heartbeats, checkpoints, yield/cancel → finish."""

from __future__ import annotations

import json
import logging
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import psutil

from apps.worker import hardware
from apps.worker.client import Coordinator, CoordinatorError
from apps.worker.config import WorkerConfig
from apps.worker.datasets import dataset_yaml, safe_extract, trim_cache
from shared.compute.schemas import (
    Backend,
    Capabilities,
    Finish,
    Heartbeat,
    JobProgress,
    Lease,
    LeaseRequest,
)

log = logging.getLogger("ctip-worker")
IDLE_POLL_S = 30


def device_for(caps: Capabilities, distributed: bool) -> str:
    if caps.backend in (Backend.CUDA, Backend.ROCM):
        return ",".join(str(i) for i in range(caps.gpus)) if distributed and caps.gpus > 1 else "0"
    return "mps" if caps.backend == Backend.MPS else "cpu"


class Agent:
    def __init__(self, cfg: WorkerConfig, coordinator: Coordinator | None = None) -> None:
        if not (cfg.server and cfg.token):
            raise ValueError("not enrolled - run: ctip-worker enroll --server https://... --token ctipe_...")
        self.cfg = cfg
        self.api = coordinator or Coordinator(cfg.server, cfg.token, cfg.ca_bundle)
        self.caps = hardware.detect(cfg.limits)
        self.stop = threading.Event()
        self._job: Lease | None = None
        self._proc: subprocess.Popen | None = None
        self._cancel = threading.Event()
        self._lost = threading.Event()
        self._yield = threading.Event()
        self._busy_since: float | None = None
        self._lock = threading.Lock()

    # ── heartbeats (own thread, also while idle) ──────────────────────────────
    def _progress(self) -> list[JobProgress]:
        with self._lock:
            job = self._job
        if job is None:
            return []
        p = self._read(self._workdir(job) / "progress.json")
        return [JobProgress(job_id=job.job_id, progress=p.get("progress", 0.0), epoch=p.get("epoch"),
                            metrics=p.get("metrics", {}), message=p.get("message", ""))]

    def _own_pids(self) -> set[int]:
        if self._proc is None:
            return set()
        try:
            parent = psutil.Process(self._proc.pid)
            return {parent.pid, *(c.pid for c in parent.children(recursive=True))}
        except psutil.Error:
            return set()

    def _heartbeat_loop(self) -> None:
        while not self.stop.wait(20):
            try:
                free = hardware.free(self.caps, self.cfg.limits, self._own_pids())
                resp = self.api.heartbeat(Heartbeat(free=free, jobs=self._progress()))
                with self._lock:
                    if self._job and self._job.job_id in resp.lost:
                        self._lost.set()
                    elif self._job and self._job.job_id in resp.cancel:
                        self._cancel.set()
                self._check_busy(free.busy_by_others)
            except Exception as e:                       # the heartbeat thread must never die
                log.warning("heartbeat failed: %s", e)

    def _check_busy(self, busy: bool) -> None:
        """Automatic load adaptation: give the machine back when someone else needs the GPU for a while."""
        if not (busy and self.cfg.limits.yield_when_busy and self._job):
            self._busy_since = None
            return
        self._busy_since = self._busy_since or time.time()
        if time.time() - self._busy_since >= self.cfg.limits.busy_grace_s and not self._yield.is_set():
            log.info("GPU busy with other programs for %ss - yielding the job", self.cfg.limits.busy_grace_s)
            self._yield.set()

    # ── jobs ──────────────────────────────────────────────────────────────────
    def _workdir(self, job: Lease) -> Path:
        return self.cfg.workdir / "jobs" / job.job_id

    @staticmethod
    def _read(path: Path) -> dict:
        try:
            return json.loads(path.read_text())
        except (OSError, ValueError):
            return {}

    def _fetch(self, artifact_id: str, name: str) -> Path:
        cache = self.cfg.workdir / "cache"
        dest = cache / artifact_id / name
        if not dest.exists():
            trim_cache(cache, int(self.cfg.limits.cache_gb * 1024**3))
            self.api.download(artifact_id, dest)
        return dest

    def _inputs(self, job: Lease, wd: Path) -> dict:
        spec = job.spec
        inputs: dict[str, str] = {}
        if "dataset" in spec:
            archive = self._fetch(spec["dataset"], "dataset.zip")
            unpacked = archive.parent / "unpacked"
            if not unpacked.exists():
                safe_extract(archive, unpacked)
            inputs["data_yaml"] = str(dataset_yaml(unpacked, wd / "dataset.yaml"))
        if "model_artifact" in spec:
            inputs["model"] = str(self._fetch(spec["model_artifact"], "model.pt"))
        if job.checkpoint:
            ck = wd / "resume_last.pt"
            self.api.download(job.checkpoint, ck)
            inputs["checkpoint"] = str(ck)
        return inputs

    @staticmethod
    def _oom_count(wd: Path) -> int:
        """How often this job already ran out of memory here (marker lines appended by _finish)."""
        f = wd / "oom.log"
        return len(f.read_text().splitlines()) if f.exists() else 0

    def _batch(self, job: Lease, wd: Path) -> int | float:
        """Batch size, halved for every earlier out-of-memory attempt on this machine (automatic load adaptation)."""
        shrink = 0.5 ** self._oom_count(wd)
        if job.spec.get("batch"):
            return max(1, int(int(job.spec["batch"]) * shrink))
        if self.caps.backend in (Backend.CUDA, Backend.ROCM):
            return round(float(self.cfg.limits.max_vram_fraction) * shrink, 3)   # AutoBatch target share of VRAM
        return max(1, int((4 if self.caps.backend == Backend.MPS else 2) * shrink))

    def _start(self, job: Lease, wd: Path) -> subprocess.Popen:
        threads = self.cfg.limits.threads()
        (wd / "job.json").write_text(json.dumps({"job_id": job.job_id, "spec": job.spec, "inputs": self._inputs(job, wd),
                                                 "device": device_for(self.caps, bool(job.spec.get("distributed"))),
                                                 "batch": self._batch(job, wd), "threads": threads,
                                                 "loader_workers": 0 if self._oom_count(wd) else min(threads, 2)}))
        env = dict(os.environ, OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads), PYTHONUNBUFFERED="1",
                   YOLO_OFFLINE="false", CTIP_JOB_ID=job.job_id)
        logf = (wd / "job.log").open("ab")
        kw: dict = {}
        if sys.platform == "win32":
            kw["creationflags"] = subprocess.BELOW_NORMAL_PRIORITY_CLASS
        else:
            kw["preexec_fn"] = lambda: os.nice(10)        # the owner's programs come first
        root = Path(__file__).resolve().parents[2]
        return subprocess.Popen([sys.executable, "-m", "apps.worker.run_job", str(wd)], cwd=root, env=env,
                                stdout=logf, stderr=subprocess.STDOUT, **kw)

    def _terminate(self) -> None:
        p = self._proc
        if p is None or p.poll() is not None:
            return
        try:
            for c in psutil.Process(p.pid).children(recursive=True):
                c.terminate()
        except psutil.Error:
            pass
        p.send_signal(signal.SIGTERM) if sys.platform != "win32" else p.terminate()
        try:
            p.wait(30)
        except subprocess.TimeoutExpired:
            p.kill()

    def _upload_checkpoint(self, job: Lease, wd: Path) -> None:
        last = wd / "runs" / "train" / "weights" / "last.pt"
        if last.exists():
            snap = wd / "checkpoint_upload.pt"
            shutil.copy2(last, snap)                     # training may overwrite last.pt while we upload
            self.api.upload(job.job_id, snap, "checkpoint")
            snap.unlink(missing_ok=True)

    def run_job(self, job: Lease) -> str:
        wd = self._workdir(job)
        wd.mkdir(parents=True, exist_ok=True)
        log.info("job %s (%s), attempt %s%s", job.job_id[:8], job.spec["kind"], job.attempt,
                 " - resuming from checkpoint" if job.checkpoint else "")
        self._cancel.clear()
        self._lost.clear()
        self._yield.clear()
        with self._lock:
            self._job = job
        try:
            self._proc = self._start(job, wd)
        except Exception as e:
            self.api.finish(job.job_id, Finish(status="failed", message=f"could not prepare the job: {e}"[:2000], retryable=True))
            with self._lock:
                self._job = None
            return "failed"
        every = int(job.spec.get("checkpoint_every", 5))
        uploaded_epoch = 0
        low_ram_since: float | None = None
        try:
            while self._proc.poll() is None:
                time.sleep(5)
                ep = self._read(wd / "progress.json").get("epoch") or 0
                if job.spec["kind"] == "yolo_train" and ep and ep - uploaded_epoch >= every:
                    try:
                        self._upload_checkpoint(job, wd)
                        uploaded_epoch = ep
                    except CoordinatorError as e:
                        log.warning("checkpoint upload failed: %s", e)
                if psutil.virtual_memory().available / 2**30 < self.cfg.limits.min_free_ram_gb:
                    low_ram_since = low_ram_since or time.time()
                    if time.time() - low_ram_since >= 20:   # protect the owner's machine from thrashing
                        log.warning("job %s: free RAM below %.1f GB for 20 s - stopping it", job.job_id[:8],
                                    self.cfg.limits.min_free_ram_gb)
                        self._terminate()
                        return self._finish(job, wd, -9)
                else:
                    low_ram_since = None
                if self._lost.is_set():
                    log.warning("job %s: lease lost (machine was unresponsive?) - stopping, the coordinator requeued it", job.job_id[:8])
                    self._terminate()
                    return "lost"
                if self._cancel.is_set():
                    self._terminate()
                    self._report(self.api.cancelled, job.job_id)
                    return "cancelled"
                if self._yield.is_set() and (job.spec["kind"] != "yolo_train" or ep > uploaded_epoch or ep == 0):
                    self._terminate()
                    if job.spec["kind"] == "yolo_train":
                        self._upload_checkpoint(job, wd)
                    self._report(self.api.finish, job.job_id, Finish(status="yielded", message="owner needed the GPU"))
                    return "yielded"
            return self._finish(job, wd, self._proc.returncode)
        finally:
            self._terminate()
            with self._lock:
                self._job, self._proc = None, None

    @staticmethod
    def _report(fn, *args) -> None:
        """Reports after the fact may meet a requeued job (409): log, never crash."""
        try:
            fn(*args)
        except CoordinatorError as e:
            log.warning("coordinator: %s", e)

    def _finish(self, job: Lease, wd: Path, code: int) -> str:
        result = self._read(wd / "result.json")
        if code == 0 and result:
            for kind, key in (("model", "best"), ("result", "results")):
                p = result.get("files", {}).get(key)
                if p and Path(p).exists():
                    self.api.upload(job.job_id, Path(p), kind)
            self.api.finish(job.job_id, Finish(status="completed", metrics=result.get("metrics", {}), message=result.get("message", "")))
            log.info("job %s completed: %s", job.job_id[:8], result.get("metrics"))
            return "completed"
        tail = ""
        logf = wd / "job.log"
        if logf.exists():
            tail = logf.read_text(errors="replace")[-1500:]
            try:
                self.api.upload(job.job_id, logf, "log")
            except CoordinatorError:
                pass
        killed = code in (-9, 137)                       # SIGKILL: the kernel's OOM killer (system RAM)
        oom = killed or "out of memory" in tail.lower()
        if oom:
            with (wd / "oom.log").open("a") as fh:
                fh.write(f"{time.time()} exit {code}\n")
        why = " (killed - out of system memory)" if killed else " (out of memory)" if oom else ""
        self.api.finish(job.job_id, Finish(status="failed", message=f"exit {code}{why}; retry uses half the batch\n{tail}"
                                           if oom else f"exit {code}\n{tail}", retryable=True))
        log.warning("job %s failed (exit %s)", job.job_id[:8], code)
        return "failed"

    # ── main loop ─────────────────────────────────────────────────────────────
    def run(self, once: bool = False) -> None:
        log.info("ctip-worker on %s (%s, %s GPU, %.1f GB VRAM) -> %s", self.caps.device, self.caps.backend.value,
                 self.caps.gpus, self.caps.vram_gb, self.cfg.server)
        self.api.heartbeat(Heartbeat(capabilities=self.caps, free=hardware.free(self.caps, self.cfg.limits)))
        threading.Thread(target=self._heartbeat_loop, daemon=True).start()
        while not self.stop.is_set():
            free = hardware.free(self.caps, self.cfg.limits)
            job = None
            if free.busy_by_others:
                log.debug("GPU busy with other programs - not taking jobs")
            else:
                try:
                    job = self.api.lease(LeaseRequest(free=free))
                except CoordinatorError as e:
                    log.warning("lease failed: %s", e)
            if job is not None:
                try:
                    self.run_job(job)
                except Exception as e:                   # one broken job must not stop the agent
                    log.exception("job %s aborted: %s", job.job_id[:8], e)
                if once:
                    return
                continue
            if once:
                return
            self.stop.wait(IDLE_POLL_S)
