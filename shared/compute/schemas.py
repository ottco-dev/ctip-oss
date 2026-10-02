"""Job specs, worker capabilities and protocol messages. Validated on both sides; the agent runs only these kinds."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

PROTOCOL = 1


class Backend(StrEnum):
    CUDA = "cuda"
    ROCM = "rocm"
    MPS = "mps"
    CPU = "cpu"


class JobKind(StrEnum):
    YOLO_TRAIN = "yolo_train"
    YOLO_EVAL = "yolo_eval"
    BENCHMARK = "detection_benchmark"


class JobStatus(StrEnum):
    QUEUED = "queued"
    LEASED = "leased"            # handed to a worker, not started yet
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


ACTIVE = (JobStatus.LEASED, JobStatus.RUNNING)


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ── job specs (one per kind) ──────────────────────────────────────────────────

YoloVariant = Literal["yolo11n", "yolo11s", "yolo11m", "yolo11l", "yolo11x"]


class YoloTrainSpec(Strict):
    kind: Literal["yolo_train"] = "yolo_train"
    dataset: str = Field(description="artifact id of a zipped YOLO dataset with dataset.yaml at its root")
    model: YoloVariant = "yolo11s"
    epochs: int = Field(100, ge=1, le=1000)
    imgsz: Literal[320, 416, 512, 640, 768, 960, 1024, 1280, 1536] = 1280
    batch: int = Field(0, ge=0, le=128, description="0 = automatic from the worker's free VRAM")
    patience: int = Field(50, ge=0, le=1000)
    seed: int = 42
    degrees: float = Field(90.0, ge=0, le=180)
    mosaic: float = Field(1.0, ge=0, le=1)
    distributed: bool = Field(False, description="DDP over all GPUs of ONE worker (needs gpus >= 2)")
    checkpoint_every: int = Field(5, ge=1, le=100, description="upload last.pt every N epochs (for resuming)")


class YoloEvalSpec(Strict):
    kind: Literal["yolo_eval"] = "yolo_eval"
    dataset: str
    model_artifact: str = Field(description="artifact id of the weights (.pt) to evaluate")
    split: Literal["val", "test"] = "test"
    imgsz: Literal[320, 416, 512, 640, 768, 960, 1024, 1280, 1536] = 1280


class BenchmarkSpec(Strict):
    kind: Literal["detection_benchmark"] = "detection_benchmark"
    model: YoloVariant = "yolo11s"
    imgsz: Literal[640, 1280] = 1280
    runs: int = Field(20, ge=3, le=200)


JobSpec = Annotated[YoloTrainSpec | YoloEvalSpec | BenchmarkSpec, Field(discriminator="kind")]


class Requirements(Strict):
    backends: list[Backend] = Field(default_factory=lambda: [Backend.CUDA, Backend.ROCM, Backend.MPS, Backend.CPU])
    min_vram_gb: float = Field(0.0, ge=0)
    min_ram_gb: float = Field(2.0, ge=0)
    gpus: int = Field(1, ge=1, le=16, description=">1 only for distributed jobs")


# ── worker side ───────────────────────────────────────────────────────────────

class Capabilities(Strict):
    backend: Backend
    device: str = Field(max_length=200)
    gpus: int = Field(0, ge=0, le=64)
    vram_gb: float = Field(0.0, ge=0)
    ram_gb: float = Field(0.0, ge=0)
    cpu_cores: int = Field(1, ge=1)
    torch: str = Field("", max_length=50)
    ctip: str = Field("", max_length=50)
    os: str = Field("", max_length=100)
    kinds: list[JobKind] = Field(default_factory=lambda: list(JobKind))


class Free(Strict):
    """What the worker can give right now (after its own limits and other programs)."""
    vram_gb: float = Field(0.0, ge=0)
    ram_gb: float = Field(0.0, ge=0)
    busy_by_others: bool = False


class JobProgress(Strict):
    job_id: str
    progress: float = Field(0.0, ge=0, le=1)
    epoch: int | None = None
    metrics: dict[str, float] = Field(default_factory=dict)
    message: str = Field("", max_length=500)

    @field_validator("metrics")
    @classmethod
    def _few(cls, v: dict[str, float]) -> dict[str, float]:
        if len(v) > 50:
            raise ValueError("at most 50 metrics")
        return v


class EnrollRequest(Strict):
    enrollment_token: str = Field(min_length=20, max_length=200)
    name: str = Field(min_length=1, max_length=80)
    capabilities: Capabilities
    protocol: int = PROTOCOL


class EnrollResponse(BaseModel):
    worker_id: str
    worker_token: str
    heartbeat_s: int


class Heartbeat(Strict):
    capabilities: Capabilities | None = None
    free: Free | None = None
    jobs: list[JobProgress] = Field(default_factory=list, max_length=8)


class HeartbeatResponse(BaseModel):
    cancel: list[str] = Field(default_factory=list, description="cancelled by an admin: stop and confirm")
    lost: list[str] = Field(default_factory=list, description="no longer leased to this worker: stop quietly")
    lease_s: int


class LeaseRequest(Strict):
    free: Free


class Lease(BaseModel):
    job_id: str
    spec: dict
    attempt: int
    lease_s: int
    checkpoint: str | None = Field(None, description="artifact id of last.pt to resume from")


class Finish(Strict):
    status: Literal["completed", "failed", "yielded"]
    metrics: dict[str, float] = Field(default_factory=dict)
    message: str = Field("", max_length=2000)
    retryable: bool = False
