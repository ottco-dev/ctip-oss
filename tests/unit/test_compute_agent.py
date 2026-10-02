"""Compute agent: transport security, safe dataset unpacking, config permissions, device choice."""

from __future__ import annotations

import os
import stat
import sys
import zipfile

import pytest

from apps.worker.agent import device_for
from apps.worker.client import check_url
from apps.worker.config import WorkerConfig
from apps.worker.datasets import dataset_yaml, safe_extract
from shared.compute.matching import fits
from shared.compute.schemas import Capabilities, Free, Requirements


def test_only_https_except_localhost():
    assert check_url("https://ctip.example.org/") == "https://ctip.example.org"
    assert check_url("http://127.0.0.1:8000") == "http://127.0.0.1:8000"
    with pytest.raises(ValueError, match="insecure"):
        check_url("http://ctip.example.org")


def test_zip_slip_and_filtering(tmp_path):
    bad = tmp_path / "bad.zip"
    with zipfile.ZipFile(bad, "w") as z:
        z.writestr("../evil.txt", "x")
    with pytest.raises(ValueError, match="unsafe"):
        safe_extract(bad, tmp_path / "out")
    good = tmp_path / "good.zip"
    with zipfile.ZipFile(good, "w") as z:
        z.writestr("ds/dataset.yaml", "path: /old/place\ntrain: images/train\nval: images/val\nnames: {0: head}\n")
        z.writestr("ds/images/train/a.jpg", b"jpg")
        z.writestr("ds/run.sh", "rm -rf ~")                               # not a dataset file: skipped
    out = safe_extract(good, tmp_path / "ds")
    assert (out / "ds/images/train/a.jpg").exists() and not (out / "ds/run.sh").exists()
    y = dataset_yaml(out, tmp_path / "work/dataset.yaml").read_text()
    assert f"path: {(out / 'ds').resolve()}" in y and "train: images/train" in y


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX permissions")
def test_config_is_owner_only(tmp_path):
    cfg = WorkerConfig(server="https://x", token="ctipw_secret", name="pc")
    p = cfg.save(tmp_path / "config.json")
    assert stat.S_IMODE(os.stat(p).st_mode) == 0o600
    assert WorkerConfig.load(p).token == "ctipw_secret"


def test_devices():
    cuda4 = Capabilities(backend="cuda", device="x", gpus=4, vram_gb=24, ram_gb=64, cpu_cores=32)
    assert device_for(cuda4, True) == "0,1,2,3" and device_for(cuda4, False) == "0"
    assert device_for(Capabilities(backend="mps", device="M3", gpus=1, ram_gb=16), False) == "mps"
    assert device_for(Capabilities(backend="cpu", device="x", ram_gb=16), False) == "cpu"


def test_matching_ram_and_cpu_jobs():
    cpu = Capabilities(backend="cpu", device="x", ram_gb=64, cpu_cores=32)
    assert fits(Requirements(), "detection_benchmark", cpu, Free(ram_gb=8))
    assert not fits(Requirements(min_ram_gb=16), "detection_benchmark", cpu, Free(ram_gb=8))
    assert not fits(Requirements(min_vram_gb=4), "yolo_train", cpu, Free(ram_gb=32))
    limited = cpu.model_copy(update={"kinds": ["detection_benchmark"]})
    assert not fits(Requirements(), "yolo_train", limited, Free(ram_gb=32))
