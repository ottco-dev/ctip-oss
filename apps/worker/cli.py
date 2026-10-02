"""
ctip-worker — lend this machine's GPU/CPU to a CTIP coordinator.

    ctip-worker doctor                                   # what this machine can do
    ctip-worker enroll --server https://ctip.example.org --token ctipe_...
    ctip-worker run                                      # work until Ctrl+C (one job at a time)
    ctip-worker limits --set max_vram_fraction=0.6 --set yield_when_busy=true
"""

from __future__ import annotations

import argparse
import json
import logging
import signal
import socket
import sys
from dataclasses import asdict, fields

from apps.worker import __version__, hardware
from apps.worker.agent import Agent
from apps.worker.client import Coordinator, CoordinatorError, check_url
from apps.worker.config import Limits, WorkerConfig, config_dir
from shared.compute.schemas import EnrollRequest


def _doctor(cfg: WorkerConfig) -> int:
    caps = hardware.detect(cfg.limits)
    free = hardware.free(caps, cfg.limits)
    print(f"ctip-worker {__version__}")
    print(f"backend      {caps.backend.value}  ({caps.device}, {caps.gpus} GPU, {caps.vram_gb} GB VRAM)")
    print(f"memory       {caps.ram_gb} GB RAM, {caps.cpu_cores} CPU threads; torch {caps.torch or 'missing'}")
    print(f"free now     {free.vram_gb} GB VRAM, {free.ram_gb} GB RAM, busy by other programs: {free.busy_by_others}")
    util = hardware.others_gpu_util()
    if util is not None:
        print(f"others' GPU  {util}% (yield threshold {cfg.limits.busy_gpu_util}%)")
    print(f"job kinds    {', '.join(k.value for k in caps.kinds)}")
    print(f"config       {config_dir() / 'config.json'}  ({'enrolled as ' + cfg.name if cfg.token else 'not enrolled'})")
    print(f"limits       {json.dumps(asdict(cfg.limits))}")
    return 0


def _enroll(cfg: WorkerConfig, server: str, token: str, name: str, ca: str) -> int:
    server = check_url(server)
    caps = hardware.detect(cfg.limits)
    try:
        resp = Coordinator(server, ca_bundle=ca).enroll(EnrollRequest(enrollment_token=token, name=name, capabilities=caps))
    except CoordinatorError as e:
        print(f"enrolment failed: {e.detail}", file=sys.stderr)
        return 1
    cfg.server, cfg.token, cfg.worker_id, cfg.name, cfg.ca_bundle = server, resp.worker_token, resp.worker_id, name, ca
    path = cfg.save()
    print(f"enrolled as '{name}' (worker {resp.worker_id[:8]}); token saved to {path} (owner-only)")
    print("start working with:  ctip-worker run")
    return 0


def _limits(cfg: WorkerConfig, sets: list[str]) -> int:
    types = {f.name: f.type for f in fields(Limits)}
    for item in sets:
        key, _, value = item.partition("=")
        if key not in types:
            print(f"unknown limit {key!r}; known: {', '.join(types)}", file=sys.stderr)
            return 2
        cur = getattr(cfg.limits, key)
        if isinstance(cur, bool):
            val = value.lower() in ("1", "true", "yes", "on")
        elif isinstance(cur, int):
            val = int(value)
        elif isinstance(cur, float):
            val = float(value)
        else:
            val = [v.strip() for v in value.split(",") if v.strip()]
        setattr(cfg.limits, key, val)
    if sets:
        cfg.save()
    print(json.dumps(asdict(cfg.limits), indent=1))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="ctip-worker", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("doctor", help="show hardware, free resources and limits")
    e = sub.add_parser("enroll", help="connect this machine to a coordinator with a one-time token")
    e.add_argument("--server", required=True)
    e.add_argument("--token", required=True)
    e.add_argument("--name", default=socket.gethostname()[:80])
    e.add_argument("--ca", default="", help="CA bundle / self-signed certificate to trust")
    r = sub.add_parser("run", help="take and run jobs until stopped")
    r.add_argument("--once", action="store_true", help="run at most one job, then exit")
    lim = sub.add_parser("limits", help="show or change resource limits")
    lim.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = WorkerConfig.load()
    if a.cmd == "doctor":
        return _doctor(cfg)
    if a.cmd == "enroll":
        return _enroll(cfg, a.server, a.token, a.name, a.ca)
    if a.cmd == "limits":
        return _limits(cfg, a.set)
    agent = Agent(cfg)

    def _stop(*_):
        logging.getLogger("ctip-worker").info("stopping after the current step (Ctrl+C again to force)")
        agent.stop.set()
        agent._yield.set()                               # hand a running job back with its checkpoint
    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)
    agent.run(once=a.once)
    return 0


if __name__ == "__main__":
    sys.exit(main())
