#!/usr/bin/env python
"""Run a stage of the grid across GPUs. Finished runs are skipped, so it can simply be restarted.

Runs start one at a time and only while `resource_guard.py` reports no limit, because starting many
processes at once coincided with the loss of the container. Thread pools are capped, otherwise each
process sizes them to the 224 host threads.
"""

import argparse
import dataclasses
import os
import subprocess
import sys
import time
from pathlib import Path

from petslice.experiments import STAGES
from petslice.paths import REPO_ROOT, RUNS_ROOT
from petslice.training.config import RunConfig

HEAVY_ENCODERS = {"medimageinsight"}
PAUSE = RUNS_ROOT / "logs" / "PAUSE"
THREAD_LIMITS = {
    "OMP_NUM_THREADS": "2",
    "MKL_NUM_THREADS": "2",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
}


def command_for(run: RunConfig) -> list[str]:
    arguments = [sys.executable, str(REPO_ROOT / "scripts" / "train.py")]
    for field in dataclasses.fields(run):
        arguments += [f"--{field.name.replace('_', '-')}", str(getattr(run, field.name))]
    return arguments


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=STAGES)
    parser.add_argument("--gpus", type=int, nargs="+", required=True)
    parser.add_argument("--per-gpu", type=int, default=2)
    parser.add_argument("--stagger", type=float, default=120, help="Seconds between run starts.")
    args = parser.parse_args()

    pending = [run for run in STAGES[args.stage]() if not (run.run_dir / "metrics.json").exists()]
    print(f"{len(pending)} runs pending in stage {args.stage}", flush=True)
    log_dir = RUNS_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    active: list[tuple[subprocess.Popen, int, RunConfig, int]] = []
    failed: list[RunConfig] = []
    finished = 0
    last_start = 0.0
    while pending or active:
        for entry in list(active):
            process, _, run, _ = entry
            if process.poll() is not None:
                active.remove(entry)
                finished += 1
                status = "ok" if process.returncode == 0 else f"FAILED {process.returncode}"
                if process.returncode != 0:
                    failed.append(run)
                print(f"[{finished}] {run.task} {run.method} fold{run.fold}: {status}", flush=True)
        load = {gpu: sum(cost for _, g, _, cost in active if g == gpu) for gpu in args.gpus}
        gpu = min(args.gpus, key=load.get)
        if pending and not PAUSE.exists() and time.time() - last_start >= args.stagger:
            cost = args.per_gpu if pending[0].encoder in HEAVY_ENCODERS else 1
            if load[gpu] + cost <= args.per_gpu:
                run = pending.pop(0)
                name = f"{run.task}__{run.method}__fold{run.fold}.log"
                environment = {**os.environ, **THREAD_LIMITS, "CUDA_VISIBLE_DEVICES": str(gpu)}
                with Path(log_dir / name).open("a") as log:
                    process = subprocess.Popen(
                        command_for(run), env=environment, stdout=log, stderr=subprocess.STDOUT
                    )
                active.append((process, gpu, run, cost))
                last_start = time.time()
        time.sleep(5)
    print(f"stage {args.stage} finished, {len(failed)} failed", flush=True)
    for run in failed:
        print(f"failed: {run.task} {run.method} fold{run.fold}", flush=True)


if __name__ == "__main__":
    main()
