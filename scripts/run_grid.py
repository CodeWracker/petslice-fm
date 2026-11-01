#!/usr/bin/env python
"""Run a stage of the grid across GPUs. Finished runs are skipped, so it can simply be restarted.

Runs start one at a time and only while `resource_guard.py` reports no limit, because starting many
processes at once coincided with the loss of the container. Thread pools are capped, otherwise each
process sizes them to the 224 host threads.
"""

import argparse
import os
import subprocess
import time
from pathlib import Path

from petslice.experiments import STAGES
from petslice.paths import RUNS_ROOT
from petslice.training.config import RunConfig
from petslice.training.running import command_for, running_elsewhere
from petslice.training.scheduling import choose_start, cost_of

PAUSE = Path(os.environ.get("PETSLICE_PAUSE", RUNS_ROOT / "logs" / "PAUSE"))
THREAD_LIMITS = {
    "OMP_NUM_THREADS": "2",
    "MKL_NUM_THREADS": "2",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
}


def free_memory_gb() -> dict[int, float]:
    output = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    rows = [line.split(", ") for line in output.strip().splitlines()]
    return {int(index): float(free) / 1024 for index, free in rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=STAGES)
    parser.add_argument("--gpus", type=int, nargs="+", required=True)
    parser.add_argument("--per-gpu", type=int, default=2)
    parser.add_argument("--stagger", type=float, default=120, help="Seconds between run starts.")
    parser.add_argument(
        "--min-free-gb", type=float, default=20, help="GPUs are shared with others."
    )
    args = parser.parse_args()

    elsewhere = running_elsewhere()
    pending = [
        run
        for run in STAGES[args.stage]()
        if not (run.run_dir / "metrics.json").exists() and run.run_dir not in elsewhere
    ]
    print(
        f"{len(pending)} runs pending in stage {args.stage}, {len(elsewhere)} running elsewhere",
        flush=True,
    )
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
        if pending and not PAUSE.exists() and time.time() - last_start >= args.stagger:
            load = {gpu: sum(cost for _, g, _, cost in active if g == gpu) for gpu in args.gpus}
            for gpu in running_elsewhere().values():
                if gpu in load:
                    load[gpu] += 1
            choice = choose_start(pending, load, free_memory_gb(), args.per_gpu, args.min_free_gb)
            if choice is not None:
                index, gpu = choice
                run = pending.pop(index)
                cost = cost_of(run, args.per_gpu)
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
