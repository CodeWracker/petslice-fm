#!/usr/bin/env python
"""Log container resources to persistent storage and pause new runs near a limit.

The host removes the container without notice, taking its logs along, so each sample is synced to
disk and the last lines describe the moments before a kill. Crossing a limit creates the PAUSE file,
which `run_grid.py` honours; running jobs are left alone. Limits are conservative local choices: the
only rule the host documents is 30 GB in the writable layer.
"""

import argparse
import json
import os
import shutil
import subprocess
import time

from petslice.paths import RUNS_ROOT

LOG = RUNS_ROOT / "logs" / "resources.jsonl"
PAUSE = RUNS_ROOT / "logs" / "PAUSE"
CGROUP = "/sys/fs/cgroup"
OVERLAY_EVERY = 300


def read_number(name: str) -> int:
    with open(f"{CGROUP}/{name}") as source:
        return int(source.read().split()[0])


def committed_memory_gb() -> float:
    """Anonymous and shared memory; page cache is excluded because the kernel reclaims it."""
    with open(f"{CGROUP}/memory.stat") as source:
        stat = dict(line.split() for line in source)
    return (int(stat["anon"]) + int(stat["shmem"])) / 1e9


def memory_available_gb() -> float:
    with open("/proc/meminfo") as source:
        for line in source:
            if line.startswith("MemAvailable"):
                return int(line.split()[1]) / 1e6
    return float("nan")


def arc_available_gb() -> float:
    with open("/proc/spl/kstat/zfs/arcstats") as source:
        for line in source:
            if line.startswith("size "):
                return int(line.split()[2]) / 1e9
    return float("nan")


def overlay_gb() -> float:
    """Size of the root filesystem without bind mounts; image layers are part of it."""
    output = subprocess.run(
        ["du", "-xsk", "/"], capture_output=True, text=True, timeout=120, check=False
    ).stdout
    return int(output.split()[0]) / 1e6 if output else float("nan")


def gpu_memory_mib() -> list[int]:
    output = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    ).stdout
    return [int(value) for value in output.split()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interval", type=float, default=15)
    parser.add_argument("--max-container-gb", type=float, default=200)
    parser.add_argument("--max-overlay-growth-gb", type=float, default=12)
    parser.add_argument("--max-threads", type=int, default=4000)
    args = parser.parse_args()

    LOG.parent.mkdir(parents=True, exist_ok=True)
    baseline = overlay_gb()
    overlay, last_overlay = baseline, time.time()
    with LOG.open("a") as log:
        log.write(json.dumps({"event": "start", "time": time.time(), "overlay": baseline}) + "\n")
        while True:
            if time.time() - last_overlay >= OVERLAY_EVERY:
                overlay, last_overlay = overlay_gb(), time.time()
            sample = {
                "time": round(time.time()),
                "container_gb": round(committed_memory_gb(), 1),
                "with_cache_gb": round(read_number("memory.current") / 1e9, 1),
                "threads": read_number("pids.current"),
                "overlay_growth_gb": round(overlay - baseline, 2),
                "shm_gb": round(shutil.disk_usage("/dev/shm").used / 1e9, 1),
                "host_available_gb": round(memory_available_gb(), 1),
                "arc_gb": round(arc_available_gb(), 1),
                "load": round(os.getloadavg()[0]),
                "gpu_mib": gpu_memory_mib(),
            }
            reasons = []
            if sample["container_gb"] > args.max_container_gb:
                reasons.append("container memory")
            if sample["overlay_growth_gb"] > args.max_overlay_growth_gb:
                reasons.append("overlay growth")
            if sample["threads"] > args.max_threads:
                reasons.append("threads")
            if reasons:
                sample["paused"] = reasons
                PAUSE.write_text(", ".join(reasons) + "\n")
            elif PAUSE.exists():
                PAUSE.unlink()
            log.write(json.dumps(sample) + "\n")
            log.flush()
            os.fsync(log.fileno())
            time.sleep(args.interval)


if __name__ == "__main__":
    main()
