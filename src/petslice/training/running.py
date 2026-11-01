"""Find training runs that another runner started, so a replacement runner can take over.

The grid runner is replaced while runs are in flight when its placement rules change, for example
when another container fills a GPU. Killing it would send SIGHUP to its children, and SIGSTOP does
not hold under tmux, which resumes a stopped pane. The old runner is therefore parked with its PAUSE
file held while a new runner, pointed at another pause file through `PETSLICE_PAUSE`, takes over.
The new runner reads the old runner's live runs from /proc to skip them and to count them against
the GPU slots, and the old runner is killed once its last run has ended.
"""

import dataclasses
import os
import sys
from pathlib import Path

from petslice.paths import REPO_ROOT
from petslice.training.config import RunConfig

TRAIN_SCRIPT = "scripts/train.py"
FIELDS = {field.name: field.type for field in dataclasses.fields(RunConfig)}


def command_for(run: RunConfig) -> list[str]:
    arguments = [sys.executable, str(REPO_ROOT / TRAIN_SCRIPT)]
    for name in FIELDS:
        arguments += [f"--{name.replace('_', '-')}", str(getattr(run, name))]
    return arguments


def config_from_command(arguments: list[str]) -> RunConfig | None:
    """Rebuild the configuration behind a `train.py` command line; None for any other process."""
    if not any(argument.endswith(TRAIN_SCRIPT) for argument in arguments):
        return None
    values = {}
    for flag, value in zip(arguments[:-1], arguments[1:], strict=True):
        name = flag[2:].replace("-", "_")
        if flag.startswith("--") and name in FIELDS:
            values[name] = FIELDS[name](value)
    try:
        return RunConfig(**values)
    except TypeError:
        return None


def _read(path: Path) -> bytes:
    try:
        return path.read_bytes()
    except OSError:
        return b""


def running_elsewhere() -> dict[Path, int]:
    """Run directories of live `train.py` processes not started by this process, with their GPU.

    DataLoader workers repeat the command line of their parent and are skipped through the parent.
    """
    runs = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        arguments = [a.decode() for a in _read(entry / "cmdline").split(b"\0")[:-1]]
        config = config_from_command(arguments)
        if config is None:
            continue
        stat = _read(entry / "stat").decode()
        parent = int(stat.rsplit(")", 1)[1].split()[1]) if ")" in stat else 0
        if parent == os.getpid() or TRAIN_SCRIPT.encode() in _read(Path(f"/proc/{parent}/cmdline")):
            continue
        environment = _read(entry / "environ").split(b"\0")
        gpu = next(
            (int(e.split(b"=")[1]) for e in environment if e.startswith(b"CUDA_VISIBLE_DEVICES=")),
            None,
        )
        if gpu is not None:
            runs[config.run_dir] = gpu
    return runs
