import json
import os
import signal
import subprocess
import sys
import time

from petslice.training.config import RunConfig
from petslice.training.running import command_for, config_from_command, running_elsewhere


def test_command_line_round_trips_to_the_same_run():
    run = RunConfig(group="study", task="lesion_fdg", input_mode="mip", fold=3, train_fraction=0.25)
    assert config_from_command(command_for(run)) == run


def test_other_processes_are_ignored():
    assert config_from_command(["python", "scripts/other.py", "--task", "diagnosis"]) is None
    assert config_from_command(["python", "scripts/train.py", "--fold", "1"]) is None


def test_a_live_training_process_is_found_with_its_gpu():
    """The run is a grandchild, like those of a frozen runner; direct children are our own."""
    run = RunConfig(group="smoke", task="diagnosis", fold=1)
    inner = [sys.executable, "-c", "import time; time.sleep(60)", *command_for(run)[1:]]
    outer = [
        sys.executable,
        "-c",
        "import json, os, subprocess; subprocess.run(json.loads(os.environ['INNER']))",
    ]
    environment = {**os.environ, "CUDA_VISIBLE_DEVICES": "3", "INNER": json.dumps(inner)}
    process = subprocess.Popen(outer, env=environment, start_new_session=True)
    try:
        deadline = time.time() + 10
        while run.run_dir not in running_elsewhere() and time.time() < deadline:
            time.sleep(0.2)
        found = running_elsewhere()
    finally:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()
    assert found.get(run.run_dir) == 3
    assert run.run_dir not in running_elsewhere()
