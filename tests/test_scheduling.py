from petslice.training.config import RunConfig
from petslice.training.scheduling import choose_start

LIGHT = RunConfig(group="study", task="diagnosis", encoder="dinov2")
HEAVY = RunConfig(group="study", task="diagnosis", encoder="medimageinsight")
FREE = {0: 70.0, 1: 70.0, 2: 70.0}


def test_runs_start_in_grid_order_on_the_emptiest_gpu():
    assert choose_start([LIGHT, HEAVY], {0: 1, 1: 0, 2: 2}, FREE, 2, 20) == (0, 1)


def test_a_blocked_heavy_run_reserves_the_emptiest_gpu_and_lets_light_runs_pass():
    load = {0: 2, 1: 1, 2: 2}
    assert choose_start([HEAVY, LIGHT], load, FREE, 2, 20) is None
    assert choose_start([HEAVY, LIGHT], {0: 1, 1: 1, 2: 2}, FREE, 2, 20) == (1, 1)
    assert choose_start([HEAVY, LIGHT, LIGHT], {0: 1, 1: 1, 2: 1}, FREE, 2, 20) == (1, 1)


def test_a_heavy_run_takes_an_empty_gpu_with_enough_memory():
    assert choose_start([HEAVY, LIGHT], {0: 1, 1: 0, 2: 2}, FREE, 2, 20) == (0, 1)
    assert choose_start([HEAVY, LIGHT], {0: 1, 1: 0, 2: 2}, {**FREE, 1: 30.0}, 2, 20) == (1, 0)


def test_gpus_short_of_memory_are_skipped():
    assert choose_start([LIGHT], {0: 0, 1: 1}, {0: 10.0, 1: 40.0}, 2, 20) == (0, 1)
    assert choose_start([LIGHT], {0: 0, 1: 1}, {0: 10.0, 1: 10.0}, 2, 20) is None
