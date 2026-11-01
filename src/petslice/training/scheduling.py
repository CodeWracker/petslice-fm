"""Choose which pending run starts next and on which GPU.

Runs are preferred in grid order, but a run that does not fit yet must not block the ones behind
it: a heavy run that needs a whole GPU reserves the emptiest one, which is left to drain, while the
other GPUs keep taking lighter runs.
"""

from collections.abc import Sequence

from petslice.training.config import RunConfig

HEAVY_ENCODERS = {"medimageinsight"}


def cost_of(run: RunConfig, per_gpu: int) -> int:
    return per_gpu if run.encoder in HEAVY_ENCODERS else 1


def choose_start(
    pending: Sequence[RunConfig],
    load: dict[int, int],
    free_gb: dict[int, float],
    per_gpu: int,
    min_free_gb: float,
) -> tuple[int, int] | None:
    """Index into `pending` and GPU of the first run that fits, or None when nothing fits."""
    reserved = None
    for index, run in enumerate(pending):
        cost = cost_of(run, per_gpu)
        usable = [
            gpu
            for gpu in load
            if gpu != reserved
            and load[gpu] + cost <= per_gpu
            and free_gb[gpu] >= min_free_gb * cost
        ]
        if usable:
            return index, min(usable, key=load.get)
        if reserved is None and cost > 1:
            reserved = min(load, key=load.get)
    return None
