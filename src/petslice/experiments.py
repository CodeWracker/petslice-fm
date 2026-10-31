"""The experiment grid. Input-dependent stages read the choice made on validation data."""

import json
from pathlib import Path
from statistics import mean

from petslice.paths import RUNS_ROOT
from petslice.training.config import RunConfig

GROUP = "study"
FOLDS = range(5)
TASK_NAMES = ("diagnosis", "lesion_fdg", "lesion_psma")
COMPARED_INPUTS = ("early", "late", "mip")
CONTROL_INPUTS = ("pet_only", "ct_only")
OTHER_ENCODERS = ("dinov2", "rad_dino", "medimageinsight")
FRACTIONS = (0.1, 0.25, 0.5)


def runs_for(task: str, **options) -> list[RunConfig]:
    return [RunConfig(group=GROUP, task=task, fold=fold, **options) for fold in FOLDS]


def input_runs() -> list[RunConfig]:
    """E2: input comparison, plus single-modality controls."""
    return [
        run
        for task in TASK_NAMES
        for mode in (*COMPARED_INPUTS, *CONTROL_INPUTS)
        for run in runs_for(task, input_mode=mode)
    ]


def validation_auroc(task: str, method: str, root: Path = RUNS_ROOT) -> float:
    scores = []
    for fold in FOLDS:
        path = root / GROUP / task / method / f"fold{fold}" / "metrics.json"
        scores.append(json.loads(path.read_text())["val_auroc"])
    return mean(scores)


def selected_input(task: str, root: Path = RUNS_ROOT) -> str:
    """Input with the best mean validation AUROC; test folds play no part in this choice."""
    return max(
        COMPARED_INPUTS,
        key=lambda mode: validation_auroc(task, f"dinov3__{mode}__lora", root),
    )


def backbone_runs() -> list[RunConfig]:
    """E3: other encoders, and DINOv3 without LoRA, on each task's selected input."""
    runs = []
    for task in TASK_NAMES:
        mode = selected_input(task)
        for encoder in OTHER_ENCODERS:
            runs += runs_for(task, encoder=encoder, input_mode=mode)
        runs += runs_for(task, input_mode=mode, adapter="none")
    return runs


def data_efficiency_runs() -> list[RunConfig]:
    """E4: nested training fractions on each task's selected input."""
    return [
        run
        for task in TASK_NAMES
        for fraction in FRACTIONS
        for run in runs_for(task, input_mode=selected_input(task), train_fraction=fraction)
    ]


STAGES = {
    "inputs": input_runs,
    "backbones": backbone_runs,
    "data_efficiency": data_efficiency_runs,
}
