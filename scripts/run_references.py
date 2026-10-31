#!/usr/bin/env python
"""Fit the logistic-regression references on every task and fold."""

from petslice.experiments import FOLDS, TASK_NAMES
from petslice.training.reference import FEATURES, run_reference


def main() -> None:
    for task in TASK_NAMES:
        for method in FEATURES:
            scores = [run_reference(task, method, fold)["test"]["auroc"] for fold in FOLDS]
            print(f"{task} {method}: mean fold AUROC {sum(scores) / len(scores):.3f}", flush=True)


if __name__ == "__main__":
    main()
