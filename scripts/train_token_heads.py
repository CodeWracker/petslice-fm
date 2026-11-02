#!/usr/bin/env python
"""Train pooling and a linear head on the cached tokens of each frozen 3D encoder."""

import argparse

from petslice.experiments import FOLDS, TASK_NAMES
from petslice.tasks import TASKS
from petslice.training.tokens import TOKEN_ROOT, train_tokens

SHARED = ("spectre__tokens__frozen", "petct_fm__tokens__frozen", "nnfoundation__tokens__frozen")
PER_FOLD = "nnunet__tokens__frozen"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--methods", nargs="+", default=[*SHARED, PER_FOLD])
    args = parser.parse_args()
    for method in args.methods:
        for task in TASK_NAMES:
            scores = []
            for fold in FOLDS:
                root = None
                if method == PER_FOLD:
                    root = TOKEN_ROOT / f"{method}_{TASKS[task].tracer}_fold{fold}"
                scores.append(train_tokens(task, method, fold, root)["test"]["auroc"])
            print(f"{task} {method}: mean fold AUROC {sum(scores) / len(scores):.3f}", flush=True)


if __name__ == "__main__":
    main()
