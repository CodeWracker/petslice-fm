#!/usr/bin/env python
"""E6: score every PSMA study with the models trained on FDG lesion detection."""

import argparse
import json
from dataclasses import replace

import pandas as pd
import torch

from petslice.experiments import FOLDS, GROUP, selected_input
from petslice.paths import MANIFEST_PATH, RUNS_ROOT
from petslice.tasks import TASKS
from petslice.training.config import RunConfig
from petslice.training.runner import build_model, predict, prediction_frame, save_attention


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-mode")
    args = parser.parse_args()
    mode = args.input_mode or selected_input("lesion_fdg")
    source = TASKS["lesion_fdg"]
    cohort = TASKS["lesion_psma"].select(pd.read_csv(MANIFEST_PATH))
    for fold in FOLDS:
        config = RunConfig(group=GROUP, task="lesion_fdg", input_mode=mode, fold=fold)
        output = RUNS_ROOT / GROUP / "fdg_to_psma" / config.method / f"fold{fold}"
        if (output / "predictions.parquet").exists():
            continue
        model = build_model(config, source).cuda()
        model.load_state_dict(torch.load(config.run_dir / "best.pt"), strict=False)
        threshold = json.loads((config.run_dir / "metrics.json").read_text())["threshold"]
        result = predict(model, cohort, replace(config, num_workers=12), "cuda")
        output.mkdir(parents=True, exist_ok=True)
        prediction_frame(result, "external", threshold).to_parquet(output / "predictions.parquet")
        save_attention(output / "attention.npz", result["attention"])
        print(f"fold {fold} done", flush=True)


if __name__ == "__main__":
    main()
