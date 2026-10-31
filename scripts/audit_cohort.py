#!/usr/bin/env python
"""Describe the cohort and test how well acquisition variables alone predict each label."""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from petslice.paths import MANIFEST_PATH, RESULTS_DIR
from petslice.tasks import TASKS

TABLES = RESULTS_DIR / "tables"
GRID_WIDTH_MM = 224 * 2.5
SIGNALS = ("z_extent_mm", "n_slices", "body_width_mm", "age")
FLAGS = ("partial_body", "contrast")


def class_counts(manifest: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for task in TASKS.values():
        cohort = task.select(manifest)
        for fold, group in cohort.groupby("fold"):
            counts = group["label"].value_counts()
            for label, name in enumerate(task.classes):
                rows.append(
                    {
                        "task": task.name,
                        "fold": fold,
                        "class": name,
                        "studies": int(counts.get(label, 0)),
                        "patients": group[group["label"] == label]["patient_id"].nunique(),
                    }
                )
    return pd.DataFrame(rows)


def signal_strength(manifest: pd.DataFrame) -> pd.DataFrame:
    """One-vs-rest AUROC of each single variable; 0.5 means no leakage through that variable."""
    rows = []
    frame = manifest.assign(
        partial_body=manifest["partial_body"].astype(float),
        contrast=(manifest["ct_contrast"] == "yes").astype(float),
    )
    for task in TASKS.values():
        cohort = task.select(frame)
        for variable in (*SIGNALS, *FLAGS):
            valid = cohort[variable].notna()
            for label, name in enumerate(task.classes):
                target = cohort.loc[valid, "label"] == label
                if target.nunique() < 2 or (len(task.classes) == 2 and label == 0):
                    continue
                value = cohort.loc[valid, variable]
                rows.append(
                    {
                        "task": task.name,
                        "class": name,
                        "variable": variable,
                        "auroc": roc_auc_score(target, value),
                        "class_median": value[target].median(),
                        "rest_median": value[~target].median(),
                    }
                )
    return pd.DataFrame(rows)


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(MANIFEST_PATH)
    counts = class_counts(manifest)
    counts.to_csv(TABLES / "cohort_counts.csv", index=False)
    strength = signal_strength(manifest)
    strength.to_csv(TABLES / "cohort_signals.csv", index=False, float_format="%.4f")

    print(counts.groupby(["task", "class"])[["studies", "patients"]].sum().to_string())
    print(counts.pivot_table(index=["task", "class"], columns="fold", values="studies").to_string())

    fdg = manifest[manifest["tracer"] == "fdg"]
    print("\nlesion_positive by diagnosis")
    print(pd.crosstab(fdg["diagnosis"], fdg["lesion_positive"]).to_string())
    print("\nacquisition by tracer")
    print(manifest.groupby("tracer")[["n_slices", "z_extent_mm", "body_width_mm"]].describe().T)
    clipped = (manifest["body_width_mm"] > GRID_WIDTH_MM).groupby(manifest["tracer"]).mean()
    print(f"\nbody wider than the {GRID_WIDTH_MM:.0f} mm grid\n{clipped.to_string()}")
    print("\nsingle-variable AUROC")
    print(strength.to_string(index=False, float_format="%.3f"))
    leak = manifest.groupby("patient_id")[["fold", "val_bucket"]].nunique()
    assert (leak["fold"] == 1).all(), "patient spans folds"
    print(f"\npatients: {len(leak)}, each confined to one fold: {np.all(leak['fold'] == 1)}")


if __name__ == "__main__":
    main()
