#!/usr/bin/env python
"""Turn finished runs into the tables behind the paper. Safe to rerun while runs are in progress."""

import json

import numpy as np
import pandas as pd

from petslice.data.manifest import study_key
from petslice.evaluation.attention import (
    correlation_test,
    study_statistics,
    summarize,
)
from petslice.evaluation.decisions import (
    attention_holds,
    best_baseline,
    input_verdict,
    transfer_holds,
)
from petslice.evaluation.metrics import bootstrap_summary, holm, paired_auroc_difference
from petslice.experiments import COMPARED_INPUTS, FOLDS, GROUP, TASK_NAMES, selected_input
from petslice.paths import MANIFEST_PATH, PROCESSED_DIR, RESULTS_DIR, RUNS_ROOT

TABLES = RESULTS_DIR / "tables"
PREDICTIONS = RESULTS_DIR / "predictions"
ATTENTION = RESULTS_DIR / "attention"


def load_method(task: str, method: str, manifest: pd.DataFrame, split: str = "test"):
    """Pool out-of-fold predictions; returns None until all five folds exist."""
    frames = []
    for fold in FOLDS:
        path = RUNS_ROOT / GROUP / task / method / f"fold{fold}" / "predictions.parquet"
        if not path.exists():
            return None
        frame = pd.read_parquet(path)
        frames.append(frame[frame["split"] == split].assign(fold=fold))
    pooled = pd.concat(frames, ignore_index=True)
    return pooled.merge(manifest[["study_id", "patient_id"]], on="study_id", validate="many_to_one")


def cost_of(task: str, method: str) -> dict:
    rows = []
    for fold in FOLDS:
        path = RUNS_ROOT / GROUP / task / method / f"fold{fold}" / "metrics.json"
        if path.exists():
            rows.append(json.loads(path.read_text()))
    if not rows:
        return {}
    return {
        "trainable_parameters": rows[0].get("trainable_parameters"),
        "gpu_hours_per_fold": float(np.mean([r.get("train_seconds", 0) for r in rows]) / 3600),
        "peak_gpu_gb": float(np.mean([r.get("peak_gpu_bytes", 0) for r in rows]) / 1e9),
        "val_auroc": float(np.mean([r["val_auroc"] for r in rows])),
        "mean_epochs": float(np.mean([r.get("epochs_run", 0) for r in rows])),
    }


def method_table(manifest: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for task_dir in sorted((RUNS_ROOT / GROUP).iterdir()):
        if task_dir.name == "fdg_to_psma":
            continue
        for method_dir in sorted(task_dir.iterdir()):
            pooled = load_method(task_dir.name, method_dir.name, manifest)
            if pooled is None:
                continue
            PREDICTIONS.mkdir(parents=True, exist_ok=True)
            pooled.to_parquet(PREDICTIONS / f"{task_dir.name}__{method_dir.name}.parquet")
            rows.append(
                {
                    "task": task_dir.name,
                    "method": method_dir.name,
                    "n_studies": len(pooled),
                    **bootstrap_summary(pooled),
                    **cost_of(task_dir.name, method_dir.name),
                }
            )
    return pd.DataFrame(rows)


def input_comparisons(manifest: pd.DataFrame) -> pd.DataFrame:
    """H2: pairwise paired differences, with p-values Holm-corrected over the pairs of a task."""
    rows = []
    for task in ("diagnosis", "lesion_fdg", "lesion_psma"):
        frames = {m: load_method(task, f"dinov3__{m}__lora", manifest) for m in COMPARED_INPUTS}
        task_rows = []
        for first, second in (("early", "late"), ("early", "mip"), ("late", "mip")):
            if frames[first] is None or frames[second] is None:
                continue
            columns = ["study_id", "label", "patient_id", *_scores(frames[first])]
            result = paired_auroc_difference(frames[first][columns], frames[second][columns])
            task_rows.append({"task": task, "first": first, "second": second, **result})
        for row, adjusted in zip(task_rows, holm([r["p_value"] for r in task_rows]), strict=True):
            rows.append({**row, "p_holm": adjusted})
    return pd.DataFrame(rows)


def _scores(frame: pd.DataFrame) -> list[str]:
    return [c for c in frame.columns if c.startswith("score_")]


def attention_tables(manifest: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """E5 on the selected slice input of each FDG task, for lesion-positive test studies."""
    positives = manifest[manifest["lesion_positive"]].set_index("study_id")
    hit_rows, correlation_rows = [], []
    for task in ("diagnosis", "lesion_fdg", "lesion_psma"):
        try:
            mode = selected_input(task)
        except FileNotFoundError:
            continue
        if mode == "mip":
            mode = "early"
        method = f"dinov3__{mode}__lora"
        rows, profiles = [], []
        for fold in FOLDS:
            path = RUNS_ROOT / GROUP / task / method / f"fold{fold}" / "attention.npz"
            if not path.exists():
                continue
            stored = np.load(path)
            for index in range(len(stored.files) // 3):
                study = str(stored[f"{index}_study"])
                if study not in positives.index:
                    continue
                lesions = np.load(PROCESSED_DIR / study_key(study) / "lesion_ml.npy")
                weights = stored[f"{index}_weights"]
                lesion_ml = lesions[stored[f"{index}_positions"]]
                if (lesion_ml > 0).any():
                    profiles.append(
                        pd.DataFrame(
                            {
                                "study_id": study,
                                "position": stored[f"{index}_positions"],
                                "weight": weights,
                                "lesion_ml": lesion_ml,
                            }
                        )
                    )
                    rows.append(
                        {
                            "study_id": study,
                            "patient_id": positives.loc[study, "patient_id"],
                            **study_statistics(weights, lesion_ml),
                        }
                    )
        if rows:
            studies = pd.DataFrame(rows)
            hit_rows.append(summarize(studies).assign(task=task, method=method))
            correlation_rows.append({"task": task, "method": method, **correlation_test(studies)})
            ATTENTION.mkdir(parents=True, exist_ok=True)
            table = pd.concat(profiles, ignore_index=True)
            table.to_parquet(ATTENTION / f"{task}__{method}.parquet")
    return (
        pd.concat(hit_rows) if hit_rows else pd.DataFrame(),
        pd.DataFrame(correlation_rows),
    )


def external_table(manifest: pd.DataFrame) -> pd.DataFrame:
    root = RUNS_ROOT / GROUP / "fdg_to_psma"
    if not root.exists():
        return pd.DataFrame()
    rows = []
    for method_dir in sorted(root.iterdir()):
        frames = [pd.read_parquet(p) for p in sorted(method_dir.glob("fold*/predictions.parquet"))]
        if len(frames) != len(FOLDS):
            continue
        scores = np.mean([f.sort_values("study_id")["score_0"].to_numpy() for f in frames], axis=0)
        predictions = np.mean(
            [f.sort_values("study_id")["prediction"].to_numpy() for f in frames], axis=0
        )
        ensemble = (
            frames[0]
            .sort_values("study_id")
            .assign(score_0=scores, prediction=(predictions >= 0.5).astype(int))
        )
        ensemble = ensemble.merge(manifest[["study_id", "patient_id"]], on="study_id")
        rows.append({"method": method_dir.name, **bootstrap_summary(ensemble)})
    return pd.DataFrame(rows)


def transfer_table(manifest: pd.DataFrame, methods: pd.DataFrame) -> pd.DataFrame:
    """H1: the recipe against the 3D reference with the best validation AUROC, paired by study."""
    rows = []
    for task in TASK_NAMES:
        try:
            recipe = f"dinov3__{selected_input(task)}__lora"
            baseline = best_baseline(methods, task)
        except (FileNotFoundError, ValueError):
            continue
        frames = {m: load_method(task, m, manifest) for m in (recipe, baseline)}
        if any(frame is None for frame in frames.values()):
            continue
        columns = ["study_id", "label", "patient_id", *_scores(frames[recipe])]
        result = paired_auroc_difference(frames[recipe][columns], frames[baseline][columns])
        rows.append(
            {
                "task": task,
                "recipe": recipe,
                "baseline": baseline,
                **result,
                "transfer": transfer_holds(result["low"]),
            }
        )
    return pd.DataFrame(rows)


def decision_table(transfer, pairs, hits, correlations) -> pd.DataFrame:
    """One row per hypothesis and task, in the terms of the analysis plan."""
    rows = []
    for row in transfer.itertuples():
        rows.append(
            {
                "hypothesis": "H1 transfer",
                "task": row.task,
                "holds": row.transfer,
                "detail": f"{row.recipe} - {row.baseline}: {row.difference:+.3f} "
                f"[{row.low:+.3f}, {row.high:+.3f}]",
            }
        )
    for task, task_pairs in pairs.groupby("task", sort=False):
        verdict = input_verdict(task_pairs)
        rows.append(
            {
                "hypothesis": "H2 input",
                "task": task,
                "holds": verdict != "no difference",
                "detail": verdict,
            }
        )
    for row in correlations.itertuples():
        task_hits = hits[(hits["task"] == row.task) & (hits["method"] == row.method)]
        rows.append(
            {
                "hypothesis": "H3 attention",
                "task": row.task,
                "holds": attention_holds(task_hits, correlations.loc[row.Index]),
                "detail": f"{row.method}: hit@1 {task_hits.iloc[0]['hit']:.2f} vs chance "
                f"{task_hits.iloc[0]['chance']:.2f}, median Spearman {row.median_spearman:+.2f}",
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(MANIFEST_PATH)
    methods = method_table(manifest)
    methods.to_csv(TABLES / "methods.csv", index=False, float_format="%.4f")
    pairs = input_comparisons(manifest)
    pairs.to_csv(TABLES / "input_pairs.csv", index=False, float_format="%.4f")
    hits, correlations = attention_tables(manifest)
    hits.to_csv(TABLES / "attention_hits.csv", index=False, float_format="%.4f")
    correlations.to_csv(TABLES / "attention_correlation.csv", index=False, float_format="%.4g")
    external_table(manifest).to_csv(TABLES / "external_psma.csv", index=False, float_format="%.4f")
    transfer = transfer_table(manifest, methods)
    transfer.to_csv(TABLES / "transfer.csv", index=False, float_format="%.4f")
    decisions = decision_table(transfer, pairs, hits, correlations)
    decisions.to_csv(TABLES / "decisions.csv", index=False)
    print(decisions.to_string(index=False), end="\n\n")
    columns = [
        "task",
        "method",
        "auroc",
        "auroc_low",
        "auroc_high",
        "balanced_accuracy",
        "macro_f1",
    ]
    print(methods[columns].to_string(index=False))


if __name__ == "__main__":
    main()
