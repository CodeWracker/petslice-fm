"""Logistic-regression references on metadata and on global uptake statistics.

They show how far each task can be solved without looking at images as images. A model that merely
matches them has not shown that it transfers.
"""

import json

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from petslice.evaluation.metrics import auroc, classification_metrics, youden_threshold
from petslice.paths import MANIFEST_PATH, RUNS_ROOT
from petslice.tasks import TASKS, split_cohort

UPTAKE = ["suv_max", "suv_mean", "suv_p999", "frac_suv_gt_2.5", "frac_suv_gt_4", "frac_suv_gt_10"]


def metadata_features(frame: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "z_extent_mm": frame["z_extent_mm"],
            "age": frame["age"],
            "male": (frame["sex"] == "M").astype(float),
            "contrast": (frame["ct_contrast"] == "yes").astype(float),
            "partial_body": frame["partial_body"].astype(float),
        }
    )


FEATURES = {
    "reference_metadata": metadata_features,
    "reference_scan_length": lambda frame: frame[["z_extent_mm"]],
    "reference_uptake": lambda frame: np.log1p(frame[UPTAKE] * 1000),
}


def fit_predict(model, x: pd.DataFrame) -> np.ndarray:
    probabilities = model.predict_proba(x)
    return probabilities[:, 1:] if probabilities.shape[1] == 2 else probabilities


def run_reference(task_name: str, method: str, fold: int, group: str = "study") -> dict:
    task = TASKS[task_name]
    cohort = task.select(pd.read_csv(MANIFEST_PATH))
    train, validation, test = split_cohort(cohort, fold)
    extract = FEATURES[method]
    model = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight="balanced"),
    ).fit(extract(train), train["label"])

    output = RUNS_ROOT / group / task_name / method / f"fold{fold}"
    output.mkdir(parents=True, exist_ok=True)
    val_scores, test_scores = (
        fit_predict(model, extract(validation)),
        fit_predict(model, extract(test)),
    )
    squeeze = (lambda s: s[:, 0]) if val_scores.shape[1] == 1 else (lambda s: s)
    labels = validation["label"].to_numpy()
    threshold = youden_threshold(labels, squeeze(val_scores)) if val_scores.shape[1] == 1 else None

    frames = []
    for name, part, scores in (("val", validation, val_scores), ("test", test, test_scores)):
        frame = pd.DataFrame(scores, columns=[f"score_{i}" for i in range(scores.shape[1])])
        frame.insert(0, "study_id", part["study_id"].to_numpy())
        frame.insert(1, "split", name)
        frame.insert(2, "label", part["label"].to_numpy())
        if scores.shape[1] == 1:
            frame["prediction"] = (scores[:, 0] >= threshold).astype(int)
        else:
            frame["prediction"] = scores.argmax(axis=1)
        frames.append(frame)
    pd.concat(frames).to_parquet(output / "predictions.parquet")
    metrics = {
        "test": classification_metrics(test["label"].to_numpy(), squeeze(test_scores), threshold),
        "val_auroc": auroc(labels, squeeze(val_scores)),
        "threshold": threshold,
        "trainable_parameters": int(sum(c.size for c in model[-1].coef_.flat)),
    }
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics
