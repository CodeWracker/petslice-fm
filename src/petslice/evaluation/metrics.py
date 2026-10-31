"""Classification metrics and patient-level bootstrap."""

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score, roc_curve

N_BOOTSTRAP = 2000
BOOTSTRAP_SEED = 2027


def auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Binary AUROC, or macro one-vs-rest for an (N, C) score matrix."""
    if scores.ndim == 1:
        return float(roc_auc_score(labels, scores))
    classes = range(scores.shape[1])
    return float(np.mean([roc_auc_score(labels == c, scores[:, c]) for c in classes]))


def youden_threshold(labels: np.ndarray, scores: np.ndarray) -> float:
    fpr, tpr, thresholds = roc_curve(labels, scores)
    return float(thresholds[np.argmax(tpr - fpr)])


def predict_labels(scores: np.ndarray, threshold: float | None) -> np.ndarray:
    if scores.ndim == 1:
        return (scores >= threshold).astype(int)
    return scores.argmax(axis=1)


def classification_metrics(labels, scores, threshold: float | None) -> dict[str, float]:
    predicted = predict_labels(scores, threshold)
    return {
        "auroc": auroc(labels, scores),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predicted)),
        "macro_f1": float(f1_score(labels, predicted, average="macro")),
    }


def score_matrix(frame: pd.DataFrame) -> np.ndarray:
    columns = sorted(c for c in frame.columns if c.startswith("score_"))
    return frame[columns[0]].to_numpy() if len(columns) == 1 else frame[columns].to_numpy()


def _pooled_metrics(frame: pd.DataFrame) -> dict[str, float]:
    """Metrics of out-of-fold test predictions that carry their own `prediction` column."""
    labels = frame["label"].to_numpy()
    predicted = frame["prediction"].to_numpy()
    return {
        "auroc": auroc(labels, score_matrix(frame)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predicted)),
        "macro_f1": float(f1_score(labels, predicted, average="macro")),
    }


def bootstrap_indices(patients: pd.Series, rng: np.random.Generator):
    """Yield study indices resampled by patient, so repeat studies stay together."""
    groups = {p: np.asarray(i) for p, i in patients.groupby(patients).indices.items()}
    keys = list(groups)
    for _ in range(N_BOOTSTRAP):
        chosen = rng.integers(0, len(keys), len(keys))
        yield np.concatenate([groups[keys[k]] for k in chosen])


def bootstrap_summary(frame: pd.DataFrame) -> dict[str, float]:
    """Point estimates and 95% percentile intervals over patients."""
    frame = frame.reset_index(drop=True)
    summary = _pooled_metrics(frame)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    samples = []
    for indices in bootstrap_indices(frame["patient_id"], rng):
        subset = frame.iloc[indices]
        if subset["label"].nunique() == frame["label"].nunique():
            samples.append(_pooled_metrics(subset))
    table = pd.DataFrame(samples)
    for metric in list(summary):
        low, high = np.percentile(table[metric], [2.5, 97.5])
        summary[f"{metric}_low"], summary[f"{metric}_high"] = float(low), float(high)
    return summary


def paired_auroc_difference(first: pd.DataFrame, second: pd.DataFrame) -> dict[str, float]:
    """AUROC(first) - AUROC(second) over the same studies, with a paired patient bootstrap."""
    merged = first.merge(second, on=["study_id", "label", "patient_id"], suffixes=("_a", "_b"))
    if len(merged) != len(first):
        raise ValueError("methods were not evaluated on the same studies")
    columns_a = sorted(c for c in merged if c.startswith("score_") and c.endswith("_a"))
    columns_b = sorted(c for c in merged if c.startswith("score_") and c.endswith("_b"))

    def difference(frame: pd.DataFrame) -> float:
        labels = frame["label"].to_numpy()
        a, b = frame[columns_a].to_numpy(), frame[columns_b].to_numpy()
        if a.shape[1] == 1:
            a, b = a[:, 0], b[:, 0]
        return auroc(labels, a) - auroc(labels, b)

    point = difference(merged)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    samples = []
    for indices in bootstrap_indices(merged["patient_id"], rng):
        subset = merged.iloc[indices]
        if subset["label"].nunique() == merged["label"].nunique():
            samples.append(difference(subset))
    low, high = np.percentile(samples, [2.5, 97.5])
    return {"difference": point, "low": float(low), "high": float(high)}
