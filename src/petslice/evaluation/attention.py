"""E5: do the slices with the highest pooling weight contain lesion?"""

from math import comb

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

from petslice.evaluation.metrics import BOOTSTRAP_SEED, N_BOOTSTRAP

TOP_K = (1, 3, 5, 10)


def chance_hit(n_slices: int, n_lesion: int, k: int) -> float:
    """Probability that k slices drawn uniformly without replacement include a lesion slice."""
    k = min(k, n_slices)
    return 1.0 - comb(n_slices - n_lesion, k) / comb(n_slices, k)


def spearman_ceiling(lesion_ml: np.ndarray) -> float:
    """Spearman reached by weights ordered exactly like the lesion volume.

    Lesion-free slices tie, so sparse lesions cap the coefficient far below 1. Reporting this bound
    keeps the observed correlation interpretable.
    """
    rng = np.random.default_rng(0)
    ideal = lesion_ml + 1e-9 * rng.random(len(lesion_ml))
    return float(spearmanr(ideal, lesion_ml).statistic)


def study_statistics(weights: np.ndarray, lesion_ml: np.ndarray) -> dict[str, float]:
    """Attention statistics of a study, on the slices it was evaluated on."""
    has_lesion = lesion_ml > 0
    order = np.argsort(-weights)
    row = {"n_slices": len(weights), "n_lesion_slices": int(has_lesion.sum())}
    for k in TOP_K:
        row[f"hit@{k}"] = float(has_lesion[order[:k]].any())
        row[f"chance@{k}"] = chance_hit(len(weights), row["n_lesion_slices"], k)
    if 0 < row["n_lesion_slices"] < len(weights) and np.ptp(weights) > 0:
        row["spearman"] = float(spearmanr(weights, lesion_ml).statistic)
        row["spearman_ceiling"] = spearman_ceiling(lesion_ml)
    else:
        row["spearman"] = row["spearman_ceiling"] = float("nan")
    return row


def summarize(studies: pd.DataFrame) -> pd.DataFrame:
    """Mean hit@k and chance with patient-level bootstrap intervals, plus a paired difference."""
    frame = studies.reset_index(drop=True)
    groups = {p: np.asarray(i) for p, i in frame.groupby("patient_id").indices.items()}
    keys = list(groups)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = [
        np.concatenate([groups[keys[i]] for i in rng.integers(0, len(keys), len(keys))])
        for _ in range(N_BOOTSTRAP)
    ]
    rows = []
    for k in TOP_K:
        hit = frame[f"hit@{k}"].to_numpy()
        chance = frame[f"chance@{k}"].to_numpy()
        hit_means = np.array([hit[d].mean() for d in draws])
        chance_means = np.array([chance[d].mean() for d in draws])
        rows.append(
            {
                "k": k,
                "hit": hit.mean(),
                "hit_low": np.percentile(hit_means, 2.5),
                "hit_high": np.percentile(hit_means, 97.5),
                "chance": chance.mean(),
                "chance_low": np.percentile(chance_means, 2.5),
                "chance_high": np.percentile(chance_means, 97.5),
            }
        )
    return pd.DataFrame(rows)


def correlation_test(studies: pd.DataFrame) -> dict[str, float]:
    values = studies["spearman"].dropna().to_numpy()
    return {
        "n_studies": len(values),
        "median_spearman": float(np.median(values)),
        "median_ceiling": float(studies["spearman_ceiling"].median()),
        "wilcoxon_p": float(wilcoxon(values).pvalue),
    }
