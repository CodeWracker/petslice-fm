"""The decision rules fixed in docs/analysis_plan.md, applied to the aggregated tables."""

import pandas as pd

TRANSFER_MARGIN = -0.02
ALPHA = 0.05
BASELINES = (
    "resnet3d__volume__none",
    "nnunet__tokens__frozen",
    "spectre__tokens__frozen",
    "petct_fm__tokens__frozen",
    "nnfoundation__tokens__frozen",
)


def best_baseline(methods: pd.DataFrame, task: str) -> str:
    """The 3D reference with the best mean validation AUROC; test results play no part."""
    rows = methods[(methods["task"] == task) & methods["method"].isin(BASELINES)]
    if rows.empty:
        raise ValueError(f"no 3D reference finished for {task}")
    return str(rows.sort_values("val_auroc", ascending=False)["method"].iloc[0])


def transfer_holds(low: float, margin: float = TRANSFER_MARGIN) -> bool:
    """H1: the lower bound of the paired interval for AUROC(recipe) - AUROC(baseline)."""
    return bool(low >= margin)


def input_verdict(pairs: pd.DataFrame) -> str:
    """H2 for one task: the pairwise differences that survive the Holm correction."""
    significant = pairs[pairs["p_holm"] < ALPHA]
    if significant.empty:
        return "no difference"
    verdicts = []
    for row in significant.itertuples():
        better, worse = (row.first, row.second) if row.difference > 0 else (row.second, row.first)
        verdicts.append(f"{better} > {worse}")
    return "; ".join(verdicts)


def attention_holds(hits: pd.DataFrame, correlation: pd.Series) -> bool:
    """H3: hit@k above chance with disjoint intervals at every k, and a positive median Spearman."""
    above_chance = bool((hits["hit_low"] > hits["chance_high"]).all()) and len(hits) > 0
    positive = correlation["median_spearman"] > 0 and correlation["wilcoxon_p"] < ALPHA
    return bool(above_chance and positive)
