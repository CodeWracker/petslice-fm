import numpy as np
import pandas as pd
import torch

from petslice.evaluation.metrics import (
    auroc,
    bootstrap_summary,
    holm,
    paired_auroc_difference,
)
from petslice.training.losses import focal_loss


def predictions(noise, seed=0, n=120):
    rng = np.random.default_rng(seed)
    labels = np.arange(n) % 2
    scores = labels + noise * rng.normal(size=n)
    return pd.DataFrame(
        {
            "study_id": [f"s{i}" for i in range(n)],
            "patient_id": [f"p{i // 2}" for i in range(n)],
            "label": labels,
            "score_0": scores,
            "prediction": (scores > 0.5).astype(int),
        }
    )


def test_multiclass_auroc_is_macro_one_vs_rest():
    labels = np.array([0, 1, 2, 0, 1, 2])
    scores = np.eye(3)[labels]
    assert auroc(labels, scores) == 1.0


def test_bootstrap_interval_brackets_the_estimate():
    summary = bootstrap_summary(predictions(1.0))
    assert summary["auroc_low"] < summary["auroc"] < summary["auroc_high"]


def test_paired_difference_detects_the_better_method():
    result = paired_auroc_difference(predictions(0.3), predictions(2.0, seed=1))
    assert result["low"] > 0
    assert result["p_value"] < 0.05


def test_paired_difference_of_equivalent_methods_is_not_significant():
    result = paired_auroc_difference(predictions(1.0, seed=2), predictions(1.0, seed=3))
    assert result["low"] < 0 < result["high"]
    assert result["p_value"] > 0.05


def test_holm_adjusts_step_down_and_keeps_the_order():
    assert holm([0.01, 0.04, 0.03]) == [0.03, 0.06, 0.06]
    assert holm([0.5]) == [0.5]
    assert holm([]) == []


def test_focal_loss_downweights_easy_examples():
    labels = torch.tensor([1])
    easy, hard = torch.tensor([[4.0]]), torch.tensor([[-1.0]])
    assert focal_loss(easy, labels) < 0.01 * focal_loss(hard, labels)
    logits = torch.tensor([[5.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
    assert focal_loss(logits[:1], torch.tensor([0])) < focal_loss(logits[1:], torch.tensor([0]))
