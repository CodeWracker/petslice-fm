import pandas as pd
import pytest

from petslice.evaluation.decisions import (
    attention_holds,
    best_baseline,
    input_verdict,
    transfer_holds,
)


def test_best_baseline_is_chosen_on_validation_only():
    methods = pd.DataFrame(
        {
            "task": ["diagnosis"] * 3,
            "method": ["resnet3d__volume__none", "spectre__tokens__frozen", "dinov3__late__lora"],
            "val_auroc": [0.80, 0.85, 0.99],
            "auroc": [0.90, 0.70, 0.95],
        }
    )
    assert best_baseline(methods, "diagnosis") == "spectre__tokens__frozen"
    with pytest.raises(ValueError):
        best_baseline(methods, "lesion_psma")


def test_transfer_margin():
    assert transfer_holds(-0.02)
    assert transfer_holds(0.1)
    assert not transfer_holds(-0.021)


def test_input_verdict_names_the_better_input_of_significant_pairs():
    pairs = pd.DataFrame(
        {
            "first": ["early", "early", "late"],
            "second": ["late", "mip", "mip"],
            "difference": [-0.05, -0.04, 0.01],
            "p_holm": [0.001, 0.002, 0.3],
        }
    )
    assert input_verdict(pairs) == "late > early; mip > early"
    assert input_verdict(pairs.assign(p_holm=[0.2, 0.3, 0.4])) == "no difference"


def test_attention_needs_every_k_above_chance_and_a_positive_correlation():
    hits = pd.DataFrame({"hit_low": [0.8, 0.85], "chance_high": [0.2, 0.4]})
    correlation = pd.Series({"median_spearman": 0.3, "wilcoxon_p": 1e-9})
    assert attention_holds(hits, correlation)
    assert not attention_holds(hits.assign(hit_low=[0.8, 0.3]), correlation)
    assert not attention_holds(hits, pd.Series({"median_spearman": -0.1, "wilcoxon_p": 1e-9}))
    assert not attention_holds(hits, pd.Series({"median_spearman": 0.1, "wilcoxon_p": 0.2}))
