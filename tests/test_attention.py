import numpy as np
import pandas as pd
import pytest

from petslice.evaluation.attention import (
    chance_hit,
    correlation_test,
    study_statistics,
    summarize,
)


def test_chance_is_exact_for_simple_cases():
    assert chance_hit(10, 1, 1) == pytest.approx(0.1)
    assert chance_hit(10, 2, 2) == pytest.approx(1 - 28 / 45)
    assert chance_hit(3, 1, 10) == 1.0


def test_hit_at_k_follows_the_weight_ranking():
    lesion = np.array([0, 0, 5, 0, 0, 0.0])
    row = study_statistics(np.array([0.1, 0.5, 0.2, 0.05, 0.1, 0.05]), lesion)
    assert row["hit@1"] == 0 and row["hit@3"] == 1
    assert row["n_lesion_slices"] == 1


def test_perfect_attention_beats_chance_and_correlates():
    rng = np.random.default_rng(0)
    rows = []
    for study in range(60):
        lesion = np.zeros(100)
        lesion[rng.integers(0, 100, 5)] = rng.uniform(1, 5, 5)
        rows.append(
            {"patient_id": str(study), **study_statistics(lesion + 1e-3 * rng.random(100), lesion)}
        )
    frame = pd.DataFrame(rows)
    summary = summarize(frame)
    assert (summary["hit_low"] > summary["chance_high"]).all()
    test = correlation_test(frame)
    # Ties on lesion-free slices cap Spearman well below 1 even for perfect attention.
    assert test["median_spearman"] > 0.3 and test["wilcoxon_p"] < 0.01
