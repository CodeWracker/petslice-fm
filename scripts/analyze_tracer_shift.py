#!/usr/bin/env python
"""E6 follow-ups: paired test of the FDG ensemble against the within-PSMA recipe, and each FDG
fold model scored on PSMA alone, so the ensemble effect is separated from the tracer transfer."""

import numpy as np
import pandas as pd

from petslice.evaluation.metrics import bootstrap_summary, paired_auroc_difference
from petslice.experiments import FOLDS, GROUP, selected_input
from petslice.paths import MANIFEST_PATH, RESULTS_DIR, RUNS_ROOT

TABLES = RESULTS_DIR / "tables"
COLUMNS = ["study_id", "label", "patient_id", "score_0"]


def main() -> None:
    manifest = pd.read_csv(MANIFEST_PATH)[["study_id", "patient_id"]]
    source = f"dinov3__{selected_input('lesion_fdg')}__lora"
    target = f"dinov3__{selected_input('lesion_psma')}__lora"
    within = pd.read_parquet(RESULTS_DIR / "predictions" / f"lesion_psma__{target}.parquet")
    within = within[within["split"] == "test"][COLUMNS]

    singles = []
    for fold in FOLDS:
        path = RUNS_ROOT / GROUP / "fdg_to_psma" / source / f"fold{fold}" / "predictions.parquet"
        frame = pd.read_parquet(path).merge(manifest, on="study_id", validate="one_to_one")
        singles.append(frame.sort_values("study_id").reset_index(drop=True))
    ensemble = singles[0].copy()
    ensemble["score_0"] = np.mean([f["score_0"].to_numpy() for f in singles], axis=0)

    paired = paired_auroc_difference(ensemble[COLUMNS], within)
    rows = [{"comparison": "fdg_ensemble - psma_within", "source": source, **paired}]

    single_rows = []
    for fold, frame in zip(FOLDS, singles, strict=True):
        summary = bootstrap_summary(frame[[*COLUMNS, "prediction"]])
        difference = paired_auroc_difference(frame[COLUMNS], within)
        single_rows.append(
            {
                "fold": fold,
                "auroc": summary["auroc"],
                "auroc_low": summary["auroc_low"],
                "auroc_high": summary["auroc_high"],
                "difference": difference["difference"],
                "low": difference["low"],
                "high": difference["high"],
                "p_value": difference["p_value"],
            }
        )
    table = pd.DataFrame(single_rows)
    aurocs = table["auroc"].to_numpy()
    table = pd.concat(
        [
            table,
            pd.DataFrame(
                [
                    {"fold": "mean", "auroc": aurocs.mean()},
                    {"fold": "sd", "auroc": aurocs.std(ddof=1)},
                ]
            ),
        ],
        ignore_index=True,
    )
    for label, index in (("best", aurocs.argmax()), ("worst", aurocs.argmin())):
        rows.append(
            {
                "comparison": f"fdg_fold{FOLDS[index]}_{label} - psma_within",
                "source": source,
                **{k: single_rows[index][k] for k in ("difference", "low", "high", "p_value")},
            }
        )
    pd.DataFrame(rows).to_csv(TABLES / "external_psma_paired.csv", index=False, float_format="%.4f")
    table.to_csv(TABLES / "external_psma_single.csv", index=False, float_format="%.4f")
    print(pd.DataFrame(rows).to_string(index=False))
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
