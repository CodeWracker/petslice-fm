#!/usr/bin/env python
"""Build the slice cache and the study manifest with patient-level folds."""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

from petslice.data.manifest import (
    assign_folds,
    load_fdg_metadata,
    load_psma_metadata,
    study_key,
)
from petslice.data.preprocess import StudyPaths, process_study
from petslice.paths import MANIFEST_PATH, PROCESSED_DIR, RAW_DIR


def work(study_id: str, raw: Path, out: Path) -> None:
    torch.set_num_threads(2)
    target = out / study_key(study_id)
    if not (target / "meta.json").exists():
        process_study(StudyPaths.from_nnunet(raw, study_id), target)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=RAW_DIR)
    parser.add_argument("--out", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--limit", type=int, help="Process only the first N available studies.")
    args = parser.parse_args()

    studies = pd.concat(
        [
            load_fdg_metadata(args.raw / "fdg_metadata.csv"),
            load_psma_metadata(args.raw / "psma_metadata.csv"),
        ],
        ignore_index=True,
    )
    available = studies["study_id"].map(
        lambda study: StudyPaths.from_nnunet(args.raw, study).label.exists()
    )
    print(f"{available.sum()} of {len(studies)} studies found under {args.raw}")
    studies = studies[available].reset_index(drop=True)
    if args.limit:
        studies = studies.head(args.limit)

    failures = {}
    with ProcessPoolExecutor(args.workers) as pool:
        futures = {
            pool.submit(work, study, args.raw, args.out): study for study in studies["study_id"]
        }
        for future in tqdm(as_completed(futures), total=len(futures)):
            if future.exception():
                failures[futures[future]] = repr(future.exception())
    if failures:
        raise SystemExit(f"{len(failures)} studies failed: {list(failures.items())[:5]}")

    meta = [
        json.loads((args.out / study_key(study) / "meta.json").read_text())
        for study in studies["study_id"]
    ]
    columns = [
        "n_slices",
        "z_extent_mm",
        "body_width_mm",
        "lesion_ml",
        "lesion_positive",
        "suv_max",
        "suv_mean",
        "suv_p999",
        "frac_suv_gt_2.5",
        "frac_suv_gt_4",
        "frac_suv_gt_10",
    ]
    studies = pd.concat([studies, pd.DataFrame(meta)[columns]], axis=1)
    if not args.limit:
        MANIFEST_PATH.parent.mkdir(exist_ok=True)
        assign_folds(studies).to_csv(MANIFEST_PATH, index=False)
        print(f"wrote {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
