#!/usr/bin/env python
"""Cache tokens of the open PET/CT foundation model for every study."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from petslice.data.manifest import study_key
from petslice.models.petct_fm_tokens import build_model, extract
from petslice.paths import DATA_ROOT, MANIFEST_PATH, PROCESSED_DIR
from petslice.training.tokens import TOKEN_ROOT

CHECKPOINT = DATA_ROOT / "weights" / "petct_fm" / "swinv2base" / "swin_mae_best_v2.pth"
OUTPUT = TOKEN_ROOT / "petct_fm__tokens__frozen"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    model = build_model(args.checkpoint)
    for study in tqdm(pd.read_csv(MANIFEST_PATH)["study_id"]):
        target = OUTPUT / f"{study_key(study)}.npy"
        if target.exists():
            continue
        folder = PROCESSED_DIR / study_key(study)
        np.save(target, extract(model, np.load(folder / "ct.npy"), np.load(folder / "pet.npy")))


if __name__ == "__main__":
    main()
