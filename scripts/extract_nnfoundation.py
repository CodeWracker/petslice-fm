#!/usr/bin/env python
"""Cache nnFoundationViT tokens for every study."""

import argparse

import numpy as np
import pandas as pd
from tqdm import tqdm

from petslice.data.manifest import study_key
from petslice.models.nnfoundation_tokens import build_model, extract
from petslice.paths import MANIFEST_PATH, PROCESSED_DIR
from petslice.training.tokens import TOKEN_ROOT

OUTPUT = TOKEN_ROOT / "nnfoundation__tokens__frozen"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    model = build_model()
    studies = pd.read_csv(MANIFEST_PATH)["study_id"].tolist()[args.shard :: args.shards]
    for study in tqdm(studies, mininterval=30):
        target = OUTPUT / f"{study_key(study)}.npy"
        if target.exists():
            continue
        folder = PROCESSED_DIR / study_key(study)
        tokens = extract(model, np.load(folder / "ct.npy"), np.load(folder / "pet.npy"))
        np.save(target, tokens)


if __name__ == "__main__":
    main()
