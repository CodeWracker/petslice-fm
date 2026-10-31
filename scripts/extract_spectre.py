#!/usr/bin/env python
"""Cache SPECTRE tokens for every study."""

import warnings

import numpy as np
import pandas as pd
from tqdm import tqdm

from petslice.data.manifest import study_key
from petslice.models.spectre_tokens import build_model, extract
from petslice.paths import MANIFEST_PATH, PROCESSED_DIR
from petslice.training.tokens import TOKEN_ROOT

OUTPUT = TOKEN_ROOT / "spectre__tokens__frozen"


def main() -> None:
    warnings.filterwarnings("ignore", message="Input does not look like Hounsfield")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    model = build_model()
    for study in tqdm(pd.read_csv(MANIFEST_PATH)["study_id"]):
        target = OUTPUT / f"{study_key(study)}.npy"
        if target.exists():
            continue
        folder = PROCESSED_DIR / study_key(study)
        tokens = extract(model, np.load(folder / "ct.npy"), np.load(folder / "pet.npy"))
        np.save(target, tokens)


if __name__ == "__main__":
    main()
