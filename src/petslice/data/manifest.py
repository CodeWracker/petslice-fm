"""Study-level manifest: identifiers, labels, acquisition metadata and patient-level folds."""

import hashlib

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

N_FOLDS = 5
N_VAL_BUCKETS = 8
SPLIT_SEED = 2027


def study_key(study_id: str) -> str:
    """Filesystem-safe key; autoPET FDG study names contain spaces."""
    return hashlib.sha1(study_id.encode()).hexdigest()[:16]


def parse_age(value) -> float:
    if isinstance(value, str):
        return float(value.rstrip("Y"))
    return float(value)


def load_fdg_metadata(path) -> pd.DataFrame:
    rows = pd.read_csv(path)
    rows = rows[rows["Modality"] == "PT"].copy()
    folders = rows["File Location"].str.split("/").str[3]
    patients = rows["Subject ID"].str.removeprefix("PETCT_")
    description = rows["Study Description"]
    return pd.DataFrame(
        {
            "study_id": "fdg_" + patients + "_" + folders,
            "tracer": "fdg",
            "patient_id": patients,
            "diagnosis": rows["diagnosis"],
            "age": rows["age"].map(parse_age),
            "sex": rows["sex"],
            "scanner": "Biograph_mCT",
            "radionuclide": "18F",
            "ct_contrast": np.where(description.str.contains("mit KM"), "yes", "no"),
            "partial_body": description.str.contains("Teilkoerper"),
        }
    )


def load_psma_metadata(path) -> pd.DataFrame:
    rows = pd.read_csv(path)
    patients = rows["Subject ID"].str.removeprefix("PSMA_")
    return pd.DataFrame(
        {
            "study_id": "psma_" + patients + "_" + rows["Study Date"],
            "tracer": "psma",
            "patient_id": patients,
            "diagnosis": "PROSTATE_CANCER",
            "age": rows["age"].astype(float),
            "sex": "M",
            "scanner": rows["manufacturer_model_name"],
            "radionuclide": rows["pet_radionuclide"],
            "ct_contrast": rows["ct_contrast_agent"],
            "partial_body": False,
        }
    )


def assign_folds(studies: pd.DataFrame) -> pd.DataFrame:
    """Assign outer folds and validation buckets, keeping each patient in a single fold.

    Strata are computed per patient from the worst-case study so that all studies of a patient
    share a stratum. The validation set of test fold k is `fold != k and val_bucket == k`.
    """
    out = studies.copy()
    out["fold"] = -1
    out["val_bucket"] = -1
    for _, cohort in out.groupby("tracer"):
        positive = cohort["lesion_positive"].astype(int)
        if cohort["tracer"].iloc[0] == "fdg":
            strata = cohort["diagnosis"]
        else:
            strata = positive.astype(str) + "_" + cohort["scanner"]
        patient_strata = strata.groupby(cohort["patient_id"]).transform("max")
        for column, splits in (("fold", N_FOLDS), ("val_bucket", N_VAL_BUCKETS)):
            seed = SPLIT_SEED + (column == "val_bucket")
            splitter = StratifiedGroupKFold(splits, shuffle=True, random_state=seed)
            index = cohort.index.to_numpy()
            parts = splitter.split(index, patient_strata, cohort["patient_id"])
            for bucket, (_, held_out) in enumerate(parts):
                out.loc[index[held_out], column] = bucket
    return out
