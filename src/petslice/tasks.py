"""Classification tasks. Each one selects a cohort and defines its integer label."""

from dataclasses import dataclass

import pandas as pd

DIAGNOSES = ("NEGATIVE", "MELANOMA", "LYMPHOMA", "LUNG_CANCER")


@dataclass(frozen=True)
class Task:
    name: str
    tracer: str
    classes: tuple[str, ...]

    @property
    def n_outputs(self) -> int:
        return 1 if len(self.classes) == 2 else len(self.classes)

    def select(self, manifest: pd.DataFrame) -> pd.DataFrame:
        cohort = manifest[manifest["tracer"] == self.tracer].copy()
        if self.name == "diagnosis":
            cohort["label"] = cohort["diagnosis"].map(self.classes.index)
        else:
            cohort["label"] = cohort["lesion_positive"].astype(int)
        return cohort.reset_index(drop=True)


TASKS = {
    "diagnosis": Task("diagnosis", "fdg", DIAGNOSES),
    "lesion_fdg": Task("lesion_fdg", "fdg", ("negative", "positive")),
    "lesion_psma": Task("lesion_psma", "psma", ("negative", "positive")),
}


def split_cohort(
    cohort: pd.DataFrame, fold: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return train, validation and test frames; patients never cross these partitions."""
    test = cohort[cohort["fold"] == fold]
    development = cohort[cohort["fold"] != fold]
    validation = development[development["val_bucket"] == fold]
    # A patient's validation bucket is unique, so dropping it keeps the partition patient-disjoint.
    train = development[development["val_bucket"] != fold]
    return train, validation, test


def subsample_train(train: pd.DataFrame, fraction: float, seed: int) -> pd.DataFrame:
    """Keep a stratified fraction of patients; smaller fractions are subsets of larger ones."""
    if fraction >= 1.0:
        return train
    patients = train.groupby("patient_id")["label"].max().sample(frac=1.0, random_state=seed)
    kept = []
    for _, group in patients.groupby(patients):
        kept.extend(group.index[: max(1, round(len(group) * fraction))])
    return train[train["patient_id"].isin(kept)]
