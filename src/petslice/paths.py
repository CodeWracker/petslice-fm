"""Filesystem layout. Heavy artifacts live outside of the repository."""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get("PETSLICE_DATA", REPO_ROOT.parent / "data"))
RUNS_ROOT = Path(os.environ.get("PETSLICE_RUNS", REPO_ROOT.parent / "runs"))
RAW_DIR = DATA_ROOT / "raw" / "PSMA-FDG-PET-CT-Lesions_v2"
PROCESSED_DIR = DATA_ROOT / "processed" / "v1"
MANIFEST_PATH = REPO_ROOT / "manifests" / "studies.csv"
RESULTS_DIR = REPO_ROOT / "results"
