import numpy as np
import pandas as pd
import torch

from petslice.data.augment import augment, spatial
from petslice.data.dataset import render_mips, render_slices, sample_positions
from petslice.data.manifest import assign_folds
from petslice.data.preprocess import lesion_ml_per_slice, rotated_mips
from petslice.tasks import TASKS, split_cohort, subsample_train


def synthetic_manifest(n_patients=200):
    rng = np.random.default_rng(0)
    rows = []
    for patient in range(n_patients):
        diagnosis = rng.choice(["NEGATIVE", "MELANOMA", "LYMPHOMA", "LUNG_CANCER"])
        for study in range(rng.integers(1, 3)):
            rows.append(
                {
                    "study_id": f"fdg_{patient}_{study}",
                    "tracer": "fdg",
                    "patient_id": str(patient),
                    "diagnosis": diagnosis,
                    "scanner": "Biograph_mCT",
                    "lesion_positive": diagnosis != "NEGATIVE",
                }
            )
    return pd.DataFrame(rows)


def test_folds_never_split_a_patient():
    studies = assign_folds(synthetic_manifest())
    assert (studies.groupby("patient_id")["fold"].nunique() == 1).all()
    assert (studies.groupby("patient_id")["val_bucket"].nunique() == 1).all()


def test_splits_are_patient_disjoint_and_exhaustive():
    cohort = TASKS["diagnosis"].select(assign_folds(synthetic_manifest()))
    for fold in range(5):
        train, validation, test = split_cohort(cohort, fold)
        sets = [set(part["patient_id"]) for part in (train, validation, test)]
        assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
        assert len(train) + len(validation) + len(test) == len(cohort)


def test_train_fractions_are_nested_and_keep_all_classes():
    cohort = TASKS["diagnosis"].select(assign_folds(synthetic_manifest()))
    train, _, _ = split_cohort(cohort, 0)
    previous = set()
    for fraction in (0.1, 0.25, 0.5, 1.0):
        subset = subsample_train(train, fraction, seed=1)
        assert previous <= set(subset["patient_id"])
        assert subset["label"].nunique() == 4
        previous = set(subset["patient_id"])


def test_training_positions_cover_the_scan_without_duplicates():
    np.random.seed(0)
    positions = sample_positions(300, 64, training=True, stride=2)
    assert len(positions) == 64 and (np.diff(positions) >= 0).all() and positions.max() < 300
    assert sample_positions(10, 64, training=False, stride=2).tolist() == [1, 3, 5, 7, 9]


def test_early_fusion_channel_order():
    ct = np.full((2, 4, 4), 50, dtype=np.int16)
    pet = np.full((2, 4, 4), 7.5, dtype=np.float16)
    image = render_slices(ct, pet, "early")
    assert image.shape == (2, 3, 4, 4)
    assert np.allclose(image[:, 0], 0.5) and np.allclose(image[:, 1], 0.5)
    assert np.allclose(image[:, 2], 0.3333333, atol=1e-5)


def test_late_fusion_puts_pet_first():
    ct = np.zeros((1, 4, 4), dtype=np.int16)
    pet = np.full((1, 4, 4), 15.0, dtype=np.float16)
    image = render_slices(ct, pet, "late")
    assert image.shape == (1, 6, 4, 4) and np.allclose(image[:, :3], 1.0)


def test_mip_keeps_a_hot_spot_and_is_square():
    pet = torch.zeros(20, 64, 64)
    pet[10, 32, 32] = 15.0
    mips = rotated_mips(pet, n_views=4)
    assert mips.shape == (4, 20, 64) and (mips.amax(dim=(1, 2)) > 5).all()
    images = render_mips(mips.numpy().astype(np.float16), size=96)
    assert images.shape == (4, 3, 96, 96)
    assert 0 <= images.min() and images.max() <= 1


def test_lesion_volume_is_conserved_across_resampling():
    label = np.zeros((10, 8, 8))
    label[3, 2:4, 2:4] = 1
    label[7, 1, 1] = 1
    spacing = (5.0, 2.0, 2.0)
    per_slice = lesion_ml_per_slice(label, spacing, n_slices=17)
    assert np.isclose(per_slice.sum(), 5 * 0.02)


def test_spatial_augmentation_is_shared_by_all_slices_of_a_study():
    torch.manual_seed(0)
    volume = torch.rand(1, 1, 3, 32, 32).repeat(1, 5, 1, 1, 1)
    warped = spatial(volume)
    assert torch.allclose(warped[:, 0], warped[:, 4])


def test_augmentation_preserves_shape_and_range():
    torch.manual_seed(0)
    output = augment(torch.rand(2, 4, 3, 32, 32))
    assert output.shape == (2, 4, 3, 32, 32) and output.min() >= 0 and output.max() <= 1
