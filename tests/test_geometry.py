import nibabel as nib
import numpy as np

from petslice.data.preprocess import (
    SLICE_SIZE,
    TARGET_XY_MM,
    TARGET_Z_MM,
    body_center,
    lesion_ml_per_slice,
    resample,
    to_zyx,
)
from petslice.experiments import FOLDS, TASK_NAMES, input_runs, selected_input


def test_orientation_puts_superior_last_and_anterior_on_top():
    data = np.zeros((10, 12, 14), dtype=np.float32)
    data[2, 9, 11] = 1.0  # right-ish x, anterior y, superior z in an RAS volume
    image = nib.Nifti1Image(data, np.diag([2.0, 3.0, 4.0, 1.0]))
    array, spacing = to_zyx(image)
    z, y, x = np.argwhere(array == 1.0)[0]
    assert array.shape == (14, 12, 10)
    assert spacing == (4.0, 3.0, 2.0)
    assert z == 11 and x == 2
    assert y == 12 - 1 - 9, "anterior must map to low row indices"


def test_orientation_handles_the_las_files_of_autopet():
    data = np.zeros((10, 12, 14), dtype=np.float32)
    data[2, 9, 11] = 1.0
    affine = np.diag([-2.0, 3.0, 4.0, 1.0])
    array, _ = to_zyx(nib.Nifti1Image(data, affine))
    assert tuple(np.argwhere(array == 1.0)[0]) == (11, 2, 10 - 1 - 2)


def test_resampling_keeps_physical_position_relative_to_the_center():
    # A bright cube 30 mm right of and 20 mm above the crop center, at 2 mm spacing.
    spacing = (3.0, 2.0, 2.0)
    volume = np.zeros((30, 200, 200), dtype=np.float32)
    center = (100.0, 100.0)
    row, col = 100 - 10, 100 + 15
    volume[10:20, row - 3 : row + 4, col - 3 : col + 4] = 1.0
    grid = resample(volume, spacing, center, fill=0.0).numpy()
    assert grid.shape == (30, SLICE_SIZE, SLICE_SIZE)
    rows, cols = np.nonzero(grid[15] > 0.5)
    middle = (SLICE_SIZE - 1) / 2
    assert abs(rows.mean() - (middle - 20 / TARGET_XY_MM)) < 0.6
    assert abs(cols.mean() - (middle + 30 / TARGET_XY_MM)) < 0.6


def test_resampling_preserves_cranio_caudal_position_across_spacings():
    source = np.zeros((40, 64, 64), dtype=np.float32)
    source[20:24] = 1.0  # 100 to 120 mm at 5 mm spacing
    grid = resample(source, (5.0, 4.0, 4.0), (32.0, 32.0), fill=0.0).numpy()
    assert grid.shape[0] == round(40 * 5.0 / TARGET_Z_MM)
    slices = np.flatnonzero(grid[:, SLICE_SIZE // 2, SLICE_SIZE // 2] > 0.5)
    assert abs(slices.mean() * TARGET_Z_MM + TARGET_Z_MM / 2 - 110) < TARGET_Z_MM


def test_lesion_slices_line_up_with_the_resampled_image():
    source = np.zeros((40, 16, 16), dtype=np.float32)
    source[20:24] = 1.0
    grid = resample(source, (5.0, 4.0, 4.0), (8.0, 8.0), fill=0.0).numpy()
    per_slice = lesion_ml_per_slice(source, (5.0, 4.0, 4.0), grid.shape[0])
    image_slices = np.flatnonzero(grid[:, SLICE_SIZE // 2, SLICE_SIZE // 2] > 0.5)
    lesion_slices = np.flatnonzero(per_slice)
    assert abs(image_slices.mean() - lesion_slices.mean()) <= 1.0


def test_body_center_ignores_a_small_separate_object():
    ct = np.full((4, 100, 100), -1000.0, dtype=np.float32)
    ct[:, 30:70, 20:60] = 0.0  # body
    ct[:, 90:96, 90:96] = 0.0  # table fragment
    row, col = body_center(ct)
    assert abs(row - 49.5) < 1 and abs(col - 39.5) < 1


def test_grid_covers_every_task_input_and_fold():
    runs = input_runs()
    assert len(runs) == len(TASK_NAMES) * 5 * len(FOLDS)
    assert len({run.run_dir for run in runs}) == len(runs)


def test_input_selection_uses_validation_scores_only(tmp_path):
    import json

    for mode, score in (("early", 0.7), ("late", 0.9), ("mip", 0.8)):
        for fold in FOLDS:
            folder = tmp_path / "study" / "diagnosis" / f"dinov3__{mode}__lora" / f"fold{fold}"
            folder.mkdir(parents=True)
            metrics = {"val_auroc": score, "test": {"auroc": 1.0 if mode == "mip" else 0.0}}
            (folder / "metrics.json").write_text(json.dumps(metrics))
    assert selected_input("diagnosis", tmp_path) == "late"
