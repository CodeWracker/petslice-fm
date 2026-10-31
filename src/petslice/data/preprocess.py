"""Resample PET/CT studies to a common physical grid and cache them for slice-level access.

Every study becomes a stack of 224x224 axial slices at 2.5 x 2.5 x 3 mm, centered on the body. The
cache also keeps rotated PET maximum intensity projections and a coarse volume for 3D baselines.
Lesion volume per slice is computed on the original mask, so small lesions do not vanish when the
mask is resampled.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage

TARGET_XY_MM = 2.5
TARGET_Z_MM = 3.0
SLICE_SIZE = 224
N_MIP_VIEWS = 16
COARSE_SHAPE = (192, 112, 112)
CT_RANGE = (-1024, 3000)


@dataclass(frozen=True)
class StudyPaths:
    ct: Path
    pet: Path
    label: Path

    @classmethod
    def from_nnunet(cls, root: Path, study_id: str) -> "StudyPaths":
        return cls(
            root / "imagesTr" / f"{study_id}_0000.nii.gz",
            root / "imagesTr" / f"{study_id}_0001.nii.gz",
            root / "labelsTr" / f"{study_id}.nii.gz",
        )


def to_zyx(image: nib.Nifti1Image) -> tuple[np.ndarray, tuple[float, float, float]]:
    """Return (Z, Y, X) data with superior-last slices, anterior on top and spacing (z, y, x)."""
    canonical = nib.as_closest_canonical(image)
    x_mm, y_mm, z_mm = (float(v) for v in canonical.header.get_zooms()[:3])
    data = np.asanyarray(canonical.dataobj, dtype=np.float32)
    return np.flip(data.transpose(2, 1, 0), axis=1).copy(), (z_mm, y_mm, x_mm)


def body_center(ct: np.ndarray) -> tuple[float, float]:
    """Centroid (row, col) of the largest air-separated component across the whole scan."""
    projection = (ct > -300).any(axis=0)
    labels, count = ndimage.label(ndimage.binary_opening(projection, iterations=3))
    if count == 0:
        return (ct.shape[1] - 1) / 2, (ct.shape[2] - 1) / 2
    largest = np.argmax(ndimage.sum_labels(np.ones_like(labels), labels, range(1, count + 1))) + 1
    rows, cols = np.nonzero(labels == largest)
    return float(rows.mean()), float(cols.mean())


def resample(volume: np.ndarray, spacing, center, fill: float) -> torch.Tensor:
    """Resample (Z, Y, X) to the target grid, cropped to SLICE_SIZE around the given center."""
    z_mm, y_mm, x_mm = spacing
    n_slices = max(1, round(volume.shape[0] * z_mm / TARGET_Z_MM))
    half = (SLICE_SIZE - 1) / 2 * TARGET_XY_MM
    tensor = torch.from_numpy(volume)[None, None]
    grid_z = (torch.arange(n_slices) + 0.5) / n_slices * 2 - 1
    # Normalized in-plane coordinates of the target crop, in source pixel units.
    offsets = torch.linspace(-half, half, SLICE_SIZE)
    grid_y = ((center[0] + offsets / y_mm) + 0.5) / volume.shape[1] * 2 - 1
    grid_x = ((center[1] + offsets / x_mm) + 0.5) / volume.shape[2] * 2 - 1
    zz, yy, xx = torch.meshgrid(grid_z, grid_y, grid_x, indexing="ij")
    grid = torch.stack((xx, yy, zz), dim=-1)[None]
    sampled = F.grid_sample(
        tensor - fill, grid.float(), mode="bilinear", padding_mode="zeros", align_corners=False
    )
    return (sampled + fill)[0, 0]


def rotated_mips(pet: torch.Tensor, n_views: int = N_MIP_VIEWS) -> torch.Tensor:
    """Maximum intensity projections around the cranio-caudal axis, over 180 degrees."""
    size = pet.shape[-1]
    slices = pet[:, None]
    views = []
    for angle in torch.arange(n_views) * torch.pi / n_views:
        cos, sin = torch.cos(angle), torch.sin(angle)
        theta = torch.tensor([[cos, -sin, 0.0], [sin, cos, 0.0]])[None]
        theta = theta.expand(slices.shape[0], -1, -1)
        grid = F.affine_grid(theta, (slices.shape[0], 1, size, size), align_corners=False)
        rotated = F.grid_sample(slices, grid, mode="bilinear", align_corners=False)
        views.append(rotated[:, 0].amax(dim=1))
    return torch.stack(views)


def lesion_ml_per_slice(label: np.ndarray, spacing, n_slices: int) -> np.ndarray:
    """Lesion volume in mL assigned to each target slice by its cranio-caudal position."""
    voxel_ml = float(np.prod(spacing)) / 1000
    per_source_slice = (label > 0).sum(axis=(1, 2)) * voxel_ml
    positions = ((np.arange(label.shape[0]) + 0.5) * spacing[0] / TARGET_Z_MM).astype(int)
    return np.bincount(
        np.clip(positions, 0, n_slices - 1), weights=per_source_slice, minlength=n_slices
    )


def uptake_features(pet: np.ndarray, body_mask: np.ndarray) -> dict[str, float]:
    """Hand-crafted features used by the trivial reference baseline."""
    body = pet[body_mask]
    features = {
        "suv_max": float(pet.max()),
        "suv_mean": float(body.mean()),
        "suv_p999": float(np.percentile(body, 99.9)),
    }
    for threshold in (2.5, 4.0, 10.0):
        features[f"frac_suv_gt_{threshold:g}"] = float((body > threshold).mean())
    return features


def process_study(paths: StudyPaths, out_dir: Path) -> dict:
    ct_image, pet_image = nib.load(paths.ct), nib.load(paths.pet)
    ct, spacing = to_zyx(ct_image)
    pet, pet_spacing = to_zyx(pet_image)
    label, _ = to_zyx(nib.load(paths.label))
    if ct.shape != pet.shape or ct.shape != label.shape:
        raise ValueError(f"shape mismatch: {ct.shape} {pet.shape} {label.shape}")
    if not np.allclose(spacing, pet_spacing, atol=1e-2):
        raise ValueError(f"spacing mismatch: {spacing} {pet_spacing}")

    center = body_center(ct)
    ct_grid = resample(ct, spacing, center, fill=-1024.0).clamp(*CT_RANGE)
    pet_grid = resample(pet, spacing, center, fill=0.0).clamp(min=0)
    mask_grid = resample(label, spacing, center, fill=0.0) >= 0.5
    n_slices = ct_grid.shape[0]
    mips = rotated_mips(pet_grid)
    coarse = F.interpolate(
        torch.stack((ct_grid, pet_grid))[None],
        size=COARSE_SHAPE,
        mode="trilinear",
        align_corners=False,
    )[0]

    body_mask = ct > -300
    lesions = lesion_ml_per_slice(label, spacing, n_slices)
    width_mm = float(np.ptp(np.nonzero(body_mask.any(axis=(0, 1)))[0]) * spacing[2])
    meta = {
        "original_shape": list(ct.shape),
        "original_spacing_zyx": list(spacing),
        "n_slices": n_slices,
        "z_extent_mm": float(ct.shape[0] * spacing[0]),
        "body_width_mm": width_mm,
        "lesion_ml": float((label > 0).sum() * np.prod(spacing) / 1000),
        "lesion_positive": bool((label > 0).any()),
        **uptake_features(pet, body_mask),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "ct.npy", ct_grid.numpy().astype(np.int16))
    np.save(out_dir / "pet.npy", pet_grid.numpy().astype(np.float16))
    np.save(out_dir / "mip.npy", mips.numpy().astype(np.float16))
    np.save(out_dir / "coarse.npy", coarse.numpy().astype(np.float16))
    np.save(out_dir / "mask.npy", mask_grid.numpy().astype(np.uint8))
    np.save(out_dir / "lesion_ml.npy", lesions.astype(np.float32))
    (out_dir / "meta.json").write_text(json.dumps(meta))
    return meta
