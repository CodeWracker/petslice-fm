"""Datasets turning the cache into encoder inputs.

Input modes:
    early     one RGB image per axial slice: [CT soft tissue, PET SUV, CT bone]
    late      six channels per slice: PET x 3 and [CT soft, CT lung, CT bone], encoded separately
    mip       rotated PET maximum intensity projections instead of slices
    pet_only  PET x 3 per slice (control)
    ct_only   [CT soft, CT lung, CT bone] per slice (control)
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from petslice.data.manifest import study_key
from petslice.data.preprocess import TARGET_XY_MM, TARGET_Z_MM

SOFT_WINDOW = (-150.0, 250.0)
LUNG_WINDOW = (-1000.0, 400.0)
BONE_WINDOW = (-450.0, 1050.0)
SUV_MAX = 15.0
SLICE_MODES = ("early", "late", "pet_only", "ct_only")
INPUT_MODES = (*SLICE_MODES, "mip")


def window(values: np.ndarray, bounds: tuple[float, float]) -> np.ndarray:
    low, high = bounds
    return np.clip((values.astype(np.float32) - low) / (high - low), 0.0, 1.0)


def scale_suv(values: np.ndarray) -> np.ndarray:
    return np.clip(values.astype(np.float32) / SUV_MAX, 0.0, 1.0)


def render_slices(ct: np.ndarray, pet: np.ndarray, mode: str) -> np.ndarray:
    """Map (S, H, W) CT and PET to (S, C, H, W) channels in [0, 1]."""
    suv = scale_suv(pet) if mode != "ct_only" else None
    if mode == "early":
        channels = [window(ct, SOFT_WINDOW), suv, window(ct, BONE_WINDOW)]
    elif mode == "pet_only":
        channels = [suv, suv, suv]
    elif mode == "ct_only":
        channels = [window(ct, w) for w in (SOFT_WINDOW, LUNG_WINDOW, BONE_WINDOW)]
    elif mode == "late":
        ct_windows = [window(ct, w) for w in (SOFT_WINDOW, LUNG_WINDOW, BONE_WINDOW)]
        channels = [suv, suv, suv, *ct_windows]
    else:
        raise ValueError(f"unknown slice mode: {mode}")
    return np.stack(channels, axis=1)


def render_mips(mips: np.ndarray, size: int) -> torch.Tensor:
    """Fit (K, Z, W) projections, head up, into square images with physical aspect ratio."""
    views = torch.from_numpy(scale_suv(np.asarray(mips))[:, ::-1].copy())[:, None]
    height_mm = views.shape[2] * TARGET_Z_MM
    width_mm = views.shape[3] * TARGET_XY_MM
    scale = size / max(height_mm, width_mm)
    height, width = round(height_mm * scale), round(width_mm * scale)
    views = F.interpolate(views, size=(height, width), mode="bilinear", align_corners=False)
    top, left = (size - height) // 2, (size - width) // 2
    views = F.pad(views, (left, size - width - left, top, size - height - top))
    return views.repeat(1, 3, 1, 1)


def sample_positions(n_total: int, n_samples: int, training: bool, stride: int) -> np.ndarray:
    """Training jitters one slice per equal segment; evaluation takes a dense regular grid."""
    if not training:
        return np.arange(stride // 2, n_total, stride)
    edges = np.linspace(0, n_total, n_samples + 1)
    return np.minimum(
        np.floor(edges[:-1] + np.random.random(n_samples) * (edges[1:] - edges[:-1])).astype(int),
        n_total - 1,
    )


class PetCtDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        root: Path,
        mode: str,
        training: bool,
        n_slices: int = 64,
        eval_stride: int = 2,
        mip_size: int = 384,
    ):
        if mode not in INPUT_MODES:
            raise ValueError(f"unknown input mode: {mode}")
        self.frame = frame.reset_index(drop=True)
        self.root = Path(root)
        self.mode = mode
        self.training = training
        self.n_slices = n_slices
        self.eval_stride = eval_stride
        self.mip_size = mip_size

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict:
        row = self.frame.iloc[index]
        folder = self.root / study_key(row["study_id"])
        if self.mode == "mip":
            images = render_mips(np.load(folder / "mip.npy"), self.mip_size)
            positions = np.arange(images.shape[0])
        else:
            ct = np.load(folder / "ct.npy", mmap_mode="r")
            pet = np.load(folder / "pet.npy", mmap_mode="r")
            positions = sample_positions(
                ct.shape[0], self.n_slices, self.training, self.eval_stride
            )
            images = torch.from_numpy(render_slices(ct[positions], pet[positions], self.mode))
        return {
            "images": images,
            "label": int(row["label"]),
            "study_id": row["study_id"],
            "positions": torch.from_numpy(positions.astype(np.int64)),
        }


class CoarseVolumeDataset(Dataset):
    """Whole studies at reduced resolution for 3D models: [CT soft tissue, PET SUV]."""

    def __init__(self, frame: pd.DataFrame, root: Path):
        self.frame = frame.reset_index(drop=True)
        self.root = Path(root)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict:
        row = self.frame.iloc[index]
        ct, pet = np.load(self.root / study_key(row["study_id"]) / "coarse.npy")
        volume = np.stack((window(ct, SOFT_WINDOW), scale_suv(pet)))
        return {
            "images": torch.from_numpy(volume),
            "label": int(row["label"]),
            "study_id": row["study_id"],
            "positions": torch.zeros(1, dtype=torch.int64),
        }
