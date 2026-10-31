"""Tokens from the open whole-body PET/CT foundation model of Liu et al. (arXiv 2605.21835).

Pretraining included 814 autoPET FDG scans, and the 200 held-out scans are not published, so its FDG
results are optimistic. PSMA was excluded from pretraining and is a clean comparison.
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from monai.networks.nets import SwinUNETR

TILE = (96, 128, 128)
PREFIX = "module.encoder_decoder."


def build_model(checkpoint: Path, device: str = "cuda") -> SwinUNETR:
    model = SwinUNETR(
        in_channels=2, out_channels=2, feature_size=48, depths=(2, 2, 6, 2), use_v2=True
    )
    state = torch.load(checkpoint, weights_only=True, map_location="cpu")["model_state_dict"]
    state = {
        key.removeprefix(PREFIX): value
        for key, value in state.items()
        if key.startswith(PREFIX + "swinViT")
    }
    report = model.load_state_dict(state, strict=False)
    if report.unexpected_keys or any("swinViT" in key for key in report.missing_keys):
        raise RuntimeError(f"checkpoint mismatch: {report}")
    return model.to(device).eval()


def zscore(volume: torch.Tensor) -> torch.Tensor:
    return (volume - volume.mean()) / volume.std().clamp(min=1e-8)


@torch.inference_mode()
def extract(model: SwinUNETR, ct: np.ndarray, pet: np.ndarray, device: str = "cuda") -> np.ndarray:
    """Tile a study and average the deepest Swin stage of each tile into one token."""
    channels = [torch.from_numpy(v.astype(np.float32)).to(device) for v in (ct, pet)]
    volume = torch.stack([zscore(c) for c in channels])
    size = 2 * TILE[1]
    volume = F.interpolate(volume, size=(size, size), mode="bilinear", align_corners=False)
    depth = TILE[0]
    n_slices = volume.shape[1]
    starts = list(range(0, max(n_slices - depth, 0) + 1, depth))
    if starts[-1] + depth < n_slices:
        starts.append(n_slices - depth)
    tiles = [
        volume[:, z : z + depth, y : y + TILE[1], x : x + TILE[2]]
        for z in starts
        for y in (0, TILE[1])
        for x in (0, TILE[2])
    ]
    tokens = []
    for start in range(0, len(tiles), 8):
        batch = torch.stack(tiles[start : start + 8])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            deepest = model.swinViT(batch, normalize=True)[-1]
        tokens.append(deepest.float().mean(dim=(2, 3, 4)))
    return torch.cat(tokens).cpu().numpy().astype(np.float16)
