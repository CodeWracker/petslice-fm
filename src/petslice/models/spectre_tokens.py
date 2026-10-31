"""SPECTRE tokens for a PET/CT study.

SPECTRE is a CT model. PET enters a second stream as pseudo Hounsfield units, the same kind of
out-of-domain input the 2D recipe receives, and both token grids are concatenated per crop.
"""

import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from spectre import SpectreImageFeatureExtractor

WEIGHTS_REPO = "cclaess/SPECTRE-Large"
WEIGHTS_REVISION = "e9ad7d194d6e8598345111234c5d0056d90ddc40"
CROP_IN_PLANE = 128
HU_LOW, HU_HIGH = -1000.0, 1000.0
SUV_MAX = 15.0


def build_model(device: str = "cuda") -> SpectreImageFeatureExtractor:
    """Load the safetensors release; the package default is a pickle checkpoint."""
    model = SpectreImageFeatureExtractor.from_pretrained("spectre-large", pretrained=False)
    path = hf_hub_download(WEIGHTS_REPO, "model.safetensors", revision=WEIGHTS_REVISION)
    state = {key.removeprefix("model."): value for key, value in load_file(path).items()}
    model.load_state_dict(state)
    return model.to(device).eval()


def to_spectre_grid(volume: np.ndarray) -> torch.Tensor:
    """(Z, H, W) to (1, H, W, D) with two in-plane crops per axis; D is the cranio-caudal axis."""
    tensor = torch.from_numpy(volume.astype(np.float32))[None]
    size = 2 * CROP_IN_PLANE
    tensor = F.interpolate(tensor, size=(size, size), mode="bilinear", align_corners=False)[0]
    return tensor.permute(1, 2, 0)[None]


def suv_to_pseudo_hu(suv: np.ndarray) -> np.ndarray:
    return np.clip(suv.astype(np.float32) / SUV_MAX, 0, 1) * (HU_HIGH - HU_LOW) + HU_LOW


@torch.inference_mode()
def extract(model, ct: np.ndarray, pet: np.ndarray, device: str = "cuda") -> np.ndarray:
    streams = []
    for volume in (ct, suv_to_pseudo_hu(pet)):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            streams.append(model.extract(to_spectre_grid(volume).to(device)).float())
    return torch.cat(streams, dim=-1).cpu().numpy().astype(np.float16)
