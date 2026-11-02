"""nnFoundationViT tokens for a PET/CT study.

nnFoundationViT (Primus-X, arXiv:2609.26924) is a 3D encoder pretrained on 2.1 M CT, MRI and PET
volumes with z-score normalisation and no resampling, on 192^3 patches of 8^3 tokens. Each modality
is z-scored over the volume, as in its pretraining, and tiled into 192^3 crops on the shared
2.5 x 2.5 x 3 mm grid of this study; CT and PET pass separately through the frozen encoder, every
crop is mean-pooled over its tokens, and the two streams are concatenated per crop, as for SPECTRE.
"""

import math

import numpy as np
import torch
import torch.nn.functional as F
from dynamic_network_architectures.architectures.primus import Primus
from einops import rearrange
from huggingface_hub import hf_hub_download

WEIGHTS_REPO = "MIC-DKFZ/nnFoundationViT"
WEIGHTS_REVISION = "main"
PATCH = 192
PRIMUS_X = {
    "input_channels": 1,
    "embed_dim": 1056,
    "patch_embed_size": (8, 8, 8),
    "num_classes": 1,
    "eva_depth": 40,
    "eva_numheads": 16,
    "input_shape": (PATCH, PATCH, PATCH),
    "init_values": 0.1,
    "scale_attn_inner": True,
}


def build_model(device: str = "cuda") -> Primus:
    """Patch embedding and EVA blocks only; the MAE decoder of the checkpoint is dropped."""
    model = Primus(**PRIMUS_X)
    path = hf_hub_download(WEIGHTS_REPO, "checkpoint_final.pth", revision=WEIGHTS_REVISION)
    weights = torch.load(path, map_location="cpu", weights_only=True)["network_weights"]
    encoder = {k: v for k, v in weights.items() if k.startswith(("down_projection.", "eva."))}
    missing, unexpected = model.load_state_dict(encoder, strict=False)
    unused = ("up_projection.", "mask_token")
    assert not unexpected and all(k.startswith(unused) for k in missing), (missing, unexpected)
    return model.to(device).eval()


def zscore(volume: np.ndarray) -> torch.Tensor:
    tensor = torch.from_numpy(volume.astype(np.float32))
    return (tensor - tensor.mean()) / tensor.std().clamp_min(1e-6)


def crops(volume: torch.Tensor) -> torch.Tensor:
    """(Z, H, W) -> (N, 1, 192, 192, 192): in-plane centre crop or pad, sliding windows along Z."""
    z, h, w = volume.shape
    pad = [max(0, PATCH - w), max(0, PATCH - h), max(0, PATCH - z)]
    volume = F.pad(volume, (0, pad[0], 0, pad[1], 0, pad[2]))
    z, h, w = volume.shape
    top, left = (h - PATCH) // 2, (w - PATCH) // 2
    volume = volume[:, top : top + PATCH, left : left + PATCH]
    n = max(1, math.ceil(z / PATCH))
    starts = [0] if n == 1 else [round(i * (z - PATCH) / (n - 1)) for i in range(n)]
    return torch.stack([volume[s : s + PATCH] for s in starts])[:, None]


@torch.inference_mode()
def encode(model: Primus, batch: torch.Tensor) -> torch.Tensor:
    """Mean-pooled EVA tokens per crop, (N, embed_dim)."""
    tokens = model.down_projection(batch)
    tokens = rearrange(tokens, "b c w h d -> b (w h d) c")
    tokens, _ = model.eva(tokens)
    return tokens.float().mean(dim=1)


@torch.inference_mode()
def extract(model: Primus, ct: np.ndarray, pet: np.ndarray, device: str = "cuda") -> np.ndarray:
    streams = []
    for volume in (ct, pet):
        batch = crops(zscore(volume)).to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            streams.append(torch.cat([encode(model, crop[None]) for crop in batch]))
    return torch.cat(streams, dim=-1).cpu().numpy().astype(np.float16)
