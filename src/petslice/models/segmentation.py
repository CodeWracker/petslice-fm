"""nnU-Net-style 3D U-Net: pretraining by lesion segmentation and encoder token extraction."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dynamic_network_architectures.architectures.unet import PlainConvUNet

from petslice.data.dataset import SOFT_WINDOW, scale_suv, window

PATCH_SIZE = (80, 160, 160)
STAGE_FEATURES = (32, 64, 128, 256, 320, 320)
STRIDES = ((1, 1, 1), (1, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2))
EMBEDDING_DIM = STAGE_FEATURES[-1]
TOKEN_GRID = (2, 3, 3)


def build_unet(in_channels: int = 2, n_classes: int = 2) -> PlainConvUNet:
    return PlainConvUNet(
        input_channels=in_channels,
        n_stages=len(STAGE_FEATURES),
        features_per_stage=STAGE_FEATURES,
        conv_op=nn.Conv3d,
        kernel_sizes=3,
        strides=STRIDES,
        n_conv_per_stage=2,
        num_classes=n_classes,
        n_conv_per_stage_decoder=2,
        conv_bias=True,
        norm_op=nn.InstanceNorm3d,
        norm_op_kwargs={"eps": 1e-5, "affine": True},
        nonlin=nn.LeakyReLU,
        nonlin_kwargs={"inplace": True},
        deep_supervision=False,
    )


def dice_cross_entropy(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    probabilities = torch.softmax(logits, dim=1)[:, 1]
    targets = labels.float()
    intersection = (probabilities * targets).sum(dim=(1, 2, 3))
    denominator = probabilities.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    dice = (2 * intersection + 1) / (denominator + 1)
    return F.cross_entropy(logits, labels) + (1 - dice).mean()


def study_channels(ct: np.ndarray, pet: np.ndarray) -> np.ndarray:
    return np.stack((window(ct, SOFT_WINDOW), scale_suv(pet)))


@torch.inference_mode()
def encoder_tokens(encoder: nn.Module, channels: np.ndarray, device: str = "cuda") -> np.ndarray:
    """Tile a whole study along z and pool each tile's bottleneck onto a coarse grid of tokens."""
    depth = PATCH_SIZE[0]
    n_slices = channels.shape[1]
    starts = list(range(0, max(n_slices - depth, 0) + 1, depth))
    if starts[-1] + depth < n_slices:
        starts.append(n_slices - depth)
    tokens = []
    for start in starts:
        tile = torch.from_numpy(channels[:, start : start + depth].copy())[None].to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            bottleneck = encoder(tile)[-1]
        pooled = F.adaptive_avg_pool3d(bottleneck.float(), TOKEN_GRID)
        tokens.append(pooled.flatten(2).transpose(1, 2)[0])
    return torch.cat(tokens).cpu().numpy().astype(np.float16)
