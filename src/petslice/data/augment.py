"""GPU augmentation applied identically to every slice of a study."""

import math

import torch
import torch.nn.functional as F


def spatial(images: torch.Tensor) -> torch.Tensor:
    """Random flip, rotation, zoom and shift per study on a (B, S, C, H, W) batch."""
    batch, n_slices, channels, height, width = images.shape
    device = images.device
    angle = torch.empty(batch, device=device).uniform_(-math.pi / 12, math.pi / 12)
    zoom = torch.empty(batch, device=device).uniform_(0.85, 1.15)
    flip = torch.where(torch.rand(batch, device=device) < 0.5, -1.0, 1.0)
    shift = torch.empty(batch, 2, device=device).uniform_(-0.1, 0.1)
    theta = torch.zeros(batch, 2, 3, device=device)
    theta[:, 0, 0] = torch.cos(angle) * zoom * flip
    theta[:, 0, 1] = -torch.sin(angle) * zoom
    theta[:, 1, 0] = torch.sin(angle) * zoom * flip
    theta[:, 1, 1] = torch.cos(angle) * zoom
    theta[:, :, 2] = shift
    theta = theta.repeat_interleave(n_slices, dim=0)
    flat = images.reshape(batch * n_slices, channels, height, width)
    grid = F.affine_grid(theta, flat.shape, align_corners=False)
    warped = F.grid_sample(flat, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    return warped.reshape(images.shape)


def intensity(images: torch.Tensor) -> torch.Tensor:
    """Random gamma, brightness and noise per study, on channels in [0, 1]."""
    batch = images.shape[0]
    device = images.device
    shape = (batch, 1, 1, 1, 1)
    gamma = torch.where(
        torch.rand(shape, device=device) < 0.3,
        torch.empty(shape, device=device).uniform_(0.7, 1.5),
        torch.ones(shape, device=device),
    )
    brightness = torch.where(
        torch.rand(shape, device=device) < 0.15,
        torch.empty(shape, device=device).uniform_(0.75, 1.25),
        torch.ones(shape, device=device),
    )
    noise_std = torch.where(
        torch.rand(shape, device=device) < 0.25,
        torch.empty(shape, device=device).uniform_(0.0, 0.02),
        torch.zeros(shape, device=device),
    )
    images = images.clamp(1e-4, 1.0).pow(gamma) * brightness
    return (images + noise_std * torch.randn_like(images)).clamp(0.0, 1.0)


def augment(images: torch.Tensor) -> torch.Tensor:
    return intensity(spatial(images))
