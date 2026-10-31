"""Foundation encoders exposing a class-token embedding per image."""

import re
from dataclasses import dataclass

import timm
import torch
import torch.nn as nn

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
TIMM_VIT_LORA = re.compile(r"^(patch_embed\.proj|blocks\.\d+\.attn\.(qkv|proj))$")


@dataclass(frozen=True)
class EncoderSpec:
    timm_name: str
    weights_revision: str | None = None
    mean: tuple[float, ...] = IMAGENET_MEAN
    std: tuple[float, ...] = IMAGENET_STD


SPECS = {
    "dinov3": EncoderSpec("vit_large_patch16_dinov3.lvd1689m"),
    "dinov2": EncoderSpec("vit_large_patch14_dinov2.lvd142m"),
}


class TimmViTEncoder(nn.Module):
    def __init__(self, spec: EncoderSpec, image_size: int, pretrained: bool = True):
        super().__init__()
        self.model = timm.create_model(
            spec.timm_name, pretrained=pretrained, num_classes=0, img_size=image_size
        )
        self.dim = self.model.embed_dim
        self.register_buffer("mean", torch.tensor(spec.mean).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(spec.std).view(1, 3, 1, 1), persistent=False)

    def lora_target(self, name: str) -> bool:
        return TIMM_VIT_LORA.match(name.removeprefix("model.")) is not None

    def enable_checkpointing(self) -> None:
        self.model.set_grad_checkpointing(True)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Encode (N, 3, H, W) images in [0, 1] into their (N, D) class tokens."""
        return self.model.forward_features((images - self.mean) / self.std)[:, 0]


def build_encoder(name: str, image_size: int, pretrained: bool = True) -> TimmViTEncoder:
    if name not in SPECS:
        raise KeyError(f"unknown encoder: {name}")
    return TimmViTEncoder(SPECS[name], image_size, pretrained)
