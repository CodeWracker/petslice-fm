"""Foundation encoders exposing one embedding per image.

Weights are pinned to a repository revision. Vision transformers return their class token; DaViT
has none, so MedImageInsight returns its globally pooled, normalized features.
"""

import re

import timm
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download, snapshot_download
from safetensors.torch import load_file

from petslice.models.davit import DaViT

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
TIMM_VIT_LORA = re.compile(r"^(patch_embed\.proj|blocks\.\d+\.attn\.(qkv|proj))$")
HF_DINOV2_LORA = re.compile(
    r"^(embeddings\.patch_embeddings\.projection"
    r"|encoder\.layer\.\d+\.attention\.(q_proj|k_proj|v_proj|o_proj))$"
)
DAVIT_LORA = re.compile(
    r"^(convs\.\d+\.proj|blocks\..*\.(channel_attn|window_attn)\.fn\.(qkv|proj))$"
)

TIMM_WEIGHTS = {
    "dinov3": ("vit_large_patch16_dinov3.lvd1689m", "30c1109559f65dea34316b0d4842d35c5771fe11"),
    "dinov2": ("vit_large_patch14_dinov2.lvd142m", "4741e1cafbf45415e77074bb0cb42dba76c8684a"),
}
RAD_DINO_REPO = "microsoft/rad-dino"
RAD_DINO_REVISION = "110cbc18d5133582e320b43d53bf5c44e410c936"
MEDIMAGEINSIGHT_REPO = "lion-ai/MedImageInsights"
MEDIMAGEINSIGHT_REVISION = "8b332dbcc542ca6b84b955269e547533960ab197"
MEDIMAGEINSIGHT_FILE = "2024.09.27/vision_model/medimageinsigt-v1.0.0.pt"
MEDIMAGEINSIGHT_SIZE = 480


class FoundationEncoder(nn.Module):
    """Normalizes images in [0, 1] and returns (N, D) embeddings."""

    lora_pattern: re.Pattern
    module_prefix: str = "model."

    def __init__(self, model: nn.Module, dim: int, mean, std):
        super().__init__()
        self.model = model
        self.dim = dim
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1), persistent=False)

    def lora_target(self, name: str) -> bool:
        return self.lora_pattern.match(name.removeprefix(self.module_prefix)) is not None

    def enable_checkpointing(self) -> None:
        pass

    def embed(self, images: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.embed((images - self.mean) / self.std)


class TimmViTEncoder(FoundationEncoder):
    lora_pattern = TIMM_VIT_LORA

    def __init__(self, name: str, image_size: int, pretrained: bool = True):
        timm_name, revision = TIMM_WEIGHTS[name]
        overlay = {"hf_hub_id": f"timm/{timm_name}@{revision}"}
        model = timm.create_model(
            timm_name,
            pretrained=pretrained,
            num_classes=0,
            img_size=image_size,
            pretrained_cfg_overlay=overlay,
        )
        super().__init__(model, model.embed_dim, IMAGENET_MEAN, IMAGENET_STD)

    def enable_checkpointing(self) -> None:
        self.model.set_grad_checkpointing(True)

    def embed(self, images: torch.Tensor) -> torch.Tensor:
        return self.model.forward_features(images)[:, 0]


class RadDinoEncoder(FoundationEncoder):
    lora_pattern = HF_DINOV2_LORA

    def __init__(self):
        from transformers import AutoImageProcessor, AutoModel

        path = snapshot_download(
            RAD_DINO_REPO,
            revision=RAD_DINO_REVISION,
            allow_patterns=["config.json", "preprocessor_config.json", "model.safetensors"],
        )
        model = AutoModel.from_pretrained(path)
        processor = AutoImageProcessor.from_pretrained(path)
        super().__init__(model, model.config.hidden_size, processor.image_mean, processor.image_std)

    def embed(self, images: torch.Tensor) -> torch.Tensor:
        return self.model(pixel_values=images).last_hidden_state[:, 0]


class MedImageInsightEncoder(FoundationEncoder):
    lora_pattern = DAVIT_LORA

    def __init__(self):
        model = DaViT(
            num_classes=0,
            depths=(1, 1, 9, 1),
            patch_size=(7, 3, 3, 3),
            patch_stride=(4, 2, 2, 2),
            patch_padding=(3, 1, 1, 1),
            patch_prenorm=(False, True, True, True),
            embed_dims=(256, 512, 1024, 2048),
            base_embed_dims=(256, 512, 1024, 2048),
            num_heads=(8, 16, 32, 64),
            num_groups=(8, 16, 32, 64),
            window_size=12,
            drop_path_rate=0.0,
            enable_checkpoint=False,
            standparam=True,
        )
        path = hf_hub_download(
            MEDIMAGEINSIGHT_REPO, MEDIMAGEINSIGHT_FILE, revision=MEDIMAGEINSIGHT_REVISION
        )
        prefix = "image_encoder."
        state = {
            key.removeprefix(prefix): value
            for key, value in load_file(path).items()
            if key.startswith(prefix)
        }
        model.load_state_dict(state)
        super().__init__(model, model.dim_out, IMAGENET_MEAN, IMAGENET_STD)

    def enable_checkpointing(self) -> None:
        self.model.enable_checkpoint = True

    def embed(self, images: torch.Tensor) -> torch.Tensor:
        return self.model.forward_features(images)


ENCODER_NAMES = (*TIMM_WEIGHTS, "rad_dino", "medimageinsight")


def build_encoder(name: str, image_size: int, pretrained: bool = True) -> FoundationEncoder:
    if name in TIMM_WEIGHTS:
        return TimmViTEncoder(name, image_size, pretrained)
    if name == "rad_dino":
        return RadDinoEncoder()
    if name == "medimageinsight":
        return MedImageInsightEncoder()
    raise KeyError(f"unknown encoder: {name}")
