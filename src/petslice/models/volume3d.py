"""3D ResNet-18 trained from scratch, the from-scratch reference of E1."""

import torch
import torch.nn as nn
from monai.networks.nets import resnet18


class Volume3DClassifier(nn.Module):
    def __init__(self, n_outputs: int, in_channels: int = 2):
        super().__init__()
        self.network = resnet18(spatial_dims=3, n_input_channels=in_channels, num_classes=n_outputs)

    def forward(self, volumes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.network(volumes)
        return logits, torch.ones(volumes.shape[0], 1, device=volumes.device)

    def trainable_state(self) -> dict[str, torch.Tensor]:
        return {key: value.cpu() for key, value in self.state_dict().items()}

    def n_trainable(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
