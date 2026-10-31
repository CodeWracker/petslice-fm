"""Frozen 2D encoder + optional LoRA + attention pooling + linear head (AnyMC3D, Eqs. 4-5)."""

import torch
import torch.nn as nn

from petslice.models.lora import inject_lora


class AttentionPool(nn.Module):
    """Single learnable query: a = softmax(H q / sqrt(d)), v = a^T H."""

    def __init__(self, dim: int):
        super().__init__()
        self.query = nn.Parameter(torch.empty(dim))
        self.scale = dim**0.5
        nn.init.trunc_normal_(self.query, std=0.02)

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        weights = torch.softmax(tokens @ self.query / self.scale, dim=1)
        return (weights.unsqueeze(-1) * tokens).sum(dim=1), weights


class SliceClassifier(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        n_outputs: int,
        late_fusion: bool = False,
        use_lora: bool = True,
        lora_rank: int = 8,
        lora_alpha: float = 16.0,
        chunk_size: int = 64,
    ):
        super().__init__()
        self.encoder = encoder
        self.late_fusion = late_fusion
        self.chunk_size = chunk_size
        self.use_lora = use_lora
        if use_lora:
            self.adapted = inject_lora(encoder, encoder.lora_target, lora_rank, lora_alpha)
            encoder.enable_checkpointing()
        else:
            self.adapted = []
            encoder.requires_grad_(False)
        pooled_dim = encoder.dim * (2 if late_fusion else 1)
        self.pool = AttentionPool(pooled_dim)
        self.head = nn.Linear(pooled_dim, n_outputs)

    def train(self, mode: bool = True):
        super().train(mode)
        self.encoder.eval()
        return self

    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        if self.training and self.use_lora:
            return self.encoder(images)
        with torch.set_grad_enabled(False):
            return torch.cat([self.encoder(c) for c in images.split(self.chunk_size)])

    def tokens(self, images: torch.Tensor) -> torch.Tensor:
        batch, n_slices, channels, height, width = images.shape
        flat = images.reshape(batch * n_slices, channels, height, width)
        if self.late_fusion:
            pet, ct = flat[:, :3], flat[:, 3:]
            embeddings = torch.cat((self.encode_images(pet), self.encode_images(ct)), dim=-1)
        else:
            embeddings = self.encode_images(flat)
        return embeddings.reshape(batch, n_slices, -1)

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pooled, weights = self.pool(self.tokens(images).float())
        return self.head(pooled), weights

    def trainable_state(self) -> dict[str, torch.Tensor]:
        names = {name for name, p in self.named_parameters() if p.requires_grad}
        return {k: v.cpu() for k, v in self.state_dict().items() if k in names}

    def n_trainable(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
