"""Low-rank adapters: W' = W + (alpha / r) B A, with B zero-initialized (AnyMC3D, Eq. 3)."""

import math
from collections.abc import Callable

import torch.nn as nn


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float):
        super().__init__()
        self.base = base
        self.lora_a = nn.Linear(base.in_features, rank, bias=False)
        self.lora_b = nn.Linear(rank, base.out_features, bias=False)
        self.scale = alpha / rank
        nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b.weight)

    def forward(self, x):
        return self.base(x) + self.lora_b(self.lora_a(x)) * self.scale


class LoRAConv2d(nn.Module):
    def __init__(self, base: nn.Conv2d, rank: int, alpha: float):
        super().__init__()
        self.base = base
        self.lora_a = nn.Conv2d(
            base.in_channels, rank, base.kernel_size, base.stride, base.padding, bias=False
        )
        self.lora_b = nn.Conv2d(rank, base.out_channels, 1, bias=False)
        self.scale = alpha / rank
        nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b.weight)

    def forward(self, x):
        return self.base(x) + self.lora_b(self.lora_a(x)) * self.scale


def inject_lora(
    model: nn.Module, is_target: Callable[[str], bool], rank: int = 8, alpha: float = 16.0
) -> list[str]:
    """Freeze `model` and wrap every targeted Linear/Conv2d. Returns the wrapped module names."""
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    wrapped = []
    for name, module in list(model.named_modules()):
        if not is_target(name):
            continue
        if isinstance(module, nn.Linear):
            replacement = LoRALinear(module, rank, alpha)
        elif isinstance(module, nn.Conv2d):
            replacement = LoRAConv2d(module, rank, alpha)
        else:
            raise TypeError(f"cannot adapt {name}: {type(module).__name__}")
        parent_name, _, child_name = name.rpartition(".")
        setattr(model.get_submodule(parent_name), child_name, replacement)
        wrapped.append(name)
    if not wrapped:
        raise ValueError("no module matched the LoRA target predicate")
    return wrapped
