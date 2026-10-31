"""Run configuration. Defaults follow AnyMC3D where the paper states a value."""

from dataclasses import asdict, dataclass
from pathlib import Path

from petslice.paths import RUNS_ROOT


@dataclass(frozen=True)
class RunConfig:
    group: str
    task: str
    encoder: str = "dinov3"
    input_mode: str = "early"
    adapter: str = "lora"
    fold: int = 0
    train_fraction: float = 1.0
    seed: int = 2027
    n_slices: int = 64
    eval_stride: int = 2
    image_size: int = 224
    mip_size: int = 384
    batch_size: int = 2
    max_epochs: int = 50
    patience: int = 10
    lora_rank: int = 8
    lora_alpha: float = 16.0
    lr_lora: float = 1e-4
    wd_lora: float = 1e-5
    lr_head: float = 1e-3
    wd_head: float = 1e-4
    focal_gamma: float = 2.0
    focal_alpha: float = 0.25
    num_workers: int = 4

    @property
    def method(self) -> str:
        name = f"{self.encoder}__{self.input_mode}__{self.adapter}"
        return name if self.train_fraction >= 1.0 else f"{name}__frac{self.train_fraction:g}"

    @property
    def run_dir(self) -> Path:
        return RUNS_ROOT / self.group / self.task / self.method / f"fold{self.fold}"

    def to_dict(self) -> dict:
        return asdict(self)
