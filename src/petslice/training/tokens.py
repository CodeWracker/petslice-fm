"""Attention pooling and a linear head trained on cached tokens from frozen 3D encoders."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from petslice.data.manifest import study_key
from petslice.evaluation.metrics import auroc, classification_metrics, youden_threshold
from petslice.models.classifier import AttentionPool
from petslice.paths import MANIFEST_PATH, PROCESSED_DIR, RUNS_ROOT
from petslice.tasks import TASKS, split_cohort, subsample_train
from petslice.training.losses import focal_loss
from petslice.training.runner import prediction_frame, seed_everything, squeeze, to_scores

TOKEN_ROOT = PROCESSED_DIR.parent / "tokens"
BATCH_SIZE = 16
MAX_EPOCHS = 200
PATIENCE = 30
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4


class TokenHead(torch.nn.Module):
    def __init__(self, dim: int, n_outputs: int):
        super().__init__()
        self.norm = torch.nn.LayerNorm(dim)
        self.pool = AttentionPool(dim)
        self.head = torch.nn.Linear(dim, n_outputs)

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Pool (B, T, D) tokens; `mask` marks real tokens in padded batches."""
        normalized = self.norm(tokens)
        scores = normalized @ self.pool.query / self.pool.scale
        weights = torch.softmax(scores.masked_fill(~mask, float("-inf")), dim=1)
        return self.head((weights.unsqueeze(-1) * normalized).sum(dim=1))


def load_tokens(frame: pd.DataFrame, root: Path) -> list[torch.Tensor]:
    return [
        torch.from_numpy(np.load(root / f"{study_key(study)}.npy").astype(np.float32))
        for study in frame["study_id"]
    ]


def pad(tokens: list[torch.Tensor], device) -> tuple[torch.Tensor, torch.Tensor]:
    longest = max(len(t) for t in tokens)
    batch = torch.zeros(len(tokens), longest, tokens[0].shape[-1], device=device)
    mask = torch.zeros(len(tokens), longest, dtype=torch.bool, device=device)
    for index, item in enumerate(tokens):
        batch[index, : len(item)] = item.to(device)
        mask[index, : len(item)] = True
    return batch, mask


@torch.inference_mode()
def score(model: TokenHead, tokens: list[torch.Tensor], device) -> np.ndarray:
    model.eval()
    outputs = []
    for start in range(0, len(tokens), 64):
        batch, mask = pad(tokens[start : start + 64], device)
        outputs.append(to_scores(model(batch, mask)).cpu().numpy())
    return np.concatenate(outputs)


def train_tokens(
    task_name: str,
    method: str,
    fold: int,
    root: Path | None = None,
    train_fraction: float = 1.0,
    group: str = "study",
    seed: int = 2027,
    device: str = "cuda",
) -> dict:
    """`method` names the token folder; fold-specific encoders keep one folder per fold."""
    name = method if train_fraction >= 1.0 else f"{method}__frac{train_fraction:g}"
    output = RUNS_ROOT / group / task_name / name / f"fold{fold}"
    if (output / "metrics.json").exists():
        return json.loads((output / "metrics.json").read_text())
    output.mkdir(parents=True, exist_ok=True)
    root = root or TOKEN_ROOT / method
    seed_everything(seed + fold)

    task = TASKS[task_name]
    cohort = task.select(pd.read_csv(MANIFEST_PATH))
    train, validation, test = split_cohort(cohort, fold)
    train = subsample_train(train, train_fraction, seed + fold)
    train_tokens_ = load_tokens(train, root)
    validation_tokens = load_tokens(validation, root)
    test_tokens = load_tokens(test, root)

    model = TokenHead(train_tokens_[0].shape[-1], task.n_outputs).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    labels = torch.tensor(train["label"].to_numpy(), device=device)
    best, best_epoch, best_state = -1.0, -1, None
    for epoch in range(MAX_EPOCHS):
        model.train()
        order = torch.randperm(len(train_tokens_)).tolist()
        for start in range(0, len(order) - BATCH_SIZE + 1, BATCH_SIZE):
            indices = order[start : start + BATCH_SIZE]
            batch, mask = pad([train_tokens_[i] for i in indices], device)
            loss = focal_loss(model(batch, mask), labels[indices])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        current = auroc(
            validation["label"].to_numpy(), squeeze(score(model, validation_tokens, device))
        )
        if current > best:
            best, best_epoch = current, epoch
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        elif epoch - best_epoch >= PATIENCE:
            break

    model.load_state_dict(best_state)
    val_scores = score(model, validation_tokens, device)
    test_scores = score(model, test_tokens, device)
    val_labels = validation["label"].to_numpy()
    threshold = (
        youden_threshold(val_labels, squeeze(val_scores)) if val_scores.shape[1] == 1 else None
    )
    frames = []
    for split, part, scores in (("val", validation, val_scores), ("test", test, test_scores)):
        result = {
            "study_id": part["study_id"].tolist(),
            "label": part["label"].to_numpy(),
            "scores": scores,
        }
        frames.append(prediction_frame(result, split, threshold))
    pd.concat(frames).to_parquet(output / "predictions.parquet")
    metrics = {
        "test": classification_metrics(test["label"].to_numpy(), squeeze(test_scores), threshold),
        "val_auroc": best,
        "best_epoch": best_epoch,
        "epochs_run": epoch + 1,
        "threshold": threshold,
        "trainable_parameters": sum(p.numel() for p in model.parameters()),
    }
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics
