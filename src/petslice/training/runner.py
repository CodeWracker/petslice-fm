"""Train one fold, select the best epoch on validation AUROC and score the test fold."""

import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from petslice.data.augment import augment
from petslice.data.dataset import PetCtDataset
from petslice.evaluation.metrics import auroc, classification_metrics, youden_threshold
from petslice.models.classifier import SliceClassifier
from petslice.models.encoders import build_encoder
from petslice.paths import MANIFEST_PATH, PROCESSED_DIR
from petslice.tasks import TASKS, Task, split_cohort, subsample_train
from petslice.training.config import RunConfig
from petslice.training.losses import focal_loss


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def build_model(config: RunConfig, task: Task, pretrained: bool = True) -> SliceClassifier:
    size = config.mip_size if config.input_mode == "mip" else config.image_size
    encoder = build_encoder(config.encoder, size, pretrained)
    return SliceClassifier(
        encoder,
        task.n_outputs,
        late_fusion=config.input_mode == "late",
        use_lora=config.adapter == "lora",
        lora_rank=config.lora_rank,
        lora_alpha=config.lora_alpha,
    )


def build_optimizer(model: SliceClassifier, config: RunConfig) -> torch.optim.Optimizer:
    lora = [p for n, p in model.encoder.named_parameters() if p.requires_grad]
    head = [*model.pool.parameters(), *model.head.parameters()]
    groups = [{"params": head, "lr": config.lr_head, "weight_decay": config.wd_head}]
    if lora:
        groups.append({"params": lora, "lr": config.lr_lora, "weight_decay": config.wd_lora})
    return torch.optim.AdamW(groups)


def to_scores(logits: torch.Tensor) -> torch.Tensor:
    if logits.shape[-1] == 1:
        return torch.sigmoid(logits)
    return torch.softmax(logits, dim=-1)


def evaluation_loader(frame, config: RunConfig) -> DataLoader:
    dataset = PetCtDataset(
        frame,
        PROCESSED_DIR,
        config.input_mode,
        training=False,
        n_slices=config.n_slices,
        eval_stride=config.eval_stride,
        mip_size=config.mip_size,
    )
    return DataLoader(dataset, batch_size=1, num_workers=config.num_workers, pin_memory=True)


@torch.inference_mode()
def predict(model: SliceClassifier, frame: pd.DataFrame, config: RunConfig, device) -> dict:
    """Score each study on its dense slice grid and keep the slice-level attention."""
    model.eval()
    scores, labels, studies, attention = [], [], [], {}
    for batch in evaluation_loader(frame, config):
        images = batch["images"].to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, weights = model(images)
        scores.append(to_scores(logits.float())[0].cpu().numpy())
        labels.append(int(batch["label"][0]))
        studies.append(batch["study_id"][0])
        attention[batch["study_id"][0]] = (
            batch["positions"][0].numpy(),
            weights[0].float().cpu().numpy(),
        )
    return {
        "study_id": studies,
        "label": np.asarray(labels),
        "scores": np.stack(scores),
        "attention": attention,
    }


def prediction_frame(result: dict, split: str, threshold: float | None) -> pd.DataFrame:
    scores = result["scores"]
    frame = pd.DataFrame(scores, columns=[f"score_{i}" for i in range(scores.shape[1])])
    frame.insert(0, "study_id", result["study_id"])
    frame.insert(1, "split", split)
    frame.insert(2, "label", result["label"])
    if scores.shape[1] == 1:
        frame["prediction"] = (scores[:, 0] >= threshold).astype(int)
    else:
        frame["prediction"] = scores.argmax(axis=1)
    return frame


def squeeze(scores: np.ndarray) -> np.ndarray:
    return scores[:, 0] if scores.shape[1] == 1 else scores


def save_attention(path: Path, attention: dict) -> None:
    arrays = {}
    for index, (study, (positions, weights)) in enumerate(attention.items()):
        arrays[f"{index}_study"] = np.array(study)
        arrays[f"{index}_positions"] = positions
        arrays[f"{index}_weights"] = weights
    np.savez_compressed(path, **arrays)


def train_run(config: RunConfig, device: str = "cuda") -> dict:
    run_dir = config.run_dir
    if (run_dir / "metrics.json").exists():
        return json.loads((run_dir / "metrics.json").read_text())
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(config.to_dict(), indent=2))

    seed_everything(config.seed + config.fold)
    task = TASKS[config.task]
    cohort = task.select(pd.read_csv(MANIFEST_PATH))
    train, validation, test = split_cohort(cohort, config.fold)
    train = subsample_train(train, config.train_fraction, config.seed + config.fold)

    model = build_model(config, task).to(device)
    optimizer = build_optimizer(model, config)
    dataset = PetCtDataset(
        train,
        PROCESSED_DIR,
        config.input_mode,
        training=True,
        n_slices=config.n_slices,
        mip_size=config.mip_size,
    )
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=config.num_workers,
        pin_memory=True,
        persistent_workers=True,
        worker_init_fn=seed_worker,
        generator=torch.Generator().manual_seed(config.seed + config.fold),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.max_epochs * len(loader)
    )

    best_auroc, best_epoch, best_state = -1.0, -1, None
    started = time.time()
    history = (run_dir / "history.jsonl").open("w")
    for epoch in range(config.max_epochs):
        model.train()
        losses = []
        for batch in loader:
            images = augment(batch["images"].to(device, non_blocking=True))
            labels = batch["label"].to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits, _ = model(images)
                loss = focal_loss(logits.float(), labels, config.focal_gamma, config.focal_alpha)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
            losses.append(loss.item())
        result = predict(model, validation, config, device)
        validation_auroc = auroc(result["label"], squeeze(result["scores"]))
        if not np.isfinite(validation_auroc):
            raise ValueError("validation AUROC is undefined; check the split and the model output")
        record = {
            "epoch": epoch,
            "loss": float(np.mean(losses)),
            "val_auroc": validation_auroc,
            "seconds": time.time() - started,
        }
        history.write(json.dumps(record) + "\n")
        history.flush()
        print(record, flush=True)
        if validation_auroc > best_auroc:
            best_auroc, best_epoch = validation_auroc, epoch
            best_state = model.trainable_state()
            torch.save(best_state, run_dir / "best.pt")
        elif epoch - best_epoch >= config.patience:
            break
    history.close()
    train_seconds = time.time() - started

    model.load_state_dict(best_state, strict=False)
    validation_result = predict(model, validation, config, device)
    labels, scores = validation_result["label"], squeeze(validation_result["scores"])
    threshold = youden_threshold(labels, scores) if scores.ndim == 1 else None
    test_result = predict(model, test, config, device)
    frames = [
        prediction_frame(validation_result, "val", threshold),
        prediction_frame(test_result, "test", threshold),
    ]
    pd.concat(frames).to_parquet(run_dir / "predictions.parquet")
    save_attention(run_dir / "attention.npz", test_result["attention"])

    metrics = {
        "test": classification_metrics(
            test_result["label"], squeeze(test_result["scores"]), threshold
        ),
        "val_auroc": best_auroc,
        "best_epoch": best_epoch,
        "epochs_run": epoch + 1,
        "threshold": threshold,
        "n_train": len(train),
        "n_val": len(validation),
        "n_test": len(test),
        "trainable_parameters": model.n_trainable(),
        "train_seconds": train_seconds,
        "peak_gpu_bytes": torch.cuda.max_memory_allocated(),
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics
