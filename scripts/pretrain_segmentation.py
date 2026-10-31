#!/usr/bin/env python
"""Pretrain a 3D U-Net on lesion masks of one fold's training split, then cache encoder tokens.

Only training patients are used, so the encoder never sees the fold's validation or test studies.
The schedule is shorter than nnU-Net's default 1000 epochs: 100 epochs of 250 iterations.
"""

import argparse
import json
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from petslice.data.manifest import study_key
from petslice.models.segmentation import (
    PATCH_SIZE,
    build_unet,
    dice_cross_entropy,
    encoder_tokens,
    study_channels,
)
from petslice.paths import MANIFEST_PATH, PROCESSED_DIR, RUNS_ROOT
from petslice.tasks import TASKS, split_cohort
from petslice.training.runner import seed_everything, seed_worker
from petslice.training.tokens import TOKEN_ROOT

EPOCHS = 100
ITERATIONS = 250
BATCH_SIZE = 4
FOREGROUND_PROBABILITY = 0.33


class PatchDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, length: int):
        self.keys = [study_key(study) for study in frame["study_id"]]
        self.length = length

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        folder = PROCESSED_DIR / self.keys[np.random.randint(len(self.keys))]
        ct = np.load(folder / "ct.npy", mmap_mode="r")
        pet = np.load(folder / "pet.npy", mmap_mode="r")
        mask = np.load(folder / "mask.npy", mmap_mode="r")
        shape = np.array(ct.shape)
        patch = np.array(PATCH_SIZE)
        corner = (np.random.random(3) * np.maximum(shape - patch + 1, 1)).astype(int)
        lesion_slices = np.flatnonzero(np.load(folder / "lesion_ml.npy"))
        if len(lesion_slices) and np.random.random() < FOREGROUND_PROBABILITY:
            z = int(np.random.choice(lesion_slices))
            rows, columns = np.nonzero(mask[z])
            if len(rows):
                pick = np.random.randint(len(rows))
                center = np.array([z, rows[pick], columns[pick]])
                corner = np.clip(center - patch // 2, 0, np.maximum(shape - patch, 0))
        window = tuple(slice(int(c), int(c + p)) for c, p in zip(corner, patch, strict=True))
        image = study_channels(ct[window], pet[window])
        label = np.asarray(mask[window]).astype(np.int64)
        padding = [(0, 0)] + [
            (0, max(0, p - s)) for p, s in zip(patch, image.shape[1:], strict=True)
        ]
        image = np.pad(image, padding)
        label = np.pad(label, padding[1:])
        if np.random.random() < 0.5:
            image, label = image[..., ::-1].copy(), label[..., ::-1].copy()
        return torch.from_numpy(image), torch.from_numpy(label)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracer", choices=("fdg", "psma"), required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--seed", type=int, default=2027)
    args = parser.parse_args()

    task = TASKS[f"lesion_{args.tracer}"]
    manifest = pd.read_csv(MANIFEST_PATH)
    cohort = task.select(manifest)
    train, _, _ = split_cohort(cohort, args.fold)
    output = RUNS_ROOT / "segmentation" / args.tracer / f"fold{args.fold}"
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "unet.pt"
    tokens_dir = TOKEN_ROOT / f"nnunet__tokens__frozen_{args.tracer}_fold{args.fold}"
    seed_everything(args.seed + args.fold)

    model = build_unet().cuda()
    if not checkpoint.exists():
        loader = DataLoader(
            PatchDataset(train, EPOCHS * ITERATIONS * BATCH_SIZE),
            batch_size=BATCH_SIZE,
            num_workers=6,
            pin_memory=True,
            worker_init_fn=seed_worker,
        )
        optimizer = torch.optim.SGD(
            model.parameters(), lr=0.01, momentum=0.99, nesterov=True, weight_decay=3e-5
        )
        total = EPOCHS * ITERATIONS
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda step: (1 - step / total) ** 0.9
        )
        started, losses = time.time(), []
        for step, (images, labels) in enumerate(loader):
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = dice_cross_entropy(model(images).float(), labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 12.0)
            optimizer.step()
            scheduler.step()
            losses.append(loss.item())
            if (step + 1) % ITERATIONS == 0:
                print(step + 1, np.mean(losses[-ITERATIONS:]), time.time() - started, flush=True)
        torch.save(model.state_dict(), checkpoint)
        (output / "training.json").write_text(
            json.dumps(
                {"seconds": time.time() - started, "final_loss": float(np.mean(losses[-250:]))}
            )
        )
    model.load_state_dict(torch.load(checkpoint))
    model.eval()

    tokens_dir.mkdir(parents=True, exist_ok=True)
    for study in tqdm(cohort["study_id"]):
        target = tokens_dir / f"{study_key(study)}.npy"
        if target.exists():
            continue
        folder = PROCESSED_DIR / study_key(study)
        channels = study_channels(np.load(folder / "ct.npy"), np.load(folder / "pet.npy"))
        np.save(target, encoder_tokens(model.encoder, channels))


if __name__ == "__main__":
    main()
