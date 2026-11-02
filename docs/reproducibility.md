# Reproducing the study

Heavy artifacts stay outside of git. By default the code expects this layout next to the repository;
`PETSLICE_DATA` and `PETSLICE_RUNS` move it elsewhere.

```
FlatPET/
├── petslice-fm/   this repository
├── data/          raw package, slice cache and cached tokens
└── runs/          one folder per method, task and fold
```

## Environment

`uv sync` recreates the environment from `uv.lock` (Python 3.12, PyTorch 2.7.1 on CUDA 12.6).

## Data

| Source | Content | Licence | Identifier |
| --- | --- | --- | --- |
| fdat `rdkqd-wdh87` | merged FDG and PSMA NIfTI package, SUV v3 | CC BY-NC 4.0 | md5 `6c69a9af7e802e1b8026c63b52931fae` |

The package ships `fdg_metadata.csv`, which holds the TCIA diagnosis labels used here, and
`psma_metadata.csv`.

```bash
uv run scripts/download_autopet.py --out-dir ../data/raw   # about 2 hours, verifies the md5
unzip ../data/raw/psma-fdg-pet-ct-lesions_v2.zip -d ../data/raw
uv run scripts/preprocess.py                               # builds the cache and the manifest
```

`manifests/studies.csv` is committed, so the folds are fixed even if a preprocessing library changes.
Regenerating it must give the same folds; `git diff manifests/` is the check.

## Pretrained weights

| Encoder | Repository | Revision | Licence |
| --- | --- | --- | --- |
| DINOv3 ViT-L/16 | `timm/vit_large_patch16_dinov3.lvd1689m` | `30c11095` | DINOv3 License |
| DINOv2 ViT-L/14 | `timm/vit_large_patch14_dinov2.lvd142m` | `4741e1ca` | Apache 2.0 |
| RAD-DINO ViT-B/14 | `microsoft/rad-dino` | `110cbc18` | MIT |
| MedSigLIP | `google/medsiglip-448`, gated | `9cea28a1` | Health AI Developer Foundations terms |
| MedImageInsight DaViT | `lion-ai/MedImageInsights` | `8b332dbc` | MIT, as stated by the mirror |
| SPECTRE Large | `cclaess/SPECTRE-Large` | `e9ad7d19` | CC BY-NC-SA 4.0 |

| Open PET/CT FM, SwinV2-Base | Dropbox link in `liu-xiaofeng/Foundation-Model-for-PET-CT` | sha256 `d067e473` | not stated |

The Dropbox folder has no revision, so `swin_mae_best_v2.pth` is identified by its sha256 prefix.
MedSigLIP needs a Hugging Face login with its terms accepted. Revisions are pinned in `src/petslice/models/`. Third-party checkpoints are read as safetensors, so
no pickle is executed.

## Running the study

```bash
uv run scripts/run_references.py
uv run scripts/run_grid.py inputs --gpus 0 1 2 3 4 5     # E2 and the E1 recipe
uv run scripts/run_grid.py volumes --gpus 0 1 2 3 4 5    # 3D ResNet, all fractions
uv run scripts/run_grid.py backbones --gpus 0 1 2 3 4 5  # E3, reads the validated input choice
uv run scripts/run_grid.py data_efficiency --gpus 0 1 2 3 4 5  # E4
uv run scripts/extract_spectre.py
uv run scripts/extract_petct_fm.py
uv run scripts/pretrain_segmentation.py --tracer fdg --fold 0   # repeat for each tracer and fold
uv run scripts/train_token_heads.py
uv run scripts/score_external.py                         # E6
uv run scripts/aggregate.py                              # tables under results/
uv run scripts/make_paper_tables.py                      # LaTeX table bodies from results/
uv run scripts/make_paper_figures.py                     # result figures from results/ (--examples needs the cache)
```

`run_grid.py` skips finished runs and each run resumes from its last epoch, so the grid can simply be
restarted after an interruption. Runs start every two minutes with capped thread pools, only on GPUs
with at least 20 GB free, and `scripts/resource_guard.py` pauses new starts near a resource limit
while logging to `runs/logs/resources.jsonl`. Start it first on shared hosts. A runner can also be
replaced while its runs are in flight: hold its pause file (`runs/logs/PAUSE` by default), start the
new runner and guard with another path in `PETSLICE_PAUSE`, and the new runner skips the live runs it
finds in `/proc` while counting them against the GPU slots. Each fold uses the seed `2027 + fold`.
Runs are not bit-exact across GPU models, because `bfloat16` autocast and non-deterministic
convolutions are left enabled for speed; the patient-level intervals are the meaningful unit of
comparison.

## Artifacts kept in the repository

`results/tables/` holds the tables behind the paper and `results/predictions/` the pooled out-of-fold
predictions. Together with `manifests/studies.csv`, they reproduce every number and figure without
retraining. Checkpoints and the cache are candidates for a Zenodo deposit.
