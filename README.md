# petslice-fm

Whole-body PET/CT classification with frozen 2D foundation models, LoRA adapters and attention
pooling across slices. This repository applies the AnyMC3D recipe (Liu et al., CVPR 2026) to a
modality its authors left untested and studies how PET and CT should be presented to the encoder.

The study asks three questions, with decision rules fixed in
[`docs/analysis_plan.md`](docs/analysis_plan.md):

- Does the recipe transfer to PET/CT, compared with 3D models trained on this domain?
- Which PET and CT input works best: three-channel fusion, late fusion or rotated PET MIPs?
- Do the pooling weights fall on the slices that hold lesions?

## Quick start

```bash
uv sync
uv run pytest
```

[`docs/reproducibility.md`](docs/reproducibility.md) covers the data, the pinned weights and the
commands behind every result.

## Layout

| Path | Content |
| --- | --- |
| `src/petslice/data` | manifest, preprocessing, datasets, augmentation |
| `src/petslice/models` | encoders, LoRA, attention pooling, 3D baselines |
| `src/petslice/training` | run configuration, training loop, reference models |
| `src/petslice/evaluation` | bootstrap metrics and attention statistics |
| `scripts` | entry points for each stage of the study |
| `manifests` | committed study list and patient-level folds |
| `results` | tables and pooled out-of-fold predictions |
