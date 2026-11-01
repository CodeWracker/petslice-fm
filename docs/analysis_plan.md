# Analysis plan

Written before any model was trained on the full cohorts. Changes made after that point are logged
in the amendments section at the end, with their reason.

## Cohorts and tasks

| Task | Cohort | Target |
| --- | --- | --- |
| `diagnosis` | autoPET FDG, 1014 studies, 900 patients | negative, melanoma, lymphoma, lung cancer |
| `lesion_fdg` | autoPET FDG | any annotated lesion in the mask |
| `lesion_psma` | autoPET PSMA, 597 studies, 378 patients | any annotated lesion in the mask |
| `hpv` | HECKTOR | HPV status; pending data access |

## Splits

Five outer folds per tracer, grouped by patient and committed in `manifests/studies.csv`. The FDG
tasks share folds. For test fold `k`, validation is `fold != k and val_bucket == k`. The test fold
is used only once, after the best epoch is selected on validation AUROC. The autoPET
`splits_final.json` is not used: nothing in its documentation guarantees patient-level grouping.

## Metrics

AUROC (macro one-vs-rest for `diagnosis`) is primary. Balanced accuracy and macro F1 are secondary.
Binary decisions use the Youden threshold of the validation split; `diagnosis` uses argmax.
Out-of-fold test predictions are pooled, and 95% intervals come from 2000 patient-level bootstrap
resamples, so repeated studies of a patient stay together. Method differences use the same bootstrap,
paired by study.

## Hypotheses and decision rules

**H1, transfer.** The recipe transfers if the lower bound of the 95% paired interval for
ΔAUROC(recipe - best 3D baseline) is at or above -0.02 on a task. The margin was fixed in advance as
a difference too small to change a clinical or modelling decision at these sample sizes. The recipe
is DINOv3 ViT-L + LoRA with the input chosen by mean validation AUROC, never by test AUROC. The best
baseline is also chosen on validation.

**H2, input.** Early fusion, late fusion and MIP are compared pairwise per task with paired
intervals; p-values from the bootstrap are Holm-corrected over the three pairs in each task.

**H3, attention.** Evaluated on lesion-positive test studies for slice inputs, on the dense
evaluation grid. `hit@k` is the fraction of studies in which any of the k slices with the highest
weight contains lesion, for k in {1, 3, 5, 10}. Chance is the exact hypergeometric expectation for
k slices drawn uniformly from each study's grid. The second measure is the within-study Spearman
correlation between weight and lesion volume per slice, summarized by its median and tested against
zero with a Wilcoxon signed-rank test over studies. H3 holds if hit@k exceeds chance with non-
overlapping 95% intervals for every k, and the median correlation is positive with p < 0.05.

## Experiments

- **E1**, main comparison: recipe vs 3D ResNet from scratch, nnU-Net-style encoder, SPECTRE and the
  open PET/CT foundation model.
- **E2**, input: early vs late vs MIP. PET-only and CT-only slices are controls that explain the
  result; they are not part of the H2 comparison.
- **E3**, backbone: DINOv3, DINOv2, MedSigLIP, MedImageInsight, RAD-DINO.
- **E4**, data efficiency: 10, 25, 50 and 100% of training patients, nested and stratified.
- **E5**, attention: H3 above, using stored weights without further training.
- **E6**, tracer shift: `lesion_fdg` models scored on all PSMA studies with their FDG thresholds,
  compared with `lesion_psma` cross-validation as the within-domain reference.

## Controls against confounding

The FDG cohort is single-scanner and melanoma studies may extend further down the body, so scan
extent or contrast use could predict the labels without any imaging signal. A logistic regression
on metadata alone (scan length, age, sex, contrast) and another on global uptake statistics are
evaluated with the same folds. They are reported next to the deep models, because a recipe AUROC
that only matches these references would not be evidence of transfer.

Pretraining leakage: the open PET/CT foundation model (arXiv 2605.21835) pretrained on 814 autoPET
FDG scans and excluded PSMA. Its FDG results are optimistic and are flagged as such; only PSMA is a
clean comparison.

## Deviations from AnyMC3D

| Item | AnyMC3D | Here | Reason |
| --- | --- | --- | --- |
| Epochs | up to 100 | up to 50, patience 10 | compute budget across the grid |
| Slices | fixed per task, e.g. 70 | 64 in training, every 2nd slice in evaluation | whole-body scans have about 300 slices |
| Optimizer | not stated | AdamW with cosine decay | common default |
| Auxiliary segmentation | optional | not used | the proposal evaluates attention without pixel supervision |
| Augmentation | nnU-Net 3D | per-study 2D affine and intensity | applied to sampled slices on GPU |

The paper's limits on the DINOv3 result also apply here: arXiv 2509.06467 reports poor DINOv3
performance on PET for segmentation with a frozen encoder, not for classification.

## Amendments

None yet.
