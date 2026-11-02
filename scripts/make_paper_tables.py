#!/usr/bin/env python
"""Write the LaTeX bodies of the result tables from `results/tables/`, so no number is retyped."""

import argparse
from pathlib import Path

import pandas as pd

from petslice.paths import REPO_ROOT, RESULTS_DIR

TASKS = ("diagnosis", "lesion_fdg", "lesion_psma")
INPUTS = ("early", "late", "mip")
MAIN_ROWS = (
    ("resnet3d__volume__none", "3D ResNet-18, from scratch"),
    ("nnunet__tokens__frozen", "nnU-Net encoder, frozen"),
    ("spectre__tokens__frozen", "SPECTRE, frozen"),
    ("petct_fm__tokens__frozen", "PET/CT FM, frozen$^\\dagger$"),
    ("reference_metadata", "Metadata only"),
    ("reference_uptake", "Uptake statistics only"),
    ("reference_scan_length", "Scan length only"),
)
INPUT_ROWS = (
    ("early", "Early fusion", "$S$"),
    ("late", "Late fusion", "$2S$"),
    ("mip", "Rotated MIPs (PET only)", "16"),
    ("pet_only", "PET only (control)", "$S$"),
    ("ct_only", "CT only (control)", "$S$"),
)
BACKBONE_ROWS = (
    ("dinov3", "lora", "DINOv3 ViT-L"),
    ("dinov3", "none", "DINOv3 ViT-L, no LoRA (pooling and head only)"),
    ("dinov2", "lora", "DINOv2 ViT-L"),
    ("rad_dino", "lora", "RAD-DINO ViT-B"),
    ("medsiglip", "lora", "MedSigLIP ViT-So400m"),
    ("medimageinsight", "lora", "MedImageInsight"),
)


def selected(methods: pd.DataFrame, task: str) -> str:
    rows = methods[
        (methods["task"] == task) & methods["method"].isin(f"dinov3__{m}__lora" for m in INPUTS)
    ]
    return str(rows.sort_values("val_auroc", ascending=False)["method"].iloc[0]).split("__")[1]


def cell(methods: pd.DataFrame, task: str, method: str, bold: bool = False) -> str:
    row = methods[(methods["task"] == task) & (methods["method"] == method)]
    if row.empty:
        return "--"
    row = row.iloc[0]
    text = f"{row['auroc']:.3f} [{row['auroc_low']:.3f}, {row['auroc_high']:.3f}]"
    return f"\\textbf{{{text}}}" if bold else text


def params(methods: pd.DataFrame, method: str) -> str:
    row = methods[methods["method"] == method].iloc[0]
    value = row["trainable_parameters"]
    return (
        f"{value / 1e6:.1f}\\,M"
        if value >= 1e5
        else f"{value / 1e3:.0f}\\,k"
        if value >= 1e3
        else f"{value:.0f}"
    )


def hours(methods: pd.DataFrame, task: str, method: str) -> str:
    row = methods[(methods["task"] == task) & (methods["method"] == method)].iloc[0]
    return f"{row['gpu_hours_per_fold']:.1f}"


def main_table(methods: pd.DataFrame) -> str:
    chosen = {task: selected(methods, task) for task in TASKS}
    names = {"early": "early fusion", "late": "late fusion", "mip": "rotated MIPs"}
    lines = [
        "\\begin{tabular}{@{}l r r l l l@{}}",
        "\\toprule",
        "Method & Trained & GPU\\,h & \\multicolumn{3}{c}{AUROC [95\\% interval]} \\\\",
        "\\cmidrule(l){4-6}",
        " & param. & diag. & Diagnosis & Lesion, FDG & Lesion, PSMA \\\\",
        "\\midrule",
        "Recipe: DINOv3 + LoRA & "
        + params(methods, f"dinov3__{chosen['diagnosis']}__lora")
        + " & "
        + hours(methods, "diagnosis", f"dinov3__{chosen['diagnosis']}__lora")
        + " & "
        + " & ".join(
            cell(methods, task, f"dinov3__{chosen[task]}__lora", bold=True) for task in TASKS
        )
        + " \\\\",
        "\\quad input chosen on validation & & & "
        + " & ".join(names[chosen[task]] for task in TASKS)
        + " \\\\",
        "\\midrule",
    ]
    for method, label in MAIN_ROWS:
        hour = (
            hours(methods, "diagnosis", method)
            if "reference" not in method and "tokens" not in method
            else "--"
        )
        lines.append(
            f"{label} & {params(methods, method)} & {hour} & "
            + " & ".join(cell(methods, task, method) for task in TASKS)
            + " \\\\"
        )
        if method == "petct_fm__tokens__frozen":
            lines.append("\\midrule")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


def input_table(methods: pd.DataFrame) -> str:
    chosen = {task: selected(methods, task) for task in TASKS}
    lines = [
        "\\begin{tabular}{@{}l c r l l l@{}}",
        "\\toprule",
        "Input $T$ & Images & GPU\\,h & \\multicolumn{3}{c}{AUROC [95\\% interval]} \\\\",
        "\\cmidrule(l){4-6}",
        " & per study & diag. & Diagnosis & Lesion, FDG & Lesion, PSMA \\\\",
        "\\midrule",
    ]
    for mode, label, images in INPUT_ROWS:
        if mode == "pet_only":
            lines.append("\\midrule")
        method = f"dinov3__{mode}__lora"
        lines.append(
            f"{label} & {images} & {hours(methods, 'diagnosis', method)} & "
            + " & ".join(cell(methods, task, method, bold=chosen[task] == mode) for task in TASKS)
            + " \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


def backbone_table(methods: pd.DataFrame) -> str:
    chosen = {task: selected(methods, task) for task in TASKS}
    lines = [
        "\\begin{tabular}{@{}l r l l l@{}}",
        "\\toprule",
        "Encoder & Trained & \\multicolumn{3}{c}{AUROC [95\\% interval]} \\\\",
        "\\cmidrule(l){3-5}",
        " & param. & Diagnosis & Lesion, FDG & Lesion, PSMA \\\\",
        "\\midrule",
    ]
    for encoder, adapter, label in BACKBONE_ROWS:
        method = {task: f"{encoder}__{chosen[task]}__{adapter}" for task in TASKS}
        lines.append(
            f"{label} & {params(methods, method['diagnosis'])} & "
            + " & ".join(cell(methods, task, method[task]) for task in TASKS)
            + " \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=REPO_ROOT.parent / "cvpt-flatpet-latex/tables")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    methods = pd.read_csv(RESULTS_DIR / "tables" / "methods.csv")
    (args.out / "main.tex").write_text(main_table(methods))
    (args.out / "inputs.tex").write_text(input_table(methods))
    (args.out / "backbones.tex").write_text(backbone_table(methods))


if __name__ == "__main__":
    main()
