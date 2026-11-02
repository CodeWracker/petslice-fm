#!/usr/bin/env python
"""Draw the data-driven paper figures.

The example figures need the preprocessed cache; the result figures read `results/` only.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from petslice.data.dataset import render_mips, render_slices
from petslice.data.manifest import study_key
from petslice.data.preprocess import TARGET_XY_MM, TARGET_Z_MM, rotated_mips
from petslice.paths import PROCESSED_DIR, REPO_ROOT, RESULTS_DIR

EXAMPLE = "fdg_01140d52d8_08-13-2005-NA-PET-CT Ganzkoerper  primaer mit KM-56839"
ACCENT = "#D55E00"
BLUE = "#0072B2"
INK = "#222222"
MUTED = "#666666"
TASK_TITLES = {
    "diagnosis": "Diagnosis, FDG",
    "lesion_fdg": "Lesion presence, FDG",
    "lesion_psma": "Lesion presence, PSMA",
}
FRACTIONS = (0.1, 0.25, 0.5, 1.0)
RESNET = "resnet3d__volume__none"
MIP_VIEWS = (0, 4, 8)
FULL_WIDTH = 7.0
COLUMN_WIDTH = 3.4
BODY_WIDTH_MM = 480


def configure() -> None:
    plt.rcParams.update(
        {
            "font.family": "STIXGeneral",
            "mathtext.fontset": "stix",
            "font.size": 7.5,
            "axes.titlesize": 7.5,
            "axes.labelsize": 7.5,
            "text.color": INK,
            "axes.labelcolor": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "axes.edgecolor": MUTED,
            "pdf.fonttype": 42,
        }
    )


def load_example() -> dict:
    folder = PROCESSED_DIR / study_key(EXAMPLE)
    return {
        "ct": np.load(folder / "ct.npy"),
        "pet": np.load(folder / "pet.npy"),
        "mip": np.load(folder / "mip.npy"),
        "mask": np.load(folder / "mask.npy"),
        "lesion_ml": np.load(folder / "lesion_ml.npy"),
        "meta": json.loads((folder / "meta.json").read_text()),
    }


def problem_figure(study: dict, path: Path) -> None:
    """One label for the whole body, while the evidence sits in a few slices."""
    n_slices, _, width = study["pet"].shape
    side = (width - round(BODY_WIDTH_MM / TARGET_XY_MM)) // 2
    columns = slice(side, width - side)
    extent = (0, (width - 2 * side) * TARGET_XY_MM, 0, n_slices * TARGET_Z_MM)
    z_mm = (np.arange(n_slices) + 0.5) * TARGET_Z_MM
    lesion = study["lesion_ml"]
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(COLUMN_WIDTH * 0.82, 2.6),
        gridspec_kw={"width_ratios": [1, 0.9], "wspace": 0.05},
    )
    uptake = study["mip"][0][:, columns].astype(np.float32)
    axes[0].imshow(uptake, cmap="gray_r", vmin=0, vmax=10, origin="lower", extent=extent)
    axes[0].contour(
        study["mask"].max(axis=1)[:, columns].astype(float),
        levels=[0.5],
        colors=ACCENT,
        linewidths=0.7,
        origin="lower",
        extent=extent,
    )
    axes[0].set_title("PET")
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    for spine in axes[0].spines.values():
        spine.set_visible(False)
    runs = np.flatnonzero(np.diff(np.r_[0, (lesion > 0).astype(int), 0]))
    for start, stop in zip(runs[::2], runs[1::2], strict=True):
        for axis in axes:
            axis.axhspan(start * TARGET_Z_MM, stop * TARGET_Z_MM, color=ACCENT, alpha=0.12, lw=0)
    axes[1].barh(z_mm, lesion, height=TARGET_Z_MM, color=ACCENT, linewidth=0)
    axes[1].set_title("Lesion per slice")
    axes[1].set_xlabel("mL")
    axes[1].set_xlim(0, lesion.max() * 1.1)
    axes[1].set_ylim(0, n_slices * TARGET_Z_MM)
    axes[1].set_yticks([])
    axes[1].tick_params(left=False, length=2, width=0.5)
    for side_name in ("top", "right", "left"):
        axes[1].spines[side_name].set_visible(False)
    axes[1].spines["bottom"].set_linewidth(0.5)
    with_lesion = int((lesion > 0).sum())
    axes[1].text(
        0.97,
        0.05,
        f"{with_lesion} of {n_slices} slices\nhold a lesion",
        transform=axes[1].transAxes,
        ha="right",
        va="bottom",
    )
    figure.canvas.draw()
    image_box = axes[0].get_position()
    profile_box = axes[1].get_position()
    axes[1].set_position([profile_box.x0, image_box.y0, profile_box.width, image_box.height])
    figure.savefig(path, bbox_inches="tight", pad_inches=0.01)
    plt.close(figure)


def show(axis, image: np.ndarray, label: str, target=None, gray: bool = False) -> None:
    axis.imshow(image, cmap="gray" if gray else None, vmin=0, vmax=1)
    axis.set_title(label, pad=2)
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)
    if target is not None:
        row, column = target
        offset = -0.16 * image.shape[0]
        axis.annotate(
            "",
            xy=(column, row),
            xytext=(column + offset, row + offset),
            arrowprops={"arrowstyle": "-|>", "color": ACCENT, "lw": 1.0, "shrinkB": 5},
        )


def centroid(mask: np.ndarray) -> tuple[float, float]:
    rows, columns = np.nonzero(mask)
    return float(rows.mean()), float(columns.mean())


def input_figure(study: dict, path: Path) -> None:
    """What the encoder receives from the same study under each representation."""
    z = int(study["lesion_ml"].argmax())
    early = render_slices(study["ct"][z : z + 1], study["pet"][z : z + 1], "early")[0]
    late = render_slices(study["ct"][z : z + 1], study["pet"][z : z + 1], "late")[0]
    mips = render_mips(study["mip"], 384).numpy()
    lesion_mask = torch.from_numpy(study["mask"].astype(np.float32) * 15)
    mask_mips = render_mips(rotated_mips(lesion_mask).numpy(), 384).numpy()
    on_slice = centroid(study["mask"][z])
    figure = plt.figure(figsize=(FULL_WIDTH, 1.5))
    grid = figure.add_gridspec(1, 3, width_ratios=[1, 2, 3], wspace=0.09)
    groups = [
        grid[index].subgridspec(1, count, wspace=0.03) for index, count in enumerate((1, 2, 3))
    ]
    show(
        figure.add_subplot(groups[0][0]),
        early.transpose(1, 2, 0),
        "Magenta CT, green PET",
        on_slice,
    )
    show(figure.add_subplot(groups[1][0]), late[0], "PET", on_slice, gray=True)
    show(
        figure.add_subplot(groups[1][1]),
        late[3:].transpose(1, 2, 0),
        "CT, three windows",
        on_slice,
    )
    for slot, view in enumerate(MIP_VIEWS):
        angle = view * 180 // len(mips)
        target = centroid(mask_mips[view, 0] > 0.5)
        show(figure.add_subplot(groups[2][slot]), mips[view, 0], f"{angle}°", target, gray=True)
    titles = ("Early fusion: 1 image", "Late fusion: 2 images", "Rotated MIPs: 16 images, 3 shown")
    for index, title in enumerate(titles):
        center = grid[index].get_position(figure)
        figure.text(
            (center.x0 + center.x1) / 2,
            0.9,
            title,
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=8,
        )
    figure.savefig(path, bbox_inches="tight", pad_inches=0.01)
    plt.close(figure)


def tidy(axis) -> None:
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_linewidth(0.5)
    axis.tick_params(length=2, width=0.5)


def selected_from_table(methods: pd.DataFrame, task: str) -> str:
    """The input with the best mean validation AUROC, read from the table rather than from runs."""
    candidates = [f"dinov3__{mode}__lora" for mode in ("early", "late", "mip")]
    rows = methods[(methods["task"] == task) & methods["method"].isin(candidates)]
    return str(rows.sort_values("val_auroc", ascending=False)["method"].iloc[0])


def fraction_series(methods: pd.DataFrame, task: str, base: str) -> np.ndarray:
    rows = []
    for fraction in FRACTIONS:
        method = base if fraction == 1 else f"{base}__frac{fraction:g}"
        row = methods[(methods["task"] == task) & (methods["method"] == method)]
        if len(row) == 1:
            rows.append([fraction * 100, *row[["auroc", "auroc_low", "auroc_high"]].iloc[0]])
    return np.array(rows)


def data_efficiency_figure(methods: pd.DataFrame, path: Path) -> None:
    """The recipe keeps its margin over the 3D ResNet at every training fraction (E4)."""
    figure, axes = plt.subplots(1, 3, figsize=(FULL_WIDTH, 1.5), gridspec_kw={"wspace": 0.22})
    for axis, task in zip(axes, TASK_TITLES, strict=True):
        pairs = (
            (selected_from_table(methods, task), ACCENT, "Recipe: frozen 2D encoder + LoRA"),
            (RESNET, BLUE, "3D ResNet, from scratch"),
        )
        for method, color, label in pairs:
            x, y, low, high = fraction_series(methods, task, method).T
            axis.fill_between(x, low, high, color=color, alpha=0.15, lw=0)
            axis.plot(x, y, color=color, lw=1.2, marker="o", ms=3, label=label)
        axis.set_xscale("log")
        axis.set_xticks([10, 25, 50, 100])
        axis.set_xticklabels(["10", "25", "50", "100"])
        axis.minorticks_off()
        axis.set_xlim(8.5, 120)
        axis.set_ylim(0.3, 1.0)
        axis.axhline(0.5, color=MUTED, lw=0.5, ls=":")
        axis.text(118, 0.505, "chance", ha="right", va="bottom", color=MUTED, fontsize=6.5)
        axis.set_xlabel("Training patients (%, log scale)")
        axis.set_title(TASK_TITLES[task])
        tidy(axis)
    axes[0].set_ylabel("AUROC")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="lower center",
        ncol=2,
        frameon=False,
        handlelength=1.6,
        bbox_to_anchor=(0.5, -0.34),
    )
    figure.savefig(path, bbox_inches="tight", pad_inches=0.01)
    plt.close(figure)


def attention_figure(hits: pd.DataFrame, profile: pd.DataFrame, path: Path) -> None:
    """Pooling weights trained with study labels alone fall on the slices that hold lesions (E5)."""
    figure = plt.figure(figsize=(FULL_WIDTH, 1.75))
    grid = figure.add_gridspec(1, 2, width_ratios=[1.15, 1], wspace=0.18)
    example = grid[0].subgridspec(2, 1, hspace=0.12)
    lesion_axis = figure.add_subplot(example[0])
    weight_axis = figure.add_subplot(example[1], sharex=lesion_axis)
    study = profile[profile["study_id"] == EXAMPLE].sort_values("position")
    z = (study["position"].to_numpy() + 0.5) * TARGET_Z_MM
    width = 2 * TARGET_Z_MM
    lesion_axis.bar(z, study["lesion_ml"], width=width, color=ACCENT, lw=0)
    weight_axis.bar(z, study["weight"], width=width, color=BLUE, lw=0)
    lesion_axis.set_ylabel("Lesion (mL)")
    weight_axis.set_ylabel("Weight $a_s$")
    weight_axis.set_xlabel("Slice height from the feet (mm)")
    lesion_axis.tick_params(labelbottom=False)
    lesion_axis.set_title(
        "One FDG test study under the diagnosis model: lesions and weights by slice"
    )
    lesion = study["lesion_ml"].to_numpy()
    runs = np.flatnonzero(np.diff(np.r_[0, (lesion > 0).astype(int), 0]))
    for start, stop in zip(runs[::2], runs[1::2], strict=True):
        weight_axis.axvspan(
            z[start] - width / 2, z[stop - 1] + width / 2, color=ACCENT, alpha=0.12, lw=0
        )
    for axis in (lesion_axis, weight_axis):
        tidy(axis)
        axis.set_xlim(0, z.max() + width)
    top = study.nlargest(3, "weight")
    peak = float(study["weight"].max())
    weight_axis.set_ylim(0, peak * 1.3)
    weight_axis.plot(
        (top["position"].to_numpy() + 0.5) * TARGET_Z_MM,
        top["weight"].to_numpy() + peak * 0.12,
        linestyle="none",
        marker="v",
        ms=3.5,
        color=INK,
    )
    weight_axis.text(
        0.02,
        0.95,
        "$\\blacktriangledown$ 3 highest weights\nshaded: slices with lesion",
        transform=weight_axis.transAxes,
        ha="left",
        va="top",
        color=MUTED,
    )

    panels = grid[1].subgridspec(1, 3, wspace=0.08)
    ks = sorted(hits["k"].unique())
    for index, task in enumerate(TASK_TITLES):
        axis = figure.add_subplot(panels[index])
        rows = hits[hits["task"] == task].set_index("k").loc[ks]
        x = np.arange(len(ks))
        axis.errorbar(
            x,
            rows["chance"],
            yerr=[rows["chance"] - rows["chance_low"], rows["chance_high"] - rows["chance"]],
            fmt="o",
            color=MUTED,
            mfc="white",
            ms=3.5,
            lw=0.7,
            capsize=0,
            label="Chance",
        )
        axis.errorbar(
            x,
            rows["hit"],
            yerr=[rows["hit"] - rows["hit_low"], rows["hit_high"] - rows["hit"]],
            fmt="o",
            color=BLUE,
            ms=3.5,
            lw=0.7,
            capsize=0,
            label="Model",
        )
        axis.set_xticks(x)
        axis.set_xticklabels([str(k) for k in ks])
        axis.set_xlim(-0.6, len(ks) - 0.4)
        axis.set_ylim(0, 1)
        axis.set_xlabel("$k$")
        axis.set_title(TASK_TITLES[task].replace(", ", "\n"), fontsize=7)
        tidy(axis)
        if index == 0:
            axis.set_ylabel("hit@$k$")
            axis.legend(frameon=False, loc="lower right", handlelength=1.0, handletextpad=0.4)
        else:
            axis.tick_params(labelleft=False)
    figure.savefig(path, bbox_inches="tight", pad_inches=0.01)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=REPO_ROOT.parent / "cvpt-flatpet-latex/figures")
    parser.add_argument("--examples", action="store_true", help="Also redraw the example figures.")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    configure()
    if args.examples:
        study = load_example()
        problem_figure(study, args.out / "problem.pdf")
        input_figure(study, args.out / "inputs.pdf")
    tables = RESULTS_DIR / "tables"
    methods = pd.read_csv(tables / "methods.csv")
    data_efficiency_figure(methods, args.out / "data_efficiency.pdf")
    hits = pd.read_csv(tables / "attention_hits.csv")
    method = selected_from_table(methods, "diagnosis")
    profile = pd.read_parquet(RESULTS_DIR / "attention" / f"diagnosis__{method}.parquet")
    attention_figure(hits, profile, args.out / "attention.pdf")


if __name__ == "__main__":
    main()
