"""Slide-ready figures for the slice-1 results.

Produces two PNGs in figures/ (committed, so teammates can use them in slides):
  fig1_auroc_by_layer.png  - the replication: probe AUROC across the 8B's layers
  fig2_baselines.png       - the problem: probe vs surface baselines

Numbers are read from the recorded 8B run and from baselines.py, so the figures
cannot drift from the results. No prompt text is used (canary rule).

Usage: .venv/bin/python scripts/make_figures.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path("figures")
OUT.mkdir(exist_ok=True)

# Per-layer AUROC from the Llama-3.1-8B run (Kaggle, 2x T4, bf16).
AUROC_8B = [
    0.5, 0.7094, 0.9703, 0.9917, 0.9944, 0.9864, 0.9822, 0.9819, 0.98,
    0.9847, 0.9847, 0.9833, 0.9831, 0.9772, 0.9811, 0.9836, 0.9794, 0.98,
    0.9819, 0.9842, 0.9844, 0.9839, 0.9864, 0.9889, 0.9903, 0.9881, 0.9878,
    0.9825, 0.9806, 0.9778, 0.9739, 0.9689, 0.9733,
]
BEST_LAYER = 4

# Surface baselines on the same held-out split (from vnaa.baselines).
BASELINES = [
    ("Probe\n(model activations)", 0.9944, "#2a78d6"),
    ("Format only\n(no words)", 0.9675, "#eb6834"),
    ("Bag-of-words\n(word counts)", 0.9517, "#eb6834"),
    ("Length", 0.4506, "#888780"),
]

plt.rcParams.update({
    "font.size": 13,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 200,
})


def fig_auroc_by_layer() -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    layers = np.arange(len(AUROC_8B))
    ax.plot(layers, AUROC_8B, color="#2a78d6", linewidth=2.5, zorder=3)
    ax.scatter([BEST_LAYER], [AUROC_8B[BEST_LAYER]], color="#eb6834", s=90,
               zorder=4, label=f"best: layer {BEST_LAYER}, AUROC {AUROC_8B[BEST_LAYER]:.3f}")
    ax.axhline(0.5, color="#888780", linestyle="--", linewidth=1.5,
               label="chance (0.5)")
    ax.set_xlabel("layer  (0 = embeddings, 32 = final block)")
    ax.set_ylabel("AUROC")
    ax.set_title("Evaluation vs deployment is linearly readable from early layers\n"
                 "Llama-3.1-8B, 200 SAD-eval vs 200 WildChat-deploy prompts",
                 fontsize=13, loc="left")
    ax.set_ylim(0.35, 1.03)
    ax.legend(frameon=False, loc="lower right")
    ax.grid(axis="y", color="#e1e0d9", linewidth=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(OUT / "fig1_auroc_by_layer.png", bbox_inches="tight")
    plt.close(fig)


def fig_baselines() -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    names = [b[0] for b in BASELINES]
    vals = [b[1] for b in BASELINES]
    colors = [b[2] for b in BASELINES]
    bars = ax.bar(names, vals, color=colors, width=0.6, zorder=3)
    ax.axhline(0.5, color="#888780", linestyle="--", linewidth=1.5, zorder=2)
    ax.text(3.42, 0.515, "chance", color="#888780", fontsize=11, ha="right")
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.015, f"{v:.3f}",
                ha="center", fontsize=12, fontweight="medium")
    ax.set_ylabel("AUROC")
    ax.set_ylim(0, 1.12)
    ax.set_title("A classifier that reads no words nearly matches the probe\n"
                 "same held-out prompts (n=120); probe's margin is within noise",
                 fontsize=13, loc="left")
    ax.grid(axis="y", color="#e1e0d9", linewidth=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(OUT / "fig2_baselines.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig_auroc_by_layer()
    fig_baselines()
    print("wrote figures/fig1_auroc_by_layer.png")
    print("wrote figures/fig2_baselines.png")
