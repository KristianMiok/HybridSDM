#!/usr/bin/env python3
"""
explore_data.py — Data exploration and diagnostics.

Produces:
  - Predictor correlation matrix (reviewer asked about ALT-BIO1 correlation)
  - Class balance per species
  - Predictor distributions (presence vs absence)
  - Missing data summary

Usage:
  python explore_data.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import EXCEL_FILE, OUTPUTS_DIR, SPECIES, PREDICTORS, FEATURE_DESCRIPTIONS
from utils import load_excel, clean_decimal_commas, build_species_frame


def main():
    if not EXCEL_FILE.exists():
        print(f"ERROR: {EXCEL_FILE} not found.")
        sys.exit(1)

    df = clean_decimal_commas(load_excel(str(EXCEL_FILE)))
    out_dir = OUTPUTS_DIR / "comparison"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("DATA EXPLORATION")
    print("="*60)

    # ── Per-species summaries ──
    for sp_code, sp in SPECIES.items():
        X, y = build_species_frame(df, sp)
        n1, n0 = int(y.sum()), int(len(y) - y.sum())
        print(f"\n{sp['full_name']} ({sp_code}):")
        print(f"  Total: {len(y)}, Presence: {n1}, Absence: {n0}, Ratio: {n1/len(y):.2%}")

        # Predictor stats by class
        X_with_y = X.copy()
        X_with_y["class"] = y
        print(f"\n  Predictor means by class:")
        print(X_with_y.groupby("class")[PREDICTORS].mean().round(3).to_string())

    # ── Correlation matrix (use AUT as representative, or pool all) ──
    # Pool all labeled data for correlation
    all_X = []
    for sp_code, sp in SPECIES.items():
        X, _ = build_species_frame(df, sp)
        all_X.append(X)

    # Use AUT for the correlation (largest native species dataset)
    X_aut, _ = build_species_frame(df, SPECIES["AUT"])
    corr = X_aut[PREDICTORS].corr()

    print(f"\n{'='*60}")
    print("Predictor correlation matrix (A. torrentium dataset):")
    print(corr.round(3).to_string())

    corr.to_csv(out_dir / "predictor_correlations.csv")

    # Specific ALT-BIO1 correlation (reviewer question)
    r_alt_bio1 = corr.loc["ALT", "BIO1"]
    print(f"\n  ** ALT–BIO1 Pearson r = {r_alt_bio1:.3f} **")
    print(f"  (This value goes into the manuscript placeholder)")

    # ── Correlation heatmap ──
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(PREDICTORS)))
    ax.set_yticks(range(len(PREDICTORS)))
    ax.set_xticklabels(PREDICTORS, rotation=45, ha="right")
    ax.set_yticklabels(PREDICTORS)
    for i in range(len(PREDICTORS)):
        for j in range(len(PREDICTORS)):
            ax.text(j, i, f"{corr.values[i,j]:.2f}", ha="center", va="center",
                    fontsize=10, color="white" if abs(corr.values[i,j]) > 0.5 else "black")
    fig.colorbar(im)
    ax.set_title("Predictor Correlations")
    fig.tight_layout()
    fig.savefig(out_dir / "correlation_heatmap.png", dpi=150)
    plt.close(fig)
    print(f"\n  Saved correlation heatmap to {out_dir}/")

    # ── Boxplots: predictor distributions by presence/absence per species ──
    fig, axes = plt.subplots(len(SPECIES), len(PREDICTORS), figsize=(4*len(PREDICTORS), 3.5*len(SPECIES)))
    for i, (sp_code, sp) in enumerate(SPECIES.items()):
        X, y = build_species_frame(df, sp)
        for j, pred in enumerate(PREDICTORS):
            ax = axes[i, j] if len(SPECIES) > 1 else axes[j]
            for cls, color, label in [(0, "#d62728", "Absence"), (1, "#2ca02c", "Presence")]:
                vals = X.loc[y == cls, pred].dropna()
                bp = ax.boxplot(vals, positions=[cls], widths=0.6,
                                patch_artist=True, showfliers=False)
                bp["boxes"][0].set_facecolor(color)
                bp["boxes"][0].set_alpha(0.5)
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["Abs", "Pres"])
            if i == 0:
                ax.set_title(pred)
            if j == 0:
                ax.set_ylabel(sp_code)

    fig.suptitle("Predictor distributions by class", y=1.01)
    fig.tight_layout()
    fig.savefig(out_dir / "predictor_boxplots.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved predictor boxplots")


if __name__ == "__main__":
    main()
