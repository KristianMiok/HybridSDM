#!/usr/bin/env python3
"""
run_shap.py — SHAP analysis for Random Forest models.

Produces per-species:
  - SHAP summary plots (beeswarm)
  - SHAP feature importance bar plots
  - Partial dependence plots for each predictor
  - Comparison data: RF response functions vs DT thresholds vs LLM rule frequencies

Requires: pip install shap matplotlib

Usage:
  python run_shap.py                 # all species
  python run_shap.py --species AUT   # one species
"""

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import PartialDependenceDisplay

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    EXCEL_FILE, OUTPUTS_DIR, SPECIES, PREDICTORS,
    N_FOLDS, RANDOM_STATE, RF_N_ESTIMATORS, RF_MAX_DEPTH,
    RF_MIN_SAMPLES_LEAF, RF_MAX_FEATURES, FEATURE_DESCRIPTIONS,
)
from utils import load_excel, clean_decimal_commas, build_species_frame

warnings.filterwarnings("ignore", category=FutureWarning)


def run_shap_analysis(X, y, sp_code, out_dir):
    """Train RF on full data (or fold-aggregated), compute SHAP values."""
    try:
        import shap
    except ImportError:
        print("  ⚠ shap not installed. Run: pip install shap")
        print("  Falling back to RF feature importances + partial dependence only.")
        shap = None

    out_dir.mkdir(parents=True, exist_ok=True)

    # Train RF on full data for SHAP (CV was done in run_cv.py for metrics)
    rf = RandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS,
        max_depth=RF_MAX_DEPTH,
        min_samples_leaf=RF_MIN_SAMPLES_LEAF,
        max_features=RF_MAX_FEATURES,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    rf.fit(X, y)

    # ── Feature importances ──
    imp = pd.DataFrame({
        "feature": PREDICTORS,
        "importance": rf.feature_importances_,
    }).sort_values("importance", ascending=False)
    imp.to_csv(out_dir / "rf_importance_full.csv", index=False)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(imp["feature"], imp["importance"])
    ax.set_xlabel("Gini Importance")
    ax.set_title(f"RF Feature Importance — {sp_code}")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(out_dir / "rf_importance_bar.png", dpi=150)
    plt.close(fig)
    print(f"  Saved RF importance plot")

    # ── Partial Dependence Plots ──
    fig, axes = plt.subplots(1, len(PREDICTORS), figsize=(4 * len(PREDICTORS), 4))
    PartialDependenceDisplay.from_estimator(
        rf, X, features=PREDICTORS, ax=axes,
        kind="both",  # individual + average
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    fig.suptitle(f"Partial Dependence — {sp_code} (RF)", y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / "rf_partial_dependence.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved partial dependence plots")

    # ── SHAP analysis ──
    if shap is not None:
        explainer = shap.TreeExplainer(rf)
        shap_values = explainer.shap_values(X)

        # For binary classification, shap_values is a list [class0, class1]
        if isinstance(shap_values, list):
            sv = shap_values[1]  # SHAP for presence class
        else:
            sv = shap_values

        # Beeswarm plot
        fig, ax = plt.subplots(figsize=(8, 5))
        shap.summary_plot(sv, X, feature_names=PREDICTORS, show=False)
        plt.title(f"SHAP Summary — {sp_code} (RF)")
        plt.tight_layout()
        plt.savefig(out_dir / "shap_summary_beeswarm.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved SHAP beeswarm plot")

        # SHAP bar plot
        fig, ax = plt.subplots(figsize=(6, 4))
        shap.summary_plot(sv, X, feature_names=PREDICTORS, plot_type="bar", show=False)
        plt.title(f"SHAP Importance — {sp_code} (RF)")
        plt.tight_layout()
        plt.savefig(out_dir / "shap_summary_bar.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved SHAP bar plot")

        # Save raw SHAP values
        sv_df = pd.DataFrame(sv, columns=PREDICTORS)
        sv_df.to_csv(out_dir / "shap_values.csv", index=False)

        # Mean absolute SHAP
        mean_shap = pd.DataFrame({
            "feature": PREDICTORS,
            "mean_abs_shap": np.abs(sv).mean(axis=0),
        }).sort_values("mean_abs_shap", ascending=False)
        mean_shap.to_csv(out_dir / "shap_mean_abs.csv", index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--species", type=str, default=None, choices=list(SPECIES.keys()))
    args = parser.parse_args()

    if not EXCEL_FILE.exists():
        print(f"ERROR: {EXCEL_FILE} not found.")
        sys.exit(1)

    df = clean_decimal_commas(load_excel(str(EXCEL_FILE)))
    species_list = [args.species] if args.species else list(SPECIES.keys())

    for sp_code in species_list:
        sp = SPECIES[sp_code]
        print(f"\n{'='*60}")
        print(f"SHAP analysis: {sp['full_name']} ({sp_code})")

        X, y = build_species_frame(df, sp)
        out_dir = OUTPUTS_DIR / sp_code / "shap"
        run_shap_analysis(X, y, sp_code, out_dir)

    print("\nDone. Check outputs/{CODE}/shap/ for plots and data.")


if __name__ == "__main__":
    main()
