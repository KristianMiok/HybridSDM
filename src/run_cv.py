#!/usr/bin/env python3
"""
run_cv.py — 5-fold cross-validation for all three species.

Models evaluated per species:
  1. Shallow DT (depth=2)
  2. LLM ensemble (majority vote)
  3. Hybrid variants: AND-presence, OR-presence, k-veto, soft-veto, soft-blend, stacked
  4. Random Forest (black-box benchmark)
  5. RF + SHAP explanations

Usage:
  python run_cv.py                 # run all species
  python run_cv.py --species FXL   # run one species
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

# Allow running from src/
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    EXCEL_FILE, OUTPUTS_DIR, LLM_TREES_DIR,
    SPECIES, PREDICTORS,
    N_FOLDS, RANDOM_STATE,
    DT_MAX_DEPTH, DT_MIN_SAMPLES_LEAF,
    RF_N_ESTIMATORS, RF_MAX_DEPTH, RF_MIN_SAMPLES_LEAF, RF_MAX_FEATURES,
    AND_TAU_LIST, SOFT_VETO_THETAS, SOFT_VETO_ALPHAS, SOFT_BLEND_WEIGHTS,
)
from utils import (
    load_excel, clean_decimal_commas, build_species_frame,
    load_llm_trees, llm_ensemble_predict, compute_metrics,
)


def banner(msg: str):
    print(f"\n{'='*80}\n{msg}\n{'='*80}")


def run_species_cv(
    X: pd.DataFrame,
    y: np.ndarray,
    sp_code: str,
    trees: list = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Run full 5-fold CV for one species.
    Returns (summary_df, per_fold_df, shap_data).
    """
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    fold_rows = []   # (fold, model, accuracy, macro_f1)
    dt_rules = {}    # fold -> export_text
    shap_data = {"fold_importances": [], "fold_oob_scores": []}

    has_llm = trees is not None and len(trees) > 0
    T = len(trees) if has_llm else 0

    for fold_id, (tr_idx, te_idx) in enumerate(skf.split(X, y), start=1):
        X_tr, X_te = X.iloc[tr_idx].copy(), X.iloc[te_idx].copy()
        y_tr, y_te = y[tr_idx], y[te_idx]

        # ── 1. Shallow Decision Tree ──
        dt = DecisionTreeClassifier(
            max_depth=DT_MAX_DEPTH,
            min_samples_leaf=DT_MIN_SAMPLES_LEAF,
            random_state=RANDOM_STATE,
        )
        dt.fit(X_tr, y_tr)
        p_dt_tr = dt.predict_proba(X_tr)[:, 1]
        p_dt_te = dt.predict_proba(X_te)[:, 1]
        y_dt_te = (p_dt_te >= 0.5).astype(int)

        m = compute_metrics(y_te, y_dt_te)
        fold_rows.append([fold_id, f"DT(d={DT_MAX_DEPTH})", m["accuracy"], m["macro_f1"]])

        # Save DT rules for stability analysis
        dt_rules[fold_id] = export_text(dt, feature_names=PREDICTORS)

        # ── 2. Random Forest (black-box benchmark) ──
        rf = RandomForestClassifier(
            n_estimators=RF_N_ESTIMATORS,
            max_depth=RF_MAX_DEPTH,
            min_samples_leaf=RF_MIN_SAMPLES_LEAF,
            max_features=RF_MAX_FEATURES,
            random_state=RANDOM_STATE,
            oob_score=True,
            n_jobs=-1,
        )
        rf.fit(X_tr, y_tr)
        y_rf_te = rf.predict(X_te)
        m_rf = compute_metrics(y_te, y_rf_te)
        fold_rows.append([fold_id, "RF(500)", m_rf["accuracy"], m_rf["macro_f1"]])

        # RF feature importances (for comparison with DT/LLM rules)
        shap_data["fold_importances"].append(
            dict(zip(PREDICTORS, rf.feature_importances_))
        )
        if hasattr(rf, "oob_score_"):
            shap_data["fold_oob_scores"].append(rf.oob_score_)

        # ── 3. LLM ensemble ──
        if has_llm:
            mat_tr, p_llm_tr = llm_ensemble_predict(X_tr, trees)
            mat_te, p_llm_te = llm_ensemble_predict(X_te, trees)
            y_llm_te = (p_llm_te >= 0.5).astype(int)
            m_llm = compute_metrics(y_te, y_llm_te)
            fold_rows.append([fold_id, f"LLM({T}-tree)", m_llm["accuracy"], m_llm["macro_f1"]])

            # ── 4. Hybrids ──

            # AND-presence
            for tau in AND_TAU_LIST:
                y_and = ((y_dt_te == 1) & (p_llm_te >= tau)).astype(int)
                m_and = compute_metrics(y_te, y_and)
                fold_rows.append([fold_id, f"AND(τ={tau:.2f})", m_and["accuracy"], m_and["macro_f1"]])

            # OR-presence
            y_or = ((y_dt_te == 1) | (y_llm_te == 1)).astype(int)
            m_or = compute_metrics(y_te, y_or)
            fold_rows.append([fold_id, "OR-presence", m_or["accuracy"], m_or["macro_f1"]])

            # k-veto
            absence_votes = (mat_te == 0).sum(axis=1)
            for k in [3, 4]:
                if k <= T:
                    y_kv = y_dt_te.copy()
                    y_kv[absence_votes >= k] = 0
                    m_kv = compute_metrics(y_te, y_kv)
                    fold_rows.append([fold_id, f"k-veto(k={k}/{T})", m_kv["accuracy"], m_kv["macro_f1"]])

            # Soft-veto
            absence_frac = 1.0 - p_llm_te
            for theta in SOFT_VETO_THETAS:
                for alpha in SOFT_VETO_ALPHAS:
                    p_sv = p_dt_te.copy()
                    p_sv[absence_frac >= theta] *= alpha
                    y_sv = (p_sv >= 0.5).astype(int)
                    m_sv = compute_metrics(y_te, y_sv)
                    fold_rows.append([fold_id, f"soft-veto(θ={theta},α={alpha})", m_sv["accuracy"], m_sv["macro_f1"]])

            # Soft-blend
            for w in SOFT_BLEND_WEIGHTS:
                p_bl = w * p_dt_te + (1.0 - w) * p_llm_te
                y_bl = (p_bl >= 0.5).astype(int)
                m_bl = compute_metrics(y_te, y_bl)
                fold_rows.append([fold_id, f"blend(w={w:.3f})", m_bl["accuracy"], m_bl["macro_f1"]])

            # Stacked logistic
            meta = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
            X_meta_tr = np.column_stack([p_dt_tr, p_llm_tr])
            X_meta_te = np.column_stack([p_dt_te, p_llm_te])
            meta.fit(X_meta_tr, y_tr)
            y_meta = (meta.predict_proba(X_meta_te)[:, 1] >= 0.5).astype(int)
            m_meta = compute_metrics(y_te, y_meta)
            fold_rows.append([fold_id, "stacked(logistic)", m_meta["accuracy"], m_meta["macro_f1"]])

    # ── Aggregate ──
    df_folds = pd.DataFrame(fold_rows, columns=["fold", "model", "accuracy", "macro_f1"])

    summary_rows = []
    for model_name, grp in df_folds.groupby("model", sort=False):
        summary_rows.append({
            "model": model_name,
            "accuracy_mean": grp["accuracy"].mean(),
            "accuracy_std": grp["accuracy"].std(ddof=1),
            "macro_f1_mean": grp["macro_f1"].mean(),
            "macro_f1_std": grp["macro_f1"].std(ddof=1),
        })

    df_summary = (
        pd.DataFrame(summary_rows)
        .sort_values("macro_f1_mean", ascending=False)
        .reset_index(drop=True)
    )

    # Add DT rules to shap_data for stability analysis
    shap_data["dt_rules"] = dt_rules

    return df_summary, df_folds, shap_data


def run_one_species(df: pd.DataFrame, sp_code: str, pure: bool = False):
    """Load data + optional LLM trees, run CV, save results."""
    sp = SPECIES[sp_code]
    mode = " [PURE]" if pure else ""
    banner(f"Species: {sp['full_name']} ({sp_code}) — {sp['context']}{mode}")

    X, y = build_species_frame(df, sp)
    n1 = int(y.sum())
    n0 = int(len(y) - n1)
    print(f"  Samples: {len(y)} (presence={n1}, absence={n0})")

    # Look for LLM trees JSON
    suffix = "_pure" if pure else ""
    llm_path = LLM_TREES_DIR / f"paper_llm_trees_{sp_code}{suffix}.json"
    trees = None
    if llm_path.exists():
        trees = load_llm_trees(str(llm_path))
        print(f"  Loaded {len(trees)} LLM trees from {llm_path.name}")
    else:
        print(f"  ⚠ No LLM trees found at {llm_path}. Running DT + RF only.")

    df_summary, df_folds, shap_data = run_species_cv(X, y, sp_code, trees)

    # Save
    out_dir = OUTPUTS_DIR / sp_code
    out_dir.mkdir(parents=True, exist_ok=True)

    df_summary.to_csv(out_dir / "cv5_summary.csv", index=False)
    df_folds.to_csv(out_dir / "cv5_perfold.csv", index=False)

    # Save RF importances
    if shap_data["fold_importances"]:
        imp_df = pd.DataFrame(shap_data["fold_importances"])
        imp_df.insert(0, "fold", range(1, len(imp_df) + 1))
        imp_df.to_csv(out_dir / "rf_feature_importances.csv", index=False)

    # Save DT rules per fold
    if shap_data.get("dt_rules"):
        rules_path = out_dir / "dt_rules_per_fold.txt"
        with open(rules_path, "w") as f:
            for fold_id, rules in shap_data["dt_rules"].items():
                f.write(f"=== Fold {fold_id} ===\n{rules}\n\n")

    print(f"\n  Top 10 models by Macro-F1:\n")
    print(df_summary.head(10).to_string(index=False))
    print(f"\n  Results saved to {out_dir}/")

    return df_summary


def main():
    parser = argparse.ArgumentParser(description="Run 5-fold CV experiments")
    parser.add_argument("--species", type=str, default=None,
                        choices=list(SPECIES.keys()),
                        help="Run for a single species (default: all)")
    parser.add_argument("--pure", action="store_true",
                        help="Use pure-mode LLM trees (paper_llm_trees_{CODE}_pure.json)")
    args = parser.parse_args()

    if not EXCEL_FILE.exists():
        print(f"ERROR: Data file not found: {EXCEL_FILE}")
        print(f"  Please place NETWORK.xlsx in {EXCEL_FILE.parent}/")
        sys.exit(1)

    banner("Loading data")
    df_raw = load_excel(str(EXCEL_FILE))
    df = clean_decimal_commas(df_raw)
    print(f"  {len(df)} rows, {len(df.columns)} columns")

    species_list = [args.species] if args.species else list(SPECIES.keys())
    all_summaries = {}

    for sp_code in species_list:
        summary = run_one_species(df, sp_code, pure=args.pure)
        all_summaries[sp_code] = summary

    # If all species were run, create comparison table
    if len(all_summaries) == len(SPECIES):
        banner("Cross-species comparison")
        comp_rows = []
        for sp_code, summary in all_summaries.items():
            sp = SPECIES[sp_code]
            # Best DT
            dt_row = summary[summary["model"].str.startswith("DT")].iloc[0]
            # Best RF
            rf_row = summary[summary["model"].str.startswith("RF")].iloc[0]
            # Best hybrid (if available)
            non_base = summary[~summary["model"].str.match(r"^(DT|RF|LLM)")]
            best_hybrid = non_base.iloc[0] if len(non_base) > 0 else None
            # LLM standalone
            llm_rows = summary[summary["model"].str.startswith("LLM")]
            llm_row = llm_rows.iloc[0] if len(llm_rows) > 0 else None

            comp_rows.append({
                "species": sp["full_name"],
                "code": sp_code,
                "DT_f1": f"{dt_row['macro_f1_mean']:.3f}±{dt_row['macro_f1_std']:.3f}",
                "LLM_f1": f"{llm_row['macro_f1_mean']:.3f}±{llm_row['macro_f1_std']:.3f}" if llm_row is not None else "N/A",
                "best_hybrid": best_hybrid["model"] if best_hybrid is not None else "N/A",
                "hybrid_f1": f"{best_hybrid['macro_f1_mean']:.3f}±{best_hybrid['macro_f1_std']:.3f}" if best_hybrid is not None else "N/A",
                "RF_f1": f"{rf_row['macro_f1_mean']:.3f}±{rf_row['macro_f1_std']:.3f}",
            })

        comp_df = pd.DataFrame(comp_rows)
        comp_path = OUTPUTS_DIR / "comparison" / "species_comparison.csv"
        comp_path.parent.mkdir(parents=True, exist_ok=True)
        comp_df.to_csv(comp_path, index=False)
        print(comp_df.to_string(index=False))
        print(f"\n  Comparison saved to {comp_path}")


if __name__ == "__main__":
    main()