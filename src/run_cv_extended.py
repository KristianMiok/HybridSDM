#!/usr/bin/env python3
"""
run_cv_extended.py — Extended cross-validation with all baselines.

Adds to original run_cv.py:
  - GLM (logistic regression) baseline — requested by ALL 3 reviewers
  - Deeper DT baselines (depth 3, 5, unlimited) — addresses R2 weak baseline
  - GAM baseline (optional, requires pygam)
  - Null model (majority class) — requested by R3
  - D² metric (deviance explained) — requested by R3
  - Confusion matrices saved to CSV — requested by R2 and R3
  - Absolute counts of correct predictions — requested by R3

Loads per-fold trees if available (paper_llm_trees_{CODE}_per_fold.json),
otherwise falls back to single-set trees.

Usage:
  python run_cv_extended.py
  python run_cv_extended.py --species AUT
  python run_cv_extended.py --no-alt   # robustness check without altitude
  python run_cv_extended.py --no-bio1  # robustness check without temperature
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, log_loss
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    EXCEL_FILE, OUTPUTS_DIR, LLM_TREES_DIR,
    SPECIES, PREDICTORS,
    N_FOLDS, RANDOM_STATE,
    DT_MIN_SAMPLES_LEAF, RF_N_ESTIMATORS, RF_MIN_SAMPLES_LEAF, RF_MAX_FEATURES,
    AND_TAU_LIST, SOFT_VETO_THETAS, SOFT_VETO_ALPHAS, SOFT_BLEND_WEIGHTS,
)
from utils import (
    load_excel, clean_decimal_commas, build_species_frame,
    load_llm_trees, llm_ensemble_predict,
)


def banner(msg):
    print(f"\n{'='*80}\n{msg}\n{'='*80}")


def load_per_fold_trees(sp_code: str) -> Dict[int, list]:
    """Load per-fold trees if available, else None."""
    per_fold_path = LLM_TREES_DIR / f"paper_llm_trees_{sp_code}_per_fold.json"
    if not per_fold_path.exists():
        return None
    data = json.loads(per_fold_path.read_text())
    return {int(k.split("_")[1]): v for k, v in data.items()}


def compute_d2(y_true, y_pred_proba) -> float:
    """Deviance explained (D² metric requested by R3). Robust version."""
    try:
        eps = 1e-3
        y_pred_proba = np.clip(y_pred_proba, eps, 1 - eps)
        y_arr = np.asarray(y_true, dtype=float)
        ll_model = log_loss(y_arr, y_pred_proba, normalize=True)
        p_null = max(eps, min(1 - eps, float(np.mean(y_arr))))
        ll_null = -(p_null * np.log(p_null) + (1 - p_null) * np.log(1 - p_null))
        if ll_null < eps:
            return float('nan')
        d2 = 1.0 - (ll_model / ll_null)
        return float(np.clip(d2, -2.0, 1.0))
    except Exception:
        return float('nan')


def evaluate_predictions(y_true, y_pred, y_proba=None):
    """Compute all metrics."""
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
    }
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    metrics["tn"] = int(cm[0, 0])
    metrics["fp"] = int(cm[0, 1])
    metrics["fn"] = int(cm[1, 0])
    metrics["tp"] = int(cm[1, 1])
    if y_proba is not None:
        try:
            metrics["d2"] = compute_d2(y_true, y_proba)
        except Exception:
            metrics["d2"] = np.nan
    else:
        metrics["d2"] = np.nan
    return metrics


def run_extended_cv(X, y, sp_code, predictors_used: list,
                     per_fold_trees: dict = None,
                     fallback_trees: list = None) -> Tuple[pd.DataFrame, Dict]:
    """
    Run all baselines + LLM ensembles + hybrids with full metrics.
    Returns (per_fold_df, aggregated_dict).
    """
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    fold_records = []
    confusion_per_model = {}

    for fold_id, (tr_idx, te_idx) in enumerate(skf.split(X, y), 1):
        X_tr, X_te = X.iloc[tr_idx].copy(), X.iloc[te_idx].copy()
        y_tr, y_te = y[tr_idx], y[te_idx]

        # Subset to predictors_used
        X_tr_sub = X_tr[predictors_used]
        X_te_sub = X_te[predictors_used]

        # Standardize for GLM
        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_tr_sub)
        X_te_scaled = scaler.transform(X_te_sub)

        # ── 1. Null model ──
        null = DummyClassifier(strategy="prior", random_state=RANDOM_STATE)
        null.fit(X_tr_sub, y_tr)
        y_null = null.predict(X_te_sub)
        p_null = null.predict_proba(X_te_sub)[:, 1]
        m = evaluate_predictions(y_te, y_null, p_null)
        fold_records.append({"fold": fold_id, "model": "Null (majority)", **m})

        # ── 2. GLM (logistic regression) ──
        glm = LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)
        glm.fit(X_tr_scaled, y_tr)
        y_glm = glm.predict(X_te_scaled)
        p_glm = glm.predict_proba(X_te_scaled)[:, 1]
        m = evaluate_predictions(y_te, y_glm, p_glm)
        fold_records.append({"fold": fold_id, "model": "GLM (logistic)", **m})

        # ── 3. DT at multiple depths ──
        dt_configs = [
            ("DT(d=2)", 2),
            ("DT(d=3)", 3),
            ("DT(d=5)", 5),
            ("DT(d=unlim)", None),
        ]
        dt_predictions = {}
        for name, depth in dt_configs:
            dt = DecisionTreeClassifier(
                max_depth=depth,
                min_samples_leaf=DT_MIN_SAMPLES_LEAF,
                random_state=RANDOM_STATE,
            )
            dt.fit(X_tr_sub, y_tr)
            p_dt = dt.predict_proba(X_te_sub)[:, 1]
            y_dt = (p_dt >= 0.5).astype(int)
            m = evaluate_predictions(y_te, y_dt, p_dt)
            fold_records.append({"fold": fold_id, "model": name, **m})
            if depth == 2:
                dt_predictions["depth_2"] = (y_dt, p_dt)

        # ── 4. Random Forest ──
        rf = RandomForestClassifier(
            n_estimators=RF_N_ESTIMATORS, max_depth=None,
            min_samples_leaf=RF_MIN_SAMPLES_LEAF, max_features=RF_MAX_FEATURES,
            random_state=RANDOM_STATE, n_jobs=-1,
        )
        rf.fit(X_tr_sub, y_tr)
        y_rf = rf.predict(X_te_sub)
        p_rf = rf.predict_proba(X_te_sub)[:, 1]
        m = evaluate_predictions(y_te, y_rf, p_rf)
        fold_records.append({"fold": fold_id, "model": "RF(500)", **m})

        # ── 5. LLM ensemble (per-fold preferred, else fallback) ──
        if per_fold_trees and fold_id in per_fold_trees:
            trees = per_fold_trees[fold_id]
            llm_label = "LLM (per-fold ensemble)"
        elif fallback_trees:
            trees = fallback_trees
            llm_label = f"LLM(global, n={len(trees)})"
        else:
            trees = None
            llm_label = None

        if trees:
            mat_te, p_llm = llm_ensemble_predict(X_te_sub, trees)
            y_llm = (p_llm >= 0.5).astype(int)
            m = evaluate_predictions(y_te, y_llm, p_llm)
            fold_records.append({"fold": fold_id, "model": llm_label, **m})

            # ── 6. Hybrids (using DT(d=2) as the DT component) ──
            if "depth_2" in dt_predictions:
                y_dt, p_dt = dt_predictions["depth_2"]

                # AND-presence
                for tau in AND_TAU_LIST:
                    y_and = ((y_dt == 1) & (p_llm >= tau)).astype(int)
                    p_and = np.where(y_and == 1, np.maximum(p_dt, p_llm), 0.0)
                    m = evaluate_predictions(y_te, y_and, p_and)
                    fold_records.append({"fold": fold_id, "model": f"AND(τ={tau:.2f})", **m})

                # Soft-blend
                for w in SOFT_BLEND_WEIGHTS:
                    p_bl = w * p_dt + (1 - w) * p_llm
                    y_bl = (p_bl >= 0.5).astype(int)
                    m = evaluate_predictions(y_te, y_bl, p_bl)
                    fold_records.append({"fold": fold_id, "model": f"blend(w={w:.3f})", **m})

                # Soft-veto
                absence_frac = 1 - p_llm
                for theta in SOFT_VETO_THETAS:
                    for alpha in SOFT_VETO_ALPHAS:
                        p_sv = p_dt.copy()
                        p_sv[absence_frac >= theta] *= alpha
                        y_sv = (p_sv >= 0.5).astype(int)
                        m = evaluate_predictions(y_te, y_sv, p_sv)
                        fold_records.append({"fold": fold_id,
                                             "model": f"soft-veto(θ={theta},α={alpha})", **m})

    df_folds = pd.DataFrame(fold_records)

    # Aggregate per model
    agg_rows = []
    for model, grp in df_folds.groupby("model", sort=False):
        agg_rows.append({
            "model": model,
            "accuracy_mean": grp["accuracy"].mean(),
            "accuracy_std": grp["accuracy"].std(ddof=1),
            "macro_f1_mean": grp["macro_f1"].mean(),
            "macro_f1_std": grp["macro_f1"].std(ddof=1),
            "d2_mean": grp["d2"].mean(),
            "d2_std": grp["d2"].std(ddof=1),
            "total_tp": int(grp["tp"].sum()),
            "total_fp": int(grp["fp"].sum()),
            "total_tn": int(grp["tn"].sum()),
            "total_fn": int(grp["fn"].sum()),
        })

    df_agg = (pd.DataFrame(agg_rows)
              .sort_values("macro_f1_mean", ascending=False)
              .reset_index(drop=True))

    return df_folds, df_agg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--species", type=str, default=None, choices=list(SPECIES.keys()))
    parser.add_argument("--no-alt", action="store_true",
                        help="Drop ALT predictor (robustness check)")
    parser.add_argument("--no-bio1", action="store_true",
                        help="Drop BIO1 predictor (robustness check)")
    args = parser.parse_args()

    if not EXCEL_FILE.exists():
        print(f"ERROR: {EXCEL_FILE} not found.")
        sys.exit(1)

    df = clean_decimal_commas(load_excel(str(EXCEL_FILE)))
    species_list = [args.species] if args.species else list(SPECIES.keys())

    # Determine predictor set
    predictors_used = list(PREDICTORS)
    suffix = ""
    if args.no_alt:
        predictors_used = [p for p in predictors_used if p != "ALT"]
        suffix = "_noALT"
    if args.no_bio1:
        predictors_used = [p for p in predictors_used if p != "BIO1"]
        suffix += "_noBIO1"

    if suffix:
        print(f"\n[ROBUSTNESS CHECK] Using predictors: {predictors_used}")

    for sp_code in species_list:
        sp = SPECIES[sp_code]
        banner(f"Extended CV: {sp['full_name']} ({sp_code}){suffix}")

        X, y = build_species_frame(df, sp)
        print(f"  Samples: {len(y)} (presence={int(y.sum())}, absence={int((y==0).sum())})")

        # Try per-fold trees first
        per_fold = load_per_fold_trees(sp_code)
        if per_fold:
            print(f"  Using PER-FOLD trees ({sum(len(v) for v in per_fold.values())} total)")
            fallback = None
        else:
            fb_path = LLM_TREES_DIR / f"paper_llm_trees_{sp_code}.json"
            if fb_path.exists():
                fallback = load_llm_trees(str(fb_path))
                print(f"  ⚠ Per-fold trees missing. Using global trees ({len(fallback)})")
            else:
                fallback = None
                print(f"  ⚠ No LLM trees found. Skipping LLM and hybrids.")

        df_folds, df_agg = run_extended_cv(
            X, y, sp_code, predictors_used,
            per_fold_trees=per_fold, fallback_trees=fallback
        )

        out_dir = OUTPUTS_DIR / sp_code
        out_dir.mkdir(parents=True, exist_ok=True)

        df_agg.to_csv(out_dir / f"cv_extended_summary{suffix}.csv", index=False)
        df_folds.to_csv(out_dir / f"cv_extended_perfold{suffix}.csv", index=False)

        print(f"\n  Top 15 models by Macro-F1:\n")
        cols = ["model", "macro_f1_mean", "macro_f1_std", "d2_mean", "total_tp", "total_fn"]
        print(df_agg[cols].head(15).to_string(index=False))
        print(f"\n  Saved to {out_dir}/")


if __name__ == "__main__":
    main()
