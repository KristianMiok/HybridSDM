#!/usr/bin/env python3
"""
export_figure_data.py — Export all data needed for TikZ figures.

Produces CSVs in outputs/figures/ that TikZ/pgfplots can read directly.
"""

import sys, json
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    EXCEL_FILE, OUTPUTS_DIR, LLM_TREES_DIR, SPECIES, PREDICTORS,
    N_FOLDS, RANDOM_STATE, DT_MAX_DEPTH, DT_MIN_SAMPLES_LEAF,
    RF_N_ESTIMATORS, RF_MAX_DEPTH, RF_MIN_SAMPLES_LEAF, RF_MAX_FEATURES,
)
from utils import (
    load_excel, clean_decimal_commas, build_species_frame,
    load_llm_trees, llm_ensemble_predict, predict_llm_tree,
)

FIG_DIR = OUTPUTS_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def export_all():
    df = clean_decimal_commas(load_excel(str(EXCEL_FILE)))

    for sp_code, sp in SPECIES.items():
        print(f"\n{'='*60}")
        print(f"Exporting figure data: {sp['full_name']} ({sp_code})")

        X, y = build_species_frame(df, sp)
        llm_path = LLM_TREES_DIR / f"paper_llm_trees_{sp_code}.json"
        if not llm_path.exists():
            print(f"  ⚠ No LLM trees for {sp_code}, skipping")
            continue
        trees = load_llm_trees(str(llm_path))

        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

        # Collectors
        all_vote_dt = []        # (dt_pred, llm_vote) for Fig 3
        all_dt_splits = []      # (fold, feature, op, threshold) for Fig 4A
        all_llm_activations = Counter()  # rule_string -> count for Fig 4B
        all_explanation_cards = []       # for Fig 2
        all_shap_importance = {}         # for Fig 5

        for fold_id, (tr_idx, te_idx) in enumerate(skf.split(X, y), 1):
            X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
            y_tr, y_te = y[tr_idx], y[te_idx]

            # DT
            dt = DecisionTreeClassifier(
                max_depth=DT_MAX_DEPTH, min_samples_leaf=DT_MIN_SAMPLES_LEAF,
                random_state=RANDOM_STATE
            )
            dt.fit(X_tr, y_tr)
            p_dt = dt.predict_proba(X_te)[:, 1]
            y_dt = (p_dt >= 0.5).astype(int)

            # LLM
            _, p_llm = llm_ensemble_predict(X_te, trees)
            y_llm = (p_llm >= 0.5).astype(int)

            # --- Fig 3: vote vs DT prediction ---
            for i in range(len(y_te)):
                all_vote_dt.append({
                    "fold": fold_id,
                    "site": X_te.index[i],
                    "dt_pred": int(y_dt[i]),
                    "llm_vote": float(p_llm[i]),
                    "observed": int(y_te[i]),
                })

            # --- Fig 4A: DT splits ---
            tree_ = dt.tree_
            feat_names = PREDICTORS
            for node_id in range(tree_.node_count):
                if tree_.feature[node_id] >= 0:  # not a leaf
                    fname = feat_names[tree_.feature[node_id]]
                    thresh = tree_.threshold[node_id]
                    all_dt_splits.append({
                        "fold": fold_id,
                        "feature": fname,
                        "threshold": round(thresh, 2),
                    })

            # --- Fig 4B: LLM rule activations ---
            for t_idx, tree in enumerate(trees, 1):
                preds = predict_llm_tree(X_te, tree)
                root = tree["root"]
                # Build rule string from root
                rule_str = _rule_string(root, t_idx)
                for i in range(len(preds)):
                    # Track which rule path was taken
                    leaf_val = preds[i]
                    full_rule = f"tree#{t_idx}: {_activated_path(root, X_te.iloc[i])}"
                    all_llm_activations[full_rule] += 1

            # --- Fig 2: explanation cards (fold 1 only, select interesting cases) ---
            if fold_id == 1:
                dt_rules_text = export_text(dt, feature_names=PREDICTORS)
                for i in range(len(y_te)):
                    idx = X_te.index[i]
                    obs = int(y_te[i])
                    dt_p = int(y_dt[i])
                    llm_v = float(p_llm[i])
                    if obs == 1 and dt_p == 1 and llm_v > 0.6:  # TP
                        ctype = "TP"
                    elif obs == 0 and dt_p == 1 and llm_v < 0.3:  # FP-DT, LLM disagrees
                        ctype = "FP_LLM_disagrees"
                    elif obs == 1 and dt_p == 0 and llm_v > 0.5:  # FN-DT, LLM corrects
                        ctype = "FN_LLM_corrects"
                    elif obs == 0 and dt_p == 1 and llm_v > 0.8:  # FP, both wrong
                        ctype = "FP_both"
                    else:
                        continue
                    all_explanation_cards.append({
                        "site": idx, "fold": fold_id, "observed": obs,
                        "dt_pred": dt_p, "llm_vote": round(llm_v, 2),
                        "type": ctype,
                        "RWQ": round(float(X_te.loc[idx, "RWQ"]), 2),
                        "ALT": round(float(X_te.loc[idx, "ALT"]), 1),
                        "FFP": round(float(X_te.loc[idx, "FFP"]), 3),
                        "BIO1": round(float(X_te.loc[idx, "BIO1"]), 2),
                    })

        # --- Fig 5: RF SHAP importance ---
        rf = RandomForestClassifier(
            n_estimators=RF_N_ESTIMATORS, max_depth=RF_MAX_DEPTH,
            min_samples_leaf=RF_MIN_SAMPLES_LEAF, max_features=RF_MAX_FEATURES,
            random_state=RANDOM_STATE, n_jobs=-1
        )
        rf.fit(X, y)
        all_shap_importance = dict(zip(PREDICTORS, rf.feature_importances_))

        # ══════════════════════════════════════
        # SAVE CSVs
        # ══════════════════════════════════════
        sp_dir = FIG_DIR / sp_code
        sp_dir.mkdir(parents=True, exist_ok=True)

        # Fig 3 data
        pd.DataFrame(all_vote_dt).to_csv(sp_dir / "fig3_vote_vs_dt.csv", index=False)
        print(f"  Saved fig3_vote_vs_dt.csv ({len(all_vote_dt)} rows)")

        # Fig 4A data
        df_splits = pd.DataFrame(all_dt_splits)
        # Aggregate: count per (feature, threshold)
        split_counts = (df_splits.groupby(["feature", "threshold"])
                        .size().reset_index(name="fold_count")
                        .sort_values("fold_count", ascending=False))
        split_counts.to_csv(sp_dir / "fig4a_dt_splits.csv", index=False)
        print(f"  Saved fig4a_dt_splits.csv ({len(split_counts)} rows)")

        # Fig 4B data
        top_rules = sorted(all_llm_activations.items(), key=lambda x: -x[1])[:15]
        df_rules = pd.DataFrame(top_rules, columns=["rule", "frequency"])
        df_rules.to_csv(sp_dir / "fig4b_llm_rules.csv", index=False)
        print(f"  Saved fig4b_llm_rules.csv ({len(df_rules)} rows)")

        # Fig 2 data
        df_cards = pd.DataFrame(all_explanation_cards)
        df_cards.to_csv(sp_dir / "fig2_explanation_cards.csv", index=False)
        print(f"  Saved fig2_explanation_cards.csv ({len(df_cards)} rows)")

        # Fig 5 data
        df_imp = pd.DataFrame([
            {"feature": f, "gini_importance": round(v, 4)}
            for f, v in sorted(all_shap_importance.items(), key=lambda x: -x[1])
        ])
        df_imp.to_csv(sp_dir / "fig5_rf_importance.csv", index=False)
        print(f"  Saved fig5_rf_importance.csv")


def _rule_string(node, tree_id):
    """Build a compact rule string from a tree node."""
    if "leaf" in node:
        return f"leaf={node['leaf']}"
    feat = node.get("feature", "?")
    op = node.get("op", "?")
    val = node.get("value", "?")
    return f"{feat}{op}{val}"


def _activated_path(node, row):
    """Trace which path a row takes through a tree, return readable string."""
    if "leaf" in node:
        return f"{node['leaf']}"
    feat = node.get("feature", "?")
    op = node.get("op", "<=")
    val = node.get("value", 0)

    ops = {"<=": lambda a, b: a <= b, ">": lambda a, b: a > b,
           "<": lambda a, b: a < b, ">=": lambda a, b: a >= b}

    x = row.get(feat, np.nan)
    if pd.isna(x):
        return f"{feat}=NA"

    if ops.get(op, lambda a, b: False)(x, val):
        child = node.get("left", {})
        child_str = _activated_path(child, row)
        return f"{feat}{op}{val} -> {child_str}"
    else:
        child = node.get("right", {})
        child_str = _activated_path(child, row)
        inv_op = {"<=": ">", ">": "<=", "<": ">=", ">=": "<"}.get(op, op)
        return f"{feat}{inv_op}{val} -> {child_str}"


if __name__ == "__main__":
    export_all()