#!/usr/bin/env python3
"""
export_figure_data_per_fold.py — Export figure data using per-fold trees.

Each fold uses ITS OWN ensemble (not a global one), matching the leakage-free pipeline.
Produces CSVs in outputs/figures/ that TikZ can read.
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
    llm_ensemble_predict, predict_llm_tree,
)

FIG_DIR = OUTPUTS_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def load_per_fold(sp_code):
    path = LLM_TREES_DIR / f"paper_llm_trees_{sp_code}_per_fold.json"
    if not path.exists():
        return None
    d = json.loads(path.read_text())
    return {int(k.split("_")[1]): v for k, v in d.items()}


def rule_signature(node, depth=0):
    """Trace the rule path: each path through a tree becomes a string."""
    if "leaf" in node:
        return [f"→{node['leaf']}"]
    feat = node.get("feature", "?")
    op = node.get("op", "?")
    val = node.get("value", "?")
    cond_l = f"{feat}{op}{val}"
    inv = {"<=": ">", ">": "<=", "<": ">=", ">=": "<"}[op]
    cond_r = f"{feat}{inv}{val}"
    left_paths = rule_signature(node.get("left", {}), depth+1)
    right_paths = rule_signature(node.get("right", {}), depth+1)
    return [f"{cond_l} {p}" if p.startswith("→") else f"{cond_l} & {p}" for p in left_paths] + \
           [f"{cond_r} {p}" if p.startswith("→") else f"{cond_r} & {p}" for p in right_paths]


def export_for_species(sp_code, sp, df):
    out_dir = FIG_DIR / sp_code
    out_dir.mkdir(parents=True, exist_ok=True)

    X, y = build_species_frame(df, sp)
    per_fold_trees = load_per_fold(sp_code)
    if not per_fold_trees:
        print(f"  ⚠ No per-fold trees for {sp_code}")
        return

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    fig3_rows, fig4a_rows, fig4b_counter = [], [], Counter()
    fig2_cards = []

    for fold_id, (tr_idx, te_idx) in enumerate(skf.split(X, y), 1):
        X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]

        # DT(d=2) fit on training fold
        dt = DecisionTreeClassifier(max_depth=DT_MAX_DEPTH,
                                     min_samples_leaf=DT_MIN_SAMPLES_LEAF,
                                     random_state=RANDOM_STATE)
        dt.fit(X_tr, y_tr)
        p_dt = dt.predict_proba(X_te)[:, 1]
        y_dt = (p_dt >= 0.5).astype(int)

        # LLM ensemble for THIS fold
        fold_trees = per_fold_trees.get(fold_id, [])
        if not fold_trees:
            continue
        _, p_llm = llm_ensemble_predict(X_te, fold_trees)

        # Fig 3 — vote vs DT
        for i in range(len(y_te)):
            fig3_rows.append({
                "fold": fold_id, "site": int(X_te.index[i]),
                "dt_pred": int(y_dt[i]), "llm_vote": float(p_llm[i]),
                "observed": int(y_te[i]),
            })

        # Fig 4A — DT splits this fold
        tree_ = dt.tree_
        for nid in range(tree_.node_count):
            if tree_.feature[nid] >= 0:
                fig4a_rows.append({
                    "fold": fold_id,
                    "feature": PREDICTORS[tree_.feature[nid]],
                    "threshold": round(tree_.threshold[nid], 2),
                })

        # Fig 4B — LLM rule paths this fold (each tree contributes its paths)
        for t in fold_trees:
            for path in rule_signature(t["root"]):
                # path looks like "RWQ<=1.07 →1" or "ALT>118 & FFP<=0.5 →0"
                fig4b_counter[path] += 1

        # Fig 2 — explanation cards (fold 1 only)
        if fold_id == 1:
            for i in range(len(y_te)):
                idx = X_te.index[i]
                obs = int(y_te[i])
                dt_p = int(y_dt[i])
                llm_v = float(p_llm[i])
                if obs == 1 and dt_p == 1 and llm_v > 0.6:
                    ctype = "TP"
                elif obs == 0 and dt_p == 1 and llm_v < 0.3:
                    ctype = "FP_LLM_disagrees"
                elif obs == 1 and dt_p == 1 and 0.5 < llm_v < 0.8:
                    ctype = "TP_moderate"
                else:
                    continue
                fig2_cards.append({
                    "site": int(idx), "fold": fold_id, "observed": obs,
                    "dt_pred": dt_p, "llm_vote": round(llm_v, 2), "type": ctype,
                    "RWQ": round(float(X_te.loc[idx, "RWQ"]), 2),
                    "ALT": round(float(X_te.loc[idx, "ALT"]), 1),
                    "FFP": round(float(X_te.loc[idx, "FFP"]), 3),
                    "BIO1": round(float(X_te.loc[idx, "BIO1"]), 2),
                })

    # RF Gini for Fig 5
    rf = RandomForestClassifier(n_estimators=RF_N_ESTIMATORS, max_depth=None,
                                 min_samples_leaf=RF_MIN_SAMPLES_LEAF,
                                 max_features=RF_MAX_FEATURES,
                                 random_state=RANDOM_STATE, n_jobs=-1).fit(X, y)
    fig5_rows = sorted(zip(PREDICTORS, rf.feature_importances_),
                       key=lambda x: -x[1])

    pd.DataFrame(fig3_rows).to_csv(out_dir / "fig3_vote_vs_dt.csv", index=False)

    splits_df = pd.DataFrame(fig4a_rows)
    splits_agg = (splits_df.groupby(["feature", "threshold"]).size()
                  .reset_index(name="fold_count")
                  .sort_values("fold_count", ascending=False))
    splits_agg.to_csv(out_dir / "fig4a_dt_splits.csv", index=False)

    top_rules = sorted(fig4b_counter.items(), key=lambda x: -x[1])[:15]
    pd.DataFrame(top_rules, columns=["rule", "frequency"]).to_csv(
        out_dir / "fig4b_llm_rules.csv", index=False)

    pd.DataFrame(fig2_cards).to_csv(out_dir / "fig2_explanation_cards.csv", index=False)
    pd.DataFrame([{"feature": f, "gini_importance": round(v, 4)}
                  for f, v in fig5_rows]).to_csv(out_dir / "fig5_rf_importance.csv", index=False)

    print(f"  {sp_code}: {len(fig3_rows)} fig3 rows, {len(splits_agg)} fig4a rows, "
          f"{len(top_rules)} fig4b rows, {len(fig2_cards)} fig2 cards")


def main():
    df = clean_decimal_commas(load_excel(str(EXCEL_FILE)))
    for sp_code, sp in SPECIES.items():
        print(f"\n=== {sp_code} ===")
        export_for_species(sp_code, sp, df)


if __name__ == "__main__":
    main()
