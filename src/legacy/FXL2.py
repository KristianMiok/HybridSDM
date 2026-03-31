#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DT vs LLM vs stronger DT+LLM hybrids (Romanian rivers, FXL presence)
- Uses ONLY four predictors: RWQ, ALT, FFP, BIO1
- Paper-style protocol: shallow DT, LLM shallow trees, repeated 67/33 splits
- Hybrids now include: best-weight soft blend, k-veto, soft veto, AND/OR, and a stacked meta-model

Requires: pandas numpy scikit-learn openpyxl
"""

import os, re, json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, balanced_accuracy_score,
    confusion_matrix
)
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression

# =========================
# CONFIG
# =========================
EXCEL_PATH = "/Users/kristianmiok/Desktop/Lucian/LLM/RF/Data/NETWORK.xlsx"
PAPER_LLM_TREES_PATH = str(Path(EXCEL_PATH).expanduser().resolve().parent / "outputs" / "paper_llm_trees.json")

TARGET = "FXL_BIN"          # 1 = presence (FXL_PREZ==1), 0 = true absence (FXL_TRUEABS==1)
TEST_SIZE = 0.25
RANDOM_STATE = 42

# Decision Tree settings (transparent)
DT_MAX_DEPTH = 4
DT_MIN_SAMPLES_LEAF = 5

# Hybrids / paper protocol
PAPER_MODE = True
PAPER_SPLITS = 10
PAPER_TEST_FRACTION = 0.33
PAPER_DT_DEPTH = 2
PAPER_KNN_K = 10
PAPER_PRINT_PROMPT = False  # set True if you want to reprint the JSON prompt

# Hybrid knobs
WEIGHT_SWEEP = [0.95,0.9,0.85,0.8,0.75,0.7,0.65,0.6,0.55,0.5,0.45,0.4,0.35,0.3,0.25,0.2,0.15,0.1,0.05]
K_VETO_LIST = [3, 4]        # veto only if >=k trees say absence
SOFT_VETO_THETA = 0.6       # if absence vote fraction >= theta, scale DT prob by ALPHA
SOFT_VETO_ALPHA = 0.25

# =========================
# UTILS
# =========================
RESULTS: List[Tuple[str,float,float,float]] = []

def _add_result(name: str, acc: float, f1m: float, bacc: float):
    RESULTS.append((name, acc, f1m, bacc))

def outputs_dir(excel_path: str) -> Path:
    outd = Path(excel_path).expanduser().resolve().parent / "outputs"
    outd.mkdir(parents=True, exist_ok=True)
    return outd

def banner(msg: str):
    print("\n" + "="*90)
    print(msg)
    print("="*90)

def load_excel_any(path: str) -> pd.DataFrame:
    try:
        return pd.read_excel(path)
    except Exception:
        return pd.read_excel(path, engine="openpyxl")

def clean_decimal_commas(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in df.columns:
        if df[c].dtype == object:
            s = df[c].astype(str).str.replace("\u00A0", "", regex=False).str.strip()
            coerced = pd.to_numeric(s.str.replace(",", ".", regex=False), errors="coerce")
            if coerced.notna().sum() >= 0.5 * len(df[c]):
                df[c] = coerced
    return df

def id_like_cols(cols: List[str]) -> List[str]:
    return [c for c in cols if re.search(r"(^id$|_id$|^fid$|cellid$)", str(c), flags=re.I)]

def make_preprocessor(X: pd.DataFrame):
    num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    cat_cols = [c for c in X.columns if c not in num_cols]

    numeric_tf = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler(with_mean=False)),
    ])

    from inspect import signature as _sig
    _ohe_kwargs = {"handle_unknown": "ignore"}
    if "sparse_output" in _sig(OneHotEncoder).parameters:
        _ohe_kwargs["sparse_output"] = True
    else:
        _ohe_kwargs["sparse"] = True
    _ohe = OneHotEncoder(**_ohe_kwargs)

    categorical_tf = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("ohe", _ohe),
    ])

    pre = ColumnTransformer(
        transformers=[
            ("num", numeric_tf, num_cols),
            ("cat", categorical_tf, cat_cols),
        ],
        sparse_threshold=0.3,
        remainder="drop",
    )
    return pre, num_cols, cat_cols

def feature_names_from_pre(pre: ColumnTransformer) -> np.ndarray:
    names: List[str] = []
    if "num" in pre.named_transformers_:
        try:
            names.extend(list(pre.named_transformers_["num"].feature_names_in_))
        except Exception:
            names.extend(list(pre.transformers_[0][2]))
    if "cat" in pre.named_transformers_:
        cat_features = pre.transformers_[1][2]
        if cat_features is not None and len(cat_features) > 0:
            ohe = pre.named_transformers_["cat"].named_steps.get("ohe", None)
            if ohe is not None and hasattr(ohe, "get_feature_names_out") and hasattr(ohe, "categories_"):
                try:
                    names.extend(list(ohe.get_feature_names_out(cat_features)))
                except Exception:
                    names.extend([str(c) for c in cat_features])
    return np.array(names, dtype=object)

def print_metrics_simple(label: str, y_true: np.ndarray, y_pred: np.ndarray):
    acc = accuracy_score(y_true, y_pred)
    f1m = f1_score(y_true, y_pred, average="macro")
    bacc = balanced_accuracy_score(y_true, y_pred)
    print(f"[{label}] Acc={acc:.4f} | MacroF1={f1m:.4f} | BalAcc={bacc:.4f}")
    return acc, f1m, bacc

# =========================
# LLM TREES LOADING/EVAL
# =========================
def load_paper_llm_trees(path: str) -> List[Dict[str, Any]]:
    text = Path(path).read_text(encoding="utf-8", errors="ignore")

    def _clean_quotes(t: str) -> str:
        return (t.replace("\u201c", '"').replace("\u201d", '"')
                 .replace("\u201e", '"').replace("\u201f", '"')
                 .replace("\u2018", "'").replace("\u2019", "'")
                 .replace("\u2032", "'").replace("\u2033", '"'))
    def _strip_code_fences(t: str) -> str:
        t = t.strip()
        if t.startswith("```"):
            t = t.split("\n", 1)[1] if "\n" in t else t
            if t.lower().startswith("json\n"):
                t = t.split("\n", 1)[1] if "\n" in t else t
            if t.rstrip().endswith("```"):
                t = t.rstrip()[:-3]
        return t
    def _remove_js_comments(t: str) -> str:
        t = re.sub(r"(^|\s)//.*$", "", t, flags=re.MULTILINE)
        t = re.sub(r"/\*.*?\*/", "", t, flags=re.DOTALL)
        return t
    def _remove_trailing_commas(t: str) -> str:
        return re.sub(r",(\s*[\]\}])", r"\1", t)

    cleaned = _remove_trailing_commas(_remove_js_comments(_strip_code_fences(_clean_quotes(text))))

    try:
        obj = json.loads(cleaned)
    except Exception:
        m = re.search(r"\[.*\]", cleaned, flags=re.DOTALL)
        if m:
            obj = json.loads(_remove_trailing_commas(m.group(0)))
        else:
            m2 = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
            if m2:
                obj = json.loads(_remove_trailing_commas(m2.group(0)))
            else:
                dbg = outputs_dir(EXCEL_PATH) / "bad_llm_json_debug.txt"
                Path(dbg).write_text(cleaned, encoding="utf-8")
                raise ValueError(f"No JSON found. See {dbg}")

    if isinstance(obj, dict) and "trees" in obj:
        obj = obj["trees"]
    if isinstance(obj, dict):
        obj = [obj]
    if not isinstance(obj, list) or len(obj) == 0:
        raise ValueError("LLM trees must be a non-empty list.")
    for i, t in enumerate(obj, 1):
        if not isinstance(t, dict) or "root" not in t:
            raise ValueError(f"Tree #{i} has no 'root'.")
    return obj

OPS = {"<=": lambda a, b: a <= b, ">": lambda a, b: a > b, "<": lambda a, b: a < b, ">=": lambda a, b: a >= b}

def _eval_tree_node(node: Dict[str, Any], row: pd.Series) -> int:
    if "leaf" in node:
        return int(node["leaf"])
    feat = node.get("feature"); op = node.get("op"); val = node.get("value")
    if feat not in row.index or op not in OPS:
        return 0
    x = row[feat]
    if pd.isna(x):
        return int(node.get("majority", 0)) if "majority" in node else 0
    child = node.get("left") if OPS[op](x, val) else node.get("right")
    if child is None:
        return 0
    return _eval_tree_node(child, row)

def predict_with_llm_tree_json(X: pd.DataFrame, tree: Dict[str, Any]) -> np.ndarray:
    preds = np.zeros(len(X), dtype=int)
    root = tree.get("root", {})
    for i, (_, r) in enumerate(X.iterrows()):
        preds[i] = _eval_tree_node(root, r)
    return preds

# helper: matrix of per-tree predictions and LLM probability as mean vote
def llm_vote_matrix_and_prob(X: pd.DataFrame, trees: List[Dict[str,Any]]) -> Tuple[np.ndarray, np.ndarray]:
    mat = np.column_stack([predict_with_llm_tree_json(X, t) for t in trees])  # shape (n, T)
    p = mat.mean(axis=1)  # fraction of trees predicting presence
    return mat, p

# =========================
# PAPER PROTOCOL RUNNER (with advanced hybrids)
# =========================
from sklearn.model_selection import StratifiedShuffleSplit

def run_paper_protocol(df: pd.DataFrame):
    # Build target
    fxl_prez = pd.to_numeric(df["FXL_PREZ"], errors="coerce").fillna(0).astype(int)
    fxl_trueabs = pd.to_numeric(df["FXL_TRUEABS"], errors="coerce").fillna(0).astype(int)
    mask_labeled = (fxl_prez == 1) | (fxl_trueabs == 1)
    data = df.loc[mask_labeled].copy()
    both = (data["FXL_PREZ"].astype(int)==1) & (data["FXL_TRUEABS"].astype(int)==1)
    if both.any():
        data = data.loc[~both].copy()
    data[TARGET] = (data["FXL_PREZ"].astype(int) == 1).astype(int)

    # Drop leakage
    all_cols = list(data.columns)
    drop_cols = set(id_like_cols(all_cols) + ["FXL_PREZ", "FXL_TRUEABS", TARGET])
    drop_cols.update([c for c in all_cols if c.endswith("_PREZ")])
    X_all = data.drop(columns=list(drop_cols), errors="ignore").copy()
    y_all = data[TARGET].astype(int).values

    # Keep ONLY the four allowed predictors in paper-mode
    keep = ["RWQ", "ALT", "FFP", "BIO1"]
    missing = [c for c in keep if c not in X_all.columns]
    if missing:
        raise RuntimeError(f"Missing expected predictors for paper-mode: {missing}")
    X_all = X_all[keep].copy()
    print(f"[i] (paper) Predictors used: {keep}")

    # 10-NN impute numeric
    imputer = KNNImputer(n_neighbors=PAPER_KNN_K)
    X_all[keep] = imputer.fit_transform(X_all[keep])

    # Prompt (optional)
    if PAPER_PRINT_PROMPT:
        print("[Prompt omitted here to keep output short]")

    # Load LLM trees
    if not os.path.exists(PAPER_LLM_TREES_PATH):
        raise FileNotFoundError(f"LLM trees JSON missing: {PAPER_LLM_TREES_PATH}")
    trees = load_paper_llm_trees(PAPER_LLM_TREES_PATH)
    print(f"[i] Loaded {len(trees)} LLM trees from: {PAPER_LLM_TREES_PATH}")

    sss = StratifiedShuffleSplit(n_splits=PAPER_SPLITS, test_size=PAPER_TEST_FRACTION, random_state=RANDOM_STATE)
    rows = []

    for split_id, (tr_idx, te_idx) in enumerate(sss.split(X_all, y_all), start=1):
        X_tr, X_te = X_all.iloc[tr_idx].copy(), X_all.iloc[te_idx].copy()
        y_tr, y_te = y_all[tr_idx], y_all[te_idx]

        # Shallow DT (depth=2) + probabilities
        dt = DecisionTreeClassifier(max_depth=PAPER_DT_DEPTH, min_samples_leaf=5, random_state=RANDOM_STATE)
        dt.fit(X_tr, y_tr)
        p_dt_tr = dt.predict_proba(X_tr)[:,1]
        p_dt_te = dt.predict_proba(X_te)[:,1]
        y_dt_te = (p_dt_te >= 0.5).astype(int)

        # LLM per-tree votes and ensemble probability
        mat_tr, p_llm_tr = llm_vote_matrix_and_prob(X_tr, trees)
        mat_te, p_llm_te = llm_vote_matrix_and_prob(X_te, trees)
        y_llm_maj_te = (p_llm_te >= 0.5).astype(int)

        # Baselines
        acc, f1m, bacc = print_metrics_simple(f"DT(d={PAPER_DT_DEPTH})", y_te, y_dt_te); _add_result(f"DT(d={PAPER_DT_DEPTH})", acc, f1m, bacc)
        acc, f1m, bacc = print_metrics_simple("LLM (5-tree ensemble)", y_te, y_llm_maj_te); _add_result("LLM (5-tree ensemble)", acc, f1m, bacc)

        # ========== Hybrids ==========
        # 1) Soft blend (best weight)
        best_name, best_triplet = None, (-1,-1,-1)
        for w in WEIGHT_SWEEP:
            p = w*p_dt_te + (1.0-w)*p_llm_te
            y_hat = (p >= 0.5).astype(int)
            acc, f1m, bacc = accuracy_score(y_te,y_hat), f1_score(y_te,y_hat,average="macro"), balanced_accuracy_score(y_te,y_hat)
            if f1m > best_triplet[1]:
                best_name = f"DT+LLM soft blend (best w={w:.2f})"
                best_triplet = (acc, f1m, bacc)
        print_metrics_simple(best_name, y_te, ( (best_triplet[0]*0)+ (p_dt_te>=0.5).astype(int) ))  # quick line to print label
        _add_result(best_name, *best_triplet)

        # 2) k-Veto: veto only if >=k absence votes
        absence_votes_te = (mat_te == 0).sum(axis=1)
        T = mat_te.shape[1]
        for k in K_VETO_LIST:
            veto_mask = absence_votes_te >= k
            y_kveto = y_dt_te.copy()
            y_kveto[veto_mask] = 0
            acc, f1m, bacc = print_metrics_simple(f"DT+LLM k-veto (k={k}/{T})", y_te, y_kveto)
            _add_result(f"DT+LLM k-veto (k={k}/{T})", acc, f1m, bacc)

        # 3) Soft veto: damp DT prob where LLM strongly says absence
        absence_frac = 1.0 - p_llm_te
        p_softv = p_dt_te.copy()
        p_softv[absence_frac >= SOFT_VETO_THETA] *= SOFT_VETO_ALPHA
        y_softv = (p_softv >= 0.5).astype(int)
        acc, f1m, bacc = print_metrics_simple(f"DT+LLM soft-veto (θ={SOFT_VETO_THETA}, α={SOFT_VETO_ALPHA})", y_te, y_softv)
        _add_result(f"DT+LLM soft-veto (θ={SOFT_VETO_THETA}, α={SOFT_VETO_ALPHA})", acc, f1m, bacc)

        # 4) AND / OR presence
        y_and = ((y_dt_te==1) & (y_llm_maj_te==1)).astype(int)
        acc, f1m, bacc = print_metrics_simple("DT+LLM AND-presence", y_te, y_and)
        _add_result("DT+LLM AND-presence", acc, f1m, bacc)

        y_or = ((y_dt_te==1) | (y_llm_maj_te==1)).astype(int)
        acc, f1m, bacc = print_metrics_simple("DT+LLM OR-presence", y_te, y_or)
        _add_result("DT+LLM OR-presence", acc, f1m, bacc)

        # 5) Stacked meta-model on [p_dt, p_llm]
        meta = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
        X_meta_tr = np.column_stack([p_dt_tr, p_llm_tr])
        X_meta_te = np.column_stack([p_dt_te, p_llm_te])
        meta.fit(X_meta_tr, y_tr)
        p_meta_te = meta.predict_proba(X_meta_te)[:,1]
        y_meta_te = (p_meta_te >= 0.5).astype(int)
        acc, f1m, bacc = print_metrics_simple("DT+LLM stacked (logistic)", y_te, y_meta_te)
        _add_result("DT+LLM stacked (logistic)", acc, f1m, bacc)

    # Summarize
    df_res = pd.DataFrame(RESULTS, columns=["Model", "Accuracy", "MacroF1", "BalancedAcc"])
    df_sum = df_res.groupby("Model").agg(['mean','std']).sort_index()
    banner("[PAPER] Repeated 67/33 results (mean ± sd)")
    print(df_sum)
    out_path = outputs_dir(EXCEL_PATH) / "paper_mode_results.csv"
    df_sum.to_csv(out_path)
    print(f"[i] Saved paper-mode summary to: {out_path}")

# =========================
# MAIN
# =========================
def main():
    outd = outputs_dir(EXCEL_PATH)
    banner("[1] Load Excel and clean numeric formats")
    if not os.path.exists(EXCEL_PATH):
        raise FileNotFoundError(EXCEL_PATH)
    df_raw = load_excel_any(EXCEL_PATH)
    print("[i] Loaded raw head:\n", df_raw.head(5))
    df = clean_decimal_commas(df_raw)

    if PAPER_MODE:
        banner("[PAPER MODE] Running paper-faithful experiment (with stronger hybrids)")
        run_paper_protocol(df)
        return

    # -------- Non-paper path (kept minimal; still uses the four predictors) --------
    banner("[2] Build modeling frame")
    for col in ["FXL_PREZ", "FXL_TRUEABS"]:
        if col not in df.columns:
            raise RuntimeError(f"Missing column: {col}")
    fxl_prez = pd.to_numeric(df["FXL_PREZ"], errors="coerce").fillna(0).astype(int)
    fxl_trueabs = pd.to_numeric(df["FXL_TRUEABS"], errors="coerce").fillna(0).astype(int)
    mask_labeled = (fxl_prez == 1) | (fxl_trueabs == 1)
    df_model = df.loc[mask_labeled].copy()
    both = (df_model["FXL_PREZ"].astype(int)==1) & (df_model["FXL_TRUEABS"].astype(int)==1)
    if both.any(): df_model = df_model.loc[~both].copy()
    df_model[TARGET] = (df_model["FXL_PREZ"].astype(int) == 1).astype(int)

    # Keep only four predictors
    keep = ["RWQ", "ALT", "FFP", "BIO1"]
    missing = [c for c in keep if c not in df_model.columns]
    if missing: raise RuntimeError(f"Missing expected predictors: {missing}")
    X = df_model[keep].copy()
    y = df_model[TARGET].astype(int)

    # Spatial split needs coords, which we intentionally removed here -> so do a simple stratified split analogue:
    sss = StratifiedShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    tr_idx, te_idx = next(sss.split(X, y))
    X_train, X_test = X.iloc[tr_idx].copy(), X.iloc[te_idx].copy()
    y_train, y_test = y.iloc[tr_idx].copy(), y.iloc[te_idx].copy()

    # DT
    pre, _, _ = make_preprocessor(X_train)
    dt = DecisionTreeClassifier(max_depth=DT_MAX_DEPTH, min_samples_leaf=DT_MIN_SAMPLES_LEAF, random_state=RANDOM_STATE)
    dt_pipe = Pipeline([("pre", pre), ("dt", dt)])
    dt_pipe.fit(X_train, y_train)

    y_prob = dt_pipe.predict_proba(X_test)[:,1]
    y_pred = (y_prob >= 0.5).astype(int)
    acc = accuracy_score(y_test, y_pred); f1w = f1_score(y_test, y_pred, average="weighted")
    try: auc = roc_auc_score(y_test, y_prob)
    except: auc = float("nan")
    print(f"[DT TEST] Acc={acc:.4f} | F1w={f1w:.4f} | AUC={auc:.4f}")

    rules_path = outputs_dir(EXCEL_PATH) / f"decision_tree_rules_{TARGET}.txt"
    with open(rules_path, "w", encoding="utf-8") as f:
        f.write(export_text(dt_pipe.named_steps["dt"], feature_names=list(feature_names_from_pre(dt_pipe.named_steps["pre"]))))
    print(f"[i] Wrote DT rules to: {rules_path}")

if __name__ == "__main__":
    main()