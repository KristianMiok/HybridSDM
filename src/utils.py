#!/usr/bin/env python3
"""
Shared utilities: data loading, LLM tree evaluation, metrics.
"""

import re, json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.metrics import accuracy_score, f1_score

from config import PREDICTORS, KNN_K


# ──────────────────────────────────────────────
# Data loading & cleaning
# ──────────────────────────────────────────────

def load_excel(path: str) -> pd.DataFrame:
    """Load Excel file, trying default then openpyxl engine."""
    try:
        return pd.read_excel(path)
    except Exception:
        return pd.read_excel(path, engine="openpyxl")


def clean_decimal_commas(df: pd.DataFrame) -> pd.DataFrame:
    """Convert European-style decimal commas to floats."""
    df = df.copy()
    for c in df.columns:
        if df[c].dtype == object:
            s = (df[c].astype(str)
                 .str.replace("\u00A0", "", regex=False)
                 .str.strip())
            coerced = pd.to_numeric(
                s.str.replace(",", ".", regex=False), errors="coerce"
            )
            if coerced.notna().sum() >= 0.5 * len(df[c]):
                df[c] = coerced
    return df


def build_species_frame(df: pd.DataFrame, sp_config: dict) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Build labeled modeling frame for a species.
    Returns (X[PREDICTORS], y) after dropping ambiguous records and imputing.
    """
    prez_col = sp_config["prez_col"]
    tabs_col = sp_config["trueabs_col"]
    code = sp_config["code"]
    target = f"{code}_BIN"

    for col in [prez_col, tabs_col]:
        if col not in df.columns:
            raise RuntimeError(f"Missing column: {col}")

    prez = pd.to_numeric(df[prez_col], errors="coerce").fillna(0).astype(int)
    tabs = pd.to_numeric(df[tabs_col], errors="coerce").fillna(0).astype(int)

    mask_labeled = (prez == 1) | (tabs == 1)
    data = df.loc[mask_labeled].copy()

    # Drop ambiguous 1/1
    both = (data[prez_col].astype(int) == 1) & (data[tabs_col].astype(int) == 1)
    if both.any():
        data = data.loc[~both].copy()

    data[target] = (data[prez_col].astype(int) == 1).astype(int)

    # Check predictors exist
    missing = [c for c in PREDICTORS if c not in data.columns]
    if missing:
        raise RuntimeError(f"Missing predictors: {missing}")

    X = data[PREDICTORS].copy()
    y = data[target].astype(int).values

    # KNN imputation
    imputer = KNNImputer(n_neighbors=KNN_K)
    X[PREDICTORS] = imputer.fit_transform(X[PREDICTORS])

    return X, y


# ──────────────────────────────────────────────
# LLM trees: loading & evaluation
# ──────────────────────────────────────────────

def load_llm_trees(path: str) -> List[Dict[str, Any]]:
    """Load and parse LLM-generated decision trees from JSON."""
    text = Path(path).read_text(encoding="utf-8", errors="ignore")

    def _clean_quotes(t):
        return (t.replace("\u201c", '"').replace("\u201d", '"')
                 .replace("\u201e", '"').replace("\u201f", '"')
                 .replace("\u2018", "'").replace("\u2019", "'")
                 .replace("\u2032", "'").replace("\u2033", '"'))

    def _strip_fences(t):
        t = t.strip()
        if t.startswith("```"):
            t = t.split("\n", 1)[1] if "\n" in t else t
            if t.lower().startswith("json\n"):
                t = t.split("\n", 1)[1] if "\n" in t else t
            if t.rstrip().endswith("```"):
                t = t.rstrip()[:-3]
        return t

    def _remove_comments(t):
        t = re.sub(r"(^|\s)//.*$", "", t, flags=re.MULTILINE)
        t = re.sub(r"/\*.*?\*/", "", t, flags=re.DOTALL)
        return t

    def _remove_trailing_commas(t):
        return re.sub(r",(\s*[\]\}])", r"\1", t)

    cleaned = _remove_trailing_commas(
        _remove_comments(_strip_fences(_clean_quotes(text)))
    )

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
                raise ValueError(f"Cannot parse JSON from {path}")

    if isinstance(obj, dict) and "trees" in obj:
        obj = obj["trees"]
    if isinstance(obj, dict):
        obj = [obj]
    if not isinstance(obj, list) or len(obj) == 0:
        raise ValueError("LLM trees must be a non-empty list.")
    for i, t in enumerate(obj, 1):
        if not isinstance(t, dict) or "root" not in t:
            raise ValueError(f"Tree #{i} has no 'root' key.")
    return obj


OPS = {
    "<=": lambda a, b: a <= b,
    ">":  lambda a, b: a > b,
    "<":  lambda a, b: a < b,
    ">=": lambda a, b: a >= b,
}


def _eval_node(node: dict, row: pd.Series) -> int:
    if "leaf" in node:
        return int(node["leaf"])
    feat = node.get("feature")
    op = node.get("op")
    val = node.get("value")
    if feat not in row.index or op not in OPS:
        return 0
    x = row[feat]
    if pd.isna(x):
        return int(node.get("majority", 0)) if "majority" in node else 0
    child = node.get("left") if OPS[op](x, val) else node.get("right")
    return _eval_node(child, row) if child else 0


def predict_llm_tree(X: pd.DataFrame, tree: dict) -> np.ndarray:
    """Predict with a single LLM tree."""
    root = tree.get("root", {})
    return np.array([_eval_node(root, row) for _, row in X.iterrows()], dtype=int)


def llm_ensemble_predict(X: pd.DataFrame, trees: list) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (vote_matrix, vote_prob).
    vote_matrix: shape (n_samples, n_trees), binary
    vote_prob: shape (n_samples,), fraction voting presence
    """
    mat = np.column_stack([predict_llm_tree(X, t) for t in trees])
    prob = mat.mean(axis=1)
    return mat, prob


# ──────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────

def compute_metrics(y_true, y_pred) -> Dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
    }
