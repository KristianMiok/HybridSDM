#!/usr/bin/env python3
"""
generate_llm_trees_per_fold.py — Per-fold LLM tree generation.

CRITICAL: This addresses Reviewer 2's circularity / data leakage concern.
Instead of generating one set of trees from the full dataset, we regenerate
trees within each CV fold using ONLY the training data of that fold.

For 5 folds × 3 species = 15 generation runs.

Budget control:
  - --max-cost: stop if estimated cost exceeds this USD amount
  - --batch-size: trees per API call (default: 25)
  - --n-candidates: candidates to generate per fold (default: 80, was 150 in v2)

Usage:
  python generate_llm_trees_per_fold.py --species AUT
  python generate_llm_trees_per_fold.py --max-cost 20.00
  python generate_llm_trees_per_fold.py --dry-run  # estimate cost only
"""

import argparse
import json
import re
import sys
import time
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    EXCEL_FILE, LLM_TREES_DIR, OUTPUTS_DIR,
    SPECIES, PREDICTORS, FEATURE_DESCRIPTIONS,
    LLM_TREE_MAX_DEPTH, QUANTILES,
    OPENAI_API_KEY, OPENAI_MODEL,
    N_FOLDS, RANDOM_STATE, DT_MAX_DEPTH, DT_MIN_SAMPLES_LEAF,
)
from utils import (
    load_excel, clean_decimal_commas, build_species_frame,
    predict_llm_tree, llm_ensemble_predict,
)

# ══════════════════════════════════════════════
# CONFIGURATION (per-fold, smaller candidate pool)
# ══════════════════════════════════════════════
N_CANDIDATES_PER_FOLD = 80     # was 150 in v2; reduced to control cost
BATCH_SIZE = 20                # trees per API call
TEMPERATURES = [0.4, 0.7, 0.9]
MIN_TREE_F1 = 0.40
MAX_ENSEMBLE_SIZE = 15         # max trees per ensemble
MAX_NO_IMPROVE = 5             # stop greedy after N non-improving additions

# Cost estimates for GPT-4o (USD per 1M tokens, as of 2026)
COST_INPUT_PER_M = 5.00
COST_OUTPUT_PER_M = 15.00
EST_INPUT_TOKENS_PER_BATCH = 2500
EST_OUTPUT_TOKENS_PER_BATCH = 4000


# ══════════════════════════════════════════════
# Ecological priors (same as v2)
# ══════════════════════════════════════════════
SPECIES_PRIORS = {
    "AUT": [
        "Austropotamobius torrentium is a cold-water species found in clean, fast-flowing montane streams.",
        "ALT: prefers moderate altitudes (200–800 m); declines at very low or very high elevations.",
        "RWQ: strongly associated with high water quality (LOWER RWQ values favor presence).",
        "BIO1: favors cooler temperatures; occurrence typically declines above ~10 °C mean annual temperature.",
        "FFP: tolerates moderate flash-flood potential but avoids extreme hydrological instability.",
    ],
    "ABI": [
        "Austropotamobius bihariensis is a micro-endemic confined to the Apuseni Mountains, Romania.",
        "ALT: restricted to higher-altitude streams (typically >400 m); very limited low-elevation records.",
        "RWQ: requires high water quality (low RWQ values); sensitive to pollution.",
        "BIO1: adapted to cool montane climates; likely absent from warmer lowland areas.",
        "FFP: ecological requirements poorly known; use priors cautiously due to limited data.",
    ],
    "FXL": [
        "Faxonius limosus is an invasive North American crayfish with generalist ecology.",
        "ALT: primarily found at lower elevations in larger, slower rivers; uncommon in montane streams.",
        "RWQ: tolerant of degraded water quality (higher RWQ); may benefit from moderate eutrophication.",
        "BIO1: tolerates a wide temperature range; may favor warmer conditions.",
        "FFP: low flash-flood potential (calm, larger rivers) strongly favors presence.",
    ],
}


# ══════════════════════════════════════════════
# COST TRACKING
# ══════════════════════════════════════════════
class BudgetTracker:
    def __init__(self, max_cost_usd: float):
        self.max_cost = max_cost_usd
        self.spent = 0.0
        self.api_calls = 0

    def estimate_call_cost(self) -> float:
        return (EST_INPUT_TOKENS_PER_BATCH * COST_INPUT_PER_M / 1e6 +
                EST_OUTPUT_TOKENS_PER_BATCH * COST_OUTPUT_PER_M / 1e6)

    def can_afford(self) -> bool:
        return self.spent + self.estimate_call_cost() <= self.max_cost

    def record_call(self, actual_cost: float = None):
        if actual_cost is None:
            actual_cost = self.estimate_call_cost()
        self.spent += actual_cost
        self.api_calls += 1

    def summary(self) -> str:
        return f"Calls: {self.api_calls}, Spent: ${self.spent:.2f} / ${self.max_cost:.2f}"


# ══════════════════════════════════════════════
# Prompt building (uses TRAINING DATA ONLY)
# ══════════════════════════════════════════════
def compute_stats_from_training(X_train: pd.DataFrame) -> Dict[str, dict]:
    """Compute stats using ONLY training fold data."""
    stats = {}
    for c in PREDICTORS:
        s = pd.to_numeric(X_train[c], errors="coerce").dropna()
        if s.empty:
            continue
        stats[c] = {
            "min": float(s.min()),
            "max": float(s.max()),
            "median": float(s.quantile(0.5)),
            "quantiles": {f"q{int(q*100):02d}": float(s.quantile(q)) for q in QUANTILES},
        }
    return stats


def compute_class_conditional_from_training(X_train, y_train) -> str:
    """Class-conditional stats from TRAINING DATA ONLY."""
    lines = []
    for c in PREDICTORS:
        vals_pres = X_train.loc[y_train == 1, c].dropna()
        vals_abs = X_train.loc[y_train == 0, c].dropna()
        if vals_pres.empty or vals_abs.empty:
            continue
        lines.append(
            f"- {c}: Presence mean={vals_pres.mean():.2f} (q25={vals_pres.quantile(0.25):.2f}, "
            f"q75={vals_pres.quantile(0.75):.2f}) | "
            f"Absence mean={vals_abs.mean():.2f} (q25={vals_abs.quantile(0.25):.2f}, "
            f"q75={vals_abs.quantile(0.75):.2f})"
        )
    return "\n".join(lines)


def get_dt_splits_from_training(X_train, y_train) -> str:
    """DT splits from TRAINING DATA ONLY."""
    dt = DecisionTreeClassifier(
        max_depth=DT_MAX_DEPTH,
        min_samples_leaf=DT_MIN_SAMPLES_LEAF,
        random_state=RANDOM_STATE,
    )
    dt.fit(X_train, y_train)
    return export_text(dt, feature_names=PREDICTORS)


def format_stats(stats: dict) -> str:
    lines = []
    for c in PREDICTORS:
        st = stats.get(c)
        if not st:
            continue
        desc = FEATURE_DESCRIPTIONS.get(c, c)
        q_parts = [f"{k}={v:.3g}" for k, v in st["quantiles"].items()]
        lines.append(
            f"- {c} ({desc}): min={st['min']:.3g}, "
            + ", ".join(q_parts)
            + f", median={st['median']:.3g}, max={st['max']:.3g}"
        )
    return "\n".join(lines)


def build_prompt_for_fold(sp_code: str, sp_name: str, X_train: pd.DataFrame, y_train: np.ndarray,
                           n_trees: int, fold_id: int) -> str:
    """Build prompt using ONLY training fold data."""
    stats = compute_stats_from_training(X_train)
    stats_block = format_stats(stats)
    class_stats = compute_class_conditional_from_training(X_train, y_train)
    dt_rules = get_dt_splits_from_training(X_train, y_train)
    priors = SPECIES_PRIORS.get(sp_code, [])
    priors_block = "\n".join(f"- {p}" for p in priors)
    n1, n0 = int(y_train.sum()), int((y_train == 0).sum())
    class_ratio = f"presence={n1} | absence={n0} (training fold {fold_id})"

    return f"""You are an aquatic ecologist building a species distribution model.
Produce EXACTLY {n_trees} shallow decision trees (depth ≤ {LLM_TREE_MAX_DEPTH}) to predict
{sp_name} ({sp_code}_BIN) presence (1) vs true absence (0) in Romanian rivers.

NOTE: This is fold {fold_id} of a 5-fold cross-validation. The statistics below come ONLY
from the training data of this fold (test data is held out and unseen).

OPTIMIZE FOR **Macro-F1**. Class balance: {class_ratio}.

ALLOWED FEATURES (ONLY): {', '.join(PREDICTORS)}

ECOLOGICAL PRIORS:
{priors_block}

CLASS-CONDITIONAL STATISTICS (training fold {fold_id} only):
{class_stats}

OVERALL PREDICTOR RANGES (training fold {fold_id} only):
{stats_block}

DATA-DRIVEN DECISION TREE (DT trained on this training fold only):
{dt_rules}

DIVERSITY & QUALITY CONSTRAINTS:
- Cycle root features across trees (use each of [{', '.join(PREDICTORS)}] roughly equally).
- Per tree, test at most 2 different features.
- Place thresholds where presence and absence distributions SEPARATE.
- Each tree should capture a DIFFERENT ecological hypothesis.

OUTPUT: STRICT JSON array of EXACTLY {n_trees} trees, NO commentary:
[
  {{
    "tree_id": 1, "target": "{sp_code}_BIN", "max_depth": {LLM_TREE_MAX_DEPTH},
    "root": {{
      "feature": "<FEAT>", "op": "<=", "value": <number>, "quantile_hint": "<qNN>",
      "left":  {{ "leaf": <0_or_1> }},
      "right": {{
        "feature": "<FEAT>", "op": ">", "value": <number>, "quantile_hint": "<qNN>",
        "left":  {{ "leaf": <0_or_1> }},
        "right": {{ "leaf": <0_or_1> }}
      }}
    }}
  }}
]"""


# ══════════════════════════════════════════════
# API & validation (same as v2)
# ══════════════════════════════════════════════
import os as _os
SAFETY_IDENTIFIER = _os.environ.get("OPENAI_SAFETY_IDENTIFIER", "")

def call_openai(prompt: str, temperature: float = 0.7) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)
    kwargs = {
        "model": OPENAI_MODEL,
        "messages": [
            {"role": "system", "content": "You are an expert aquatic ecologist. Output ONLY valid JSON."},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "max_tokens": 16000,
    }
    if SAFETY_IDENTIFIER:
        kwargs["safety_identifier"] = SAFETY_IDENTIFIER
    response = client.chat.completions.create(**kwargs)
    return response.choices[0].message.content


def extract_json(text: str) -> list:
    for old, new in [("\u201c", '"'), ("\u201d", '"'), ("\u201e", '"'),
                     ("\u201f", '"'), ("\u2018", "'"), ("\u2019", "'")]:
        text = text.replace(old, new)
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text
        if text.lower().startswith("json\n"):
            text = text.split("\n", 1)[1]
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3]
    text = re.sub(r"(^|\s)//.*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    text = re.sub(r",(\s*[\]\}])", r"\1", text)
    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            return obj
        if isinstance(obj, dict) and "trees" in obj:
            return obj["trees"]
        return [obj]
    except json.JSONDecodeError:
        m = re.search(r"\[.*\]", text, flags=re.DOTALL)
        if m:
            return json.loads(re.sub(r",(\s*[\]\}])", r"\1", m.group(0)))
    raise ValueError("Could not parse JSON")


def repair_tree(tree: dict, stats: dict) -> dict:
    if "root" not in tree:
        return tree
    def fix_node(node):
        if "leaf" in node:
            node["leaf"] = int(bool(node["leaf"]))
            return node
        feat = node.get("feature")
        val = node.get("value")
        if feat in stats and val is not None:
            node["value"] = max(stats[feat]["min"], min(stats[feat]["max"], val))
        for key in ("left", "right"):
            if key not in node:
                node[key] = {"leaf": 0}
            else:
                node[key] = fix_node(node[key])
        return node
    tree["root"] = fix_node(tree["root"])
    return tree


def is_valid_tree(tree: dict) -> bool:
    if not isinstance(tree, dict) or "root" not in tree:
        return False
    def check(node, depth=0):
        if "leaf" in node:
            return node["leaf"] in (0, 1)
        if depth > LLM_TREE_MAX_DEPTH:
            return False
        feat = node.get("feature")
        op = node.get("op")
        if feat not in PREDICTORS or op not in ("<=", ">", "<", ">="):
            return False
        return (check(node.get("left", {}), depth+1) and
                check(node.get("right", {}), depth+1))
    return check(tree["root"])


def deduplicate(trees: list) -> list:
    seen = set()
    unique = []
    for t in trees:
        t_copy = {k: v for k, v in t.items() if k != "tree_id"}
        h = hashlib.md5(json.dumps(t_copy, sort_keys=True).encode()).hexdigest()
        if h not in seen:
            seen.add(h)
            unique.append(t)
    return unique


# ══════════════════════════════════════════════
# Tree scoring (uses TRAINING DATA ONLY)
# ══════════════════════════════════════════════
def score_tree_on_training(tree: dict, X_train, y_train) -> float:
    y_pred = predict_llm_tree(X_train, tree)
    if len(np.unique(y_pred)) == 1:
        return 0.0
    return f1_score(y_train, y_pred, average="macro")


def score_ensemble_on_training(trees: list, X_train, y_train) -> float:
    """Use 3-fold CV WITHIN training data for ensemble selection."""
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    f1s = []
    for tr_idx, va_idx in skf.split(X_train, y_train):
        X_va = X_train.iloc[va_idx]
        y_va = y_train[va_idx]
        _, p = llm_ensemble_predict(X_va, trees)
        y_pred = (p >= 0.5).astype(int)
        if len(np.unique(y_pred)) == 1:
            f1s.append(0.0)
        else:
            f1s.append(f1_score(y_va, y_pred, average="macro"))
    return np.mean(f1s)


# ══════════════════════════════════════════════
# Per-fold generation pipeline
# ══════════════════════════════════════════════
def generate_for_fold(sp_code: str, sp_name: str, fold_id: int,
                      X_train, y_train, budget: BudgetTracker,
                      dry_run: bool = False) -> List[dict]:
    """Generate, validate, and select trees for a single fold."""
    stats = compute_stats_from_training(X_train)
    audit_dir = OUTPUTS_DIR / sp_code / "llm_audit_per_fold" / f"fold_{fold_id}"
    audit_dir.mkdir(parents=True, exist_ok=True)

    n_batches = N_CANDIDATES_PER_FOLD // BATCH_SIZE
    print(f"\n  Fold {fold_id}: generating {N_CANDIDATES_PER_FOLD} candidates in {n_batches} batches")

    all_candidates = []

    for batch_idx in range(n_batches):
        if not budget.can_afford():
            print(f"    BUDGET EXCEEDED. Stopping generation.")
            break

        n_this = min(BATCH_SIZE, N_CANDIDATES_PER_FOLD - batch_idx * BATCH_SIZE)
        temp = TEMPERATURES[batch_idx % len(TEMPERATURES)]
        print(f"    Batch {batch_idx+1}/{n_batches} (n={n_this}, temp={temp})...", end=" ", flush=True)

        prompt = build_prompt_for_fold(sp_code, sp_name, X_train, y_train,
                                        n_this, fold_id)

        # Save prompt
        (audit_dir / f"batch_{batch_idx+1}_prompt.txt").write_text(prompt, encoding="utf-8")

        if dry_run:
            print(f"[DRY RUN]")
            budget.record_call()
            continue

        # API call with retries
        batch_trees = []
        for attempt in range(1, 4):
            try:
                raw = call_openai(prompt, temperature=temp)
                batch_trees = extract_json(raw)
                (audit_dir / f"batch_{batch_idx+1}_raw.txt").write_text(raw, encoding="utf-8")
                budget.record_call()
                break
            except Exception as e:
                print(f"failed ({e}, retry {attempt})")
                if attempt == 3:
                    batch_trees = []
                time.sleep(2 ** attempt)

        # Validate
        valid = []
        for t in batch_trees:
            t = repair_tree(t, stats)
            if is_valid_tree(t):
                valid.append(t)
        print(f"received {len(batch_trees)}, valid {len(valid)}")
        all_candidates.extend(valid)

    if dry_run:
        return []

    # Deduplicate
    all_candidates = deduplicate(all_candidates)
    print(f"    {len(all_candidates)} unique candidates")

    if not all_candidates:
        print(f"    ⚠ No candidates! Returning empty.")
        return []

    # Score on training data
    scored = [(t, score_tree_on_training(t, X_train, y_train)) for t in all_candidates]
    scored.sort(key=lambda x: -x[1])
    good = [(t, f1) for t, f1 in scored if f1 >= MIN_TREE_F1]
    if len(good) < 5:
        good = scored[:10]

    # Greedy selection on TRAINING DATA
    print(f"    Greedy selection from {len(good)} good candidates...")
    selected = [good[0][0]]
    remaining = [t for t, _ in good[1:]]
    best_score = score_ensemble_on_training(selected, X_train, y_train)
    peak_score = best_score
    peak_size = 1
    no_improve = 0

    while len(selected) < MAX_ENSEMBLE_SIZE and remaining and no_improve < MAX_NO_IMPROVE:
        best_add = None
        best_new = best_score
        for i, cand in enumerate(remaining):
            s = score_ensemble_on_training(selected + [cand], X_train, y_train)
            if s > best_new:
                best_new = s
                best_add = i
        if best_add is not None:
            selected.append(remaining.pop(best_add))
            best_score = best_new
            no_improve = 0
            if best_score > peak_score:
                peak_score = best_score
                peak_size = len(selected)
        else:
            no_improve += 1
            selected.append(remaining.pop(0))

    if peak_size < len(selected):
        selected = selected[:peak_size]

    print(f"    Final fold ensemble: {len(selected)} trees, training F1={peak_score:.4f}")

    # Save fold metadata
    meta = {
        "fold": fold_id,
        "n_candidates": len(all_candidates),
        "n_selected": len(selected),
        "training_cv_f1": peak_score,
        "n_train_samples": len(y_train),
        "timestamp": datetime.now().isoformat(),
    }
    (audit_dir / "fold_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return selected


def generate_per_fold_for_species(sp_code: str, sp_name: str,
                                    X, y, budget: BudgetTracker,
                                    dry_run: bool = False) -> Dict[int, List[dict]]:
    """Generate trees for all 5 folds of a species."""
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    fold_trees = {}
    for fold_id, (tr_idx, te_idx) in enumerate(skf.split(X, y), 1):
        X_train = X.iloc[tr_idx].copy()
        y_train = y[tr_idx]
        trees = generate_for_fold(sp_code, sp_name, fold_id, X_train, y_train,
                                   budget, dry_run=dry_run)
        fold_trees[fold_id] = trees

        print(f"  Budget after fold {fold_id}: {budget.summary()}")
        if not budget.can_afford():
            print(f"  ⚠ Budget nearly exhausted. Skipping remaining folds.")
            break

    return fold_trees


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--species", type=str, default=None, choices=list(SPECIES.keys()))
    parser.add_argument("--max-cost", type=float, default=50.0,
                        help="Maximum USD to spend on API calls (default: 50)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Estimate cost and save prompts without calling API")
    args = parser.parse_args()

    if not EXCEL_FILE.exists():
        print(f"ERROR: {EXCEL_FILE} not found.")
        sys.exit(1)
    if not args.dry_run and not OPENAI_API_KEY:
        print("ERROR: OPENAI_API_KEY not set.")
        sys.exit(1)

    df = clean_decimal_commas(load_excel(str(EXCEL_FILE)))
    species_list = [args.species] if args.species else list(SPECIES.keys())

    LLM_TREES_DIR.mkdir(parents=True, exist_ok=True)
    budget = BudgetTracker(args.max_cost)

    # Estimate total cost upfront
    n_calls_total = (N_CANDIDATES_PER_FOLD // BATCH_SIZE) * N_FOLDS * len(species_list)
    est_cost = n_calls_total * budget.estimate_call_cost()
    print(f"\nEstimated total: {n_calls_total} API calls, ~${est_cost:.2f}")
    print(f"Budget cap: ${args.max_cost:.2f}")
    if est_cost > args.max_cost:
        print(f"⚠ Estimated cost exceeds budget. Will stop when budget reached.")

    if not args.dry_run:
        confirm = input("Proceed? (y/n): ")
        if confirm.lower() != "y":
            print("Aborted.")
            sys.exit(0)

    for sp_code in species_list:
        sp = SPECIES[sp_code]
        print(f"\n{'='*70}")
        print(f"Per-fold generation: {sp['full_name']} ({sp_code})")
        print(f"{'='*70}")

        X, y = build_species_frame(df, sp)
        print(f"  Total samples: {len(y)} (presence={int(y.sum())}, absence={int((y==0).sum())})")

        fold_trees = generate_per_fold_for_species(sp_code, sp["full_name"], X, y,
                                                    budget, dry_run=args.dry_run)

        if fold_trees and not args.dry_run:
            # Save per-fold trees (one JSON file with all folds)
            out_path = LLM_TREES_DIR / f"paper_llm_trees_{sp_code}_per_fold.json"
            out_data = {
                f"fold_{fold_id}": [{"tree_id": i+1, **t} for i, t in enumerate(trees)]
                for fold_id, trees in fold_trees.items()
            }
            out_path.write_text(json.dumps(out_data, indent=2), encoding="utf-8")
            print(f"\n  ✓ Saved per-fold trees to {out_path}")

        if not budget.can_afford():
            print(f"\n  Budget exhausted before completing all species.")
            break

    print(f"\n{'='*70}")
    print(f"Final budget: {budget.summary()}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
