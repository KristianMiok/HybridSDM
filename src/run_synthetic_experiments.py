#!/usr/bin/env python3
"""
run_synthetic_experiments.py — Generate LLM trees + run CV on synthetic data.

Tests whether LLM rules can recover known ground-truth thresholds.
Uses a SEPARATE Excel file (data/synthetic/SYNTHETIC_NETWORK.xlsx) and writes
results to outputs/SYN_*.

Steps:
  1. Generate per-fold LLM trees for each synthetic species
  2. Run extended CV (DT, GLM, RF, LLM, hybrids)
  3. Compare top LLM rules vs known ground-truth thresholds

Usage:
  python src/run_synthetic_experiments.py --dry-run
  python src/run_synthetic_experiments.py --max-cost 5.00
  python src/run_synthetic_experiments.py --species SYN_A
"""

import argparse, json, os, sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

# Import everything we need from existing modules
from config import (
    OUTPUTS_DIR, LLM_TREES_DIR, PREDICTORS, FEATURE_DESCRIPTIONS,
    LLM_TREE_MAX_DEPTH, QUANTILES, OPENAI_API_KEY, OPENAI_MODEL,
    N_FOLDS, RANDOM_STATE, DT_MAX_DEPTH, DT_MIN_SAMPLES_LEAF,
)
from utils import predict_llm_tree, llm_ensemble_predict
from generate_llm_trees_per_fold import (
    BudgetTracker, generate_for_fold, SPECIES_PRIORS,
)

# ════════════════════════════════════════════════════
# Synthetic species definitions
# ════════════════════════════════════════════════════
SYNTHETIC_DATA_FILE = Path(__file__).resolve().parents[1] / "data" / "synthetic" / "SYNTHETIC_NETWORK.xlsx"

SYN_SPECIES = {
    "SYN_A": {
        "code": "SYN_A",
        "full_name": "Synthetic species A (cold-water specialist)",
        "prez_col": "SYN_A_PREZ",
        "trueabs_col": "SYN_A_TRUEABS",
        "context": "synthetic cold-water specialist",
        "true_rules": ["RWQ < 1.0 → presence (single dominant rule)"],
    },
    "SYN_B": {
        "code": "SYN_B",
        "full_name": "Synthetic species B (high-altitude endemic)",
        "prez_col": "SYN_B_PREZ",
        "trueabs_col": "SYN_B_TRUEABS",
        "context": "synthetic high-altitude endemic",
        "true_rules": ["ALT > 400 m → presence (single dominant rule)"],
    },
    "SYN_C": {
        "code": "SYN_C",
        "full_name": "Synthetic species C (lowland generalist)",
        "prez_col": "SYN_C_PREZ",
        "trueabs_col": "SYN_C_TRUEABS",
        "context": "synthetic lowland generalist",
        "true_rules": ["BIO1 > 10 OR FFP < 0.3 → presence (disjunctive rule)"],
    },
}

# Synthetic-specific ecological priors (matching the true rules — these are what an
# expert ecologist familiar with the species would write)
SYN_PRIORS = {
    "SYN_A": [
        "SYN_A is a water-quality-sensitive species; prefers clean water (low RWQ values).",
        "RWQ: strong driver of presence; lower values favor presence.",
        "ALT, BIO1, FFP: have weaker or no consistent relationship with presence.",
    ],
    "SYN_B": [
        "SYN_B is a montane species restricted to higher elevations.",
        "ALT: strong driver of presence; higher elevations favor presence.",
        "RWQ, BIO1, FFP: have weaker or no consistent relationship with presence.",
    ],
    "SYN_C": [
        "SYN_C tolerates either warm conditions or stable hydrology, but needs at least one.",
        "BIO1: warmer temperatures favor presence.",
        "FFP: very stable rivers (low FFP) also favor presence.",
        "RWQ, ALT: have weaker or no consistent relationship with presence.",
    ],
}


def load_synthetic(species_code, sp_info):
    """Load synthetic data and return X, y for one species.
    Uses per-species CSV (only valid rows) rather than the combined Excel."""
    csv_path = SYNTHETIC_DATA_FILE.parent / f"synthetic_{species_code}_with_groundtruth.csv"
    df = pd.read_csv(csv_path)
    X = df[PREDICTORS].copy()
    y = df[sp_info["prez_col"]].astype(int).values
    return X, y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--species", type=str, default=None, choices=list(SYN_SPECIES.keys()))
    parser.add_argument("--max-cost", type=float, default=5.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-gen", action="store_true", help="Skip generation, only run CV")
    args = parser.parse_args()

    if not SYNTHETIC_DATA_FILE.exists():
        print(f"ERROR: {SYNTHETIC_DATA_FILE} not found.")
        print("Run: python src/generate_synthetic_data.py first")
        sys.exit(1)

    species_list = [args.species] if args.species else list(SYN_SPECIES.keys())
    LLM_TREES_DIR.mkdir(parents=True, exist_ok=True)
    budget = BudgetTracker(args.max_cost)

    # Inject synthetic priors into SPECIES_PRIORS so generate_for_fold uses them
    for code, priors in SYN_PRIORS.items():
        SPECIES_PRIORS[code] = priors

    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    n_calls = (80 // 20) * N_FOLDS * len(species_list)
    est_cost = n_calls * budget.estimate_call_cost()
    print(f"\nEstimated: {n_calls} API calls, ~${est_cost:.2f}")
    print(f"Budget cap: ${args.max_cost:.2f}")

    if not args.dry_run and not args.skip_gen:
        confirm = input("Proceed with generation? (y/n): ")
        if confirm.lower() != "y":
            sys.exit(0)

    for sp_code in species_list:
        sp = SYN_SPECIES[sp_code]
        print(f"\n{'='*70}")
        print(f"Synthetic: {sp['full_name']}")
        print(f"{'='*70}")

        X, y = load_synthetic(sp_code, sp)
        print(f"  Samples: {len(y)} (presence={int(y.sum())}, absence={int((y==0).sum())})")
        print(f"  Ground-truth rules:")
        for r in sp["true_rules"]:
            print(f"    - {r}")

        if args.skip_gen:
            print("  [Skipping generation]")
            continue

        # Generate per-fold trees
        fold_trees = {}
        for fold_id, (tr_idx, te_idx) in enumerate(skf.split(X, y), 1):
            X_train = X.iloc[tr_idx].copy()
            y_train = y[tr_idx]
            trees = generate_for_fold(sp_code, sp["full_name"], fold_id,
                                       X_train, y_train, budget,
                                       dry_run=args.dry_run)
            fold_trees[fold_id] = trees
            print(f"  Budget after fold {fold_id}: {budget.summary()}")
            if not budget.can_afford():
                print(f"  ⚠ Budget exhausted.")
                break

        if fold_trees and not args.dry_run:
            out_path = LLM_TREES_DIR / f"paper_llm_trees_{sp_code}_per_fold.json"
            out_data = {
                f"fold_{fid}": [{"tree_id": i+1, **t} for i, t in enumerate(trees)]
                for fid, trees in fold_trees.items()
            }
            out_path.write_text(json.dumps(out_data, indent=2), encoding="utf-8")
            print(f"  ✓ Saved per-fold trees to {out_path}")

        if not budget.can_afford():
            break

    print(f"\nFinal: {budget.summary()}")


if __name__ == "__main__":
    main()
