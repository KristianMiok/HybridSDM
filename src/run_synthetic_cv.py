#!/usr/bin/env python3
"""Run extended CV on synthetic species (SYN_A, SYN_B, SYN_C)."""

import argparse, json, sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    OUTPUTS_DIR, LLM_TREES_DIR, PREDICTORS,
    N_FOLDS, RANDOM_STATE,
    DT_MIN_SAMPLES_LEAF, RF_N_ESTIMATORS, RF_MIN_SAMPLES_LEAF, RF_MAX_FEATURES,
    AND_TAU_LIST, SOFT_VETO_THETAS, SOFT_VETO_ALPHAS, SOFT_BLEND_WEIGHTS,
)
from run_cv_extended import run_extended_cv, load_per_fold_trees
from run_synthetic_experiments import SYN_SPECIES, SYNTHETIC_DATA_FILE, load_synthetic


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--species", type=str, default=None, choices=list(SYN_SPECIES.keys()))
    args = parser.parse_args()

    if not SYNTHETIC_DATA_FILE.exists():
        print(f"ERROR: {SYNTHETIC_DATA_FILE} not found.")
        sys.exit(1)

    species_list = [args.species] if args.species else list(SYN_SPECIES.keys())
    predictors_used = list(PREDICTORS)

    for sp_code in species_list:
        sp = SYN_SPECIES[sp_code]
        print(f"\n{'='*80}\nSynthetic CV: {sp['full_name']}\n{'='*80}")

        X, y = load_synthetic(sp_code, sp)
        print(f"  Samples: {len(y)} (presence={int(y.sum())}, absence={int((y==0).sum())})")
        print(f"  Ground-truth rules:")
        for r in sp["true_rules"]:
            print(f"    - {r}")

        per_fold = load_per_fold_trees(sp_code)
        if per_fold:
            print(f"  Using PER-FOLD trees ({sum(len(v) for v in per_fold.values())} total)")
        else:
            print(f"  ⚠ No per-fold trees found")

        df_folds, df_agg = run_extended_cv(
            X, y, sp_code, predictors_used,
            per_fold_trees=per_fold, fallback_trees=None
        )

        out_dir = OUTPUTS_DIR / sp_code
        out_dir.mkdir(parents=True, exist_ok=True)
        df_agg.to_csv(out_dir / "cv_extended_summary.csv", index=False)
        df_folds.to_csv(out_dir / "cv_extended_perfold.csv", index=False)

        print(f"\n  Top 12 models by Macro-F1:")
        cols = ["model", "macro_f1_mean", "macro_f1_std", "d2_mean", "total_tp", "total_fn"]
        print(df_agg[cols].head(12).to_string(index=False))


if __name__ == "__main__":
    main()
