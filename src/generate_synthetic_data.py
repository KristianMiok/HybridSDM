#!/usr/bin/env python3
"""
generate_synthetic_data.py — Generate synthetic SDM dataset with KNOWN ground truth.

Addresses Reviewer 2's concern about single-dataset evaluation by providing:
  1. A second test environment with different data characteristics
  2. KNOWN response curves so we can validate that LLM rules recover true thresholds

Three synthetic species (matching ecological profiles of real species):
  - SYN_A: cold-water specialist (analog of A. torrentium)
  - SYN_B: data-poor montane endemic (analog of A. bihariensis)
  - SYN_C: warm-water lowland generalist (analog of F. limosus)

Each species has KNOWN environmental response curves used to generate presence/absence.
The LLM rules can be validated against these known thresholds.

Usage:
  python generate_synthetic_data.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import OUTPUTS_DIR, RANDOM_STATE


# ══════════════════════════════════════════════
# Ground-truth ecological response functions
# ══════════════════════════════════════════════

def syn_a_probability(rwq, alt, ffp, bio1):
    """
    SYN_A: Single dominant rule — presence if RWQ < 1.0.
    Other variables irrelevant (noise).
    """
    matches = (rwq < 1.0).astype(float)
    return 0.10 + 0.80 * matches


def syn_b_probability(rwq, alt, ffp, bio1):
    """
    SYN_B: Single dominant rule — presence if ALT > 400.
    Other variables irrelevant.
    """
    matches = (alt > 400).astype(float)
    return 0.10 + 0.80 * matches


def syn_c_probability(rwq, alt, ffp, bio1):
    """
    SYN_C: Two-rule disjunction — presence if (BIO1 > 10) OR (FFP < 0.3).
    Tests whether LLM can find both rules.
    """
    matches = ((bio1 > 10) | (ffp < 0.3)).astype(float)
    return 0.10 + 0.80 * matches


SPECIES_PROFILES = {
    "SYN_A": {
        "full_name": "Synthetic species A (cold-water specialist)",
        "prob_fn": syn_a_probability,
        "n_sites": 280,
        "true_rules": ["RWQ < 1.0 → presence (single dominant rule)"],
        "noise_level": 0.10,
        "threshold": 0.5,
    },
    "SYN_B": {
        "full_name": "Synthetic species B (high-altitude endemic)",
        "prob_fn": syn_b_probability,
        "n_sites": 220,
        "true_rules": ["ALT > 400 m → presence (single dominant rule)"],
        "noise_level": 0.08,
        "threshold": 0.5,
    },
    "SYN_C": {
        "full_name": "Synthetic species C (lowland generalist)",
        "prob_fn": syn_c_probability,
        "n_sites": 320,
        "true_rules": ["BIO1 > 10 OR FFP < 0.3 → presence (disjunctive rule)"],
        "noise_level": 0.12,
        "threshold": 0.5,
    },
}


def sample_predictors(n: int, rng: np.random.Generator) -> pd.DataFrame:
    """
    Sample predictor values matching the distribution of the real Romanian dataset.
    Approximate ranges from the real data:
      RWQ: 0.0 - 5.0, log-normal-ish
      ALT: 0 - 1500m, broad
      FFP: 0 - 5, exponential-ish
      BIO1: 5 - 13 °C, normal-ish
    """
    rwq = np.clip(rng.lognormal(mean=-0.2, sigma=0.7, size=n), 0, 5)
    alt = np.clip(rng.gamma(shape=2.5, scale=200, size=n), 5, 1500)
    ffp = np.clip(rng.exponential(scale=0.8, size=n), 0, 5)
    # BIO1 inversely related to altitude (lapse rate ~0.6°C / 100m)
    bio1_base = 12.0 - 0.006 * alt
    bio1 = np.clip(bio1_base + rng.normal(0, 0.5, size=n), 5, 13)

    return pd.DataFrame({
        "RWQ": rwq.round(3),
        "ALT": alt.round(2),
        "FFP": ffp.round(4),
        "BIO1": bio1.round(4),
    })


def generate_species_data(sp_code: str, profile: dict, rng: np.random.Generator) -> pd.DataFrame:
    """Generate sites with presence/absence labels using known response function."""
    n = profile["n_sites"]
    X = sample_predictors(n, rng)

    # Compute true probability of presence at each site
    p_true = profile["prob_fn"](
        X["RWQ"].values, X["ALT"].values, X["FFP"].values, X["BIO1"].values
    )

    # Add noise
    noise = profile["noise_level"]
    p_observed = np.clip(p_true + rng.normal(0, noise, size=n), 0, 1)

    # Use species-specific threshold to get target prevalence ~30-40%
    threshold = profile.get("threshold", 0.5)
    presence = (p_observed > threshold).astype(int)
    absence = 1 - presence

    df = X.copy()
    df.insert(0, "FID", range(n))
    df.insert(1, "CellID", range(1000, 1000 + n))
    df["Y_WGS84_DD"] = rng.uniform(45, 48, size=n).round(6)
    df["X_WGS84_DD"] = rng.uniform(20, 28, size=n).round(6)
    df[f"{sp_code}_PREZ"] = presence
    df[f"{sp_code}_TRUEABS"] = absence
    df[f"{sp_code}_TRUE_PROB"] = p_true.round(4)  # ground truth probability

    return df


def main():
    print("="*70)
    print("Synthetic SDM dataset generator")
    print("="*70)

    rng = np.random.default_rng(RANDOM_STATE)
    out_dir = OUTPUTS_DIR.parent / "data" / "synthetic"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Generate each species
    all_dfs = []
    metadata = {}

    for sp_code, profile in SPECIES_PROFILES.items():
        print(f"\n{sp_code}: {profile['full_name']}")
        df = generate_species_data(sp_code, profile, rng)
        n_pres = int(df[f"{sp_code}_PREZ"].sum())
        n_abs = int(df[f"{sp_code}_TRUEABS"].sum())
        print(f"  Generated {len(df)} sites: presence={n_pres}, absence={n_abs}")
        print(f"  Ground-truth rules:")
        for rule in profile["true_rules"]:
            print(f"    - {rule}")
        all_dfs.append(df)
        metadata[sp_code] = {
            "full_name": profile["full_name"],
            "n_sites": len(df),
            "n_presence": n_pres,
            "n_absence": n_abs,
            "true_rules": profile["true_rules"],
            "noise_level": profile["noise_level"],
        }

    # Merge all species into one combined dataset (like NETWORK.xlsx structure)
    base = all_dfs[0][["FID", "CellID", "RWQ", "ALT", "FFP", "BIO1",
                        "Y_WGS84_DD", "X_WGS84_DD"]].copy()
    for df, sp_code in zip(all_dfs, SPECIES_PROFILES.keys()):
        base[f"{sp_code}_PREZ"] = df[f"{sp_code}_PREZ"]
        base[f"{sp_code}_TRUEABS"] = df[f"{sp_code}_TRUEABS"]

    # Save Excel (matches real data structure)
    xlsx_path = out_dir / "SYNTHETIC_NETWORK.xlsx"
    base.to_excel(xlsx_path, index=False)
    print(f"\n✓ Saved combined dataset: {xlsx_path}")

    # Save per-species CSVs with ground truth probabilities
    for df, sp_code in zip(all_dfs, SPECIES_PROFILES.keys()):
        csv_path = out_dir / f"synthetic_{sp_code}_with_groundtruth.csv"
        df.to_csv(csv_path, index=False)

    # Save metadata
    meta_path = out_dir / "synthetic_metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"✓ Saved metadata: {meta_path}")

    print(f"\nNext steps:")
    print(f"  1. Add 'SYN_A', 'SYN_B', 'SYN_C' to SPECIES dict in config.py")
    print(f"  2. Run experiments: python run_cv.py --species SYN_A")
    print(f"  3. Compare LLM rules vs known ground-truth thresholds")


if __name__ == "__main__":
    main()
