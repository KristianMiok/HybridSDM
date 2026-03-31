#!/usr/bin/env python3
"""
generate_prompts.py — Build LLM prompts for all three species.

For each species, generates:
  - paper_llm_trees_{CODE}_prompt.txt  (tree generation prompt)
  - paper_llm_trees_{CODE}_referee.txt (JSON repair prompt)

Usage:
  python generate_prompts.py                 # all species
  python generate_prompts.py --species AUT   # one species
"""

import argparse
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    EXCEL_FILE, PROMPTS_DIR, SPECIES, PREDICTORS,
    FEATURE_DESCRIPTIONS, N_LLM_TREES, LLM_TREE_MAX_DEPTH, QUANTILES,
)
from utils import load_excel, clean_decimal_commas, build_species_frame


# ──────────────────────────────────────────────
# Ecological priors per species
# ──────────────────────────────────────────────
SPECIES_PRIORS = {
    "AUT": [
        "Austropotamobius torrentium is a cold-water species found in clean, fast-flowing montane streams.",
        "ALT: prefers moderate altitudes (200–800 m); declines at very low or very high elevations.",
        "RWQ: strongly associated with high water quality (higher RWQ values).",
        "BIO1: favors cooler temperatures; occurrence typically declines above ~10 °C mean annual temperature.",
        "FFP: tolerates moderate flash-flood potential but avoids extreme hydrological instability.",
    ],
    "ABI": [
        "Austropotamobius bihariensis is a micro-endemic confined to the Apuseni Mountains, Romania.",
        "ALT: restricted to higher-altitude streams (typically >400 m); very limited low-elevation records.",
        "RWQ: requires high water quality; sensitive to pollution and habitat degradation.",
        "BIO1: adapted to cool montane climates; likely absent from warmer lowland areas.",
        "FFP: ecological requirements poorly known; use priors cautiously due to limited data.",
    ],
    "FXL": [
        "Faxonius limosus is an invasive North American crayfish with generalist ecology.",
        "ALT: primarily found at lower elevations in larger, slower rivers; uncommon in montane streams.",
        "RWQ: tolerant of degraded water quality; may even benefit from moderate eutrophication.",
        "BIO1: tolerates a wide temperature range; may favor warmer conditions relative to native crayfish.",
        "FFP: tolerates varied hydrological conditions; presence relates to accessible, perennial habitats.",
    ],
}


def compute_stats(X: pd.DataFrame) -> Dict[str, dict]:
    """Compute summary stats for each predictor."""
    stats = {}
    for c in PREDICTORS:
        if c not in X.columns:
            continue
        s = pd.to_numeric(X[c], errors="coerce").dropna()
        if s.empty:
            continue
        stats[c] = {
            "min": float(s.min()),
            "max": float(s.max()),
            "median": float(s.quantile(0.5)),
            "quantiles": {f"q{int(q*100):02d}": float(s.quantile(q)) for q in QUANTILES},
            "missing": int(X[c].isna().sum()),
        }
    return stats


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
            + f", median={st['median']:.3g}, max={st['max']:.3g} "
            + f"(missing={st['missing']})"
        )
    return "\n".join(lines)


def make_generation_prompt(sp_code: str, sp_name: str, X: pd.DataFrame, y: np.ndarray) -> str:
    stats = compute_stats(X)
    stats_block = format_stats(stats)
    priors = SPECIES_PRIORS.get(sp_code, [])
    priors_block = "\n".join(f"- {p}" for p in priors)
    roots_hint = ", ".join(PREDICTORS)

    n1, n0 = int(y.sum()), int((y == 0).sum())
    class_ratio = f"presence={n1} | true_absence={n0} (total={len(y)})"

    return f"""
You are an aquatic ecologist. Produce EXACTLY {N_LLM_TREES} shallow, human-auditable decision trees
(depth ≤ {LLM_TREE_MAX_DEPTH}) to predict {sp_name} ({sp_code}_BIN) presence (1) vs true absence (0) for Romanian rivers.

OPTIMIZE FOR
- **Macro-F1** (not accuracy). Class balance on labeled data: {class_ratio}.
- Prefer thresholds near **empirical quantiles** for stability.

ALLOWED FEATURES (ONLY): {', '.join(PREDICTORS)}

ECOLOGICAL PRIORS (soft — can be overridden if data strongly suggest otherwise):
{priors_block}

SAFE AGGREGATE STATS (choose thresholds within these ranges; anchor near quantiles):
{stats_block}

DIVERSITY & SIMPLICITY CONSTRAINTS
- Each tree MUST use a DIFFERENT root feature (cycle across [{roots_hint}]).
- Per tree, do not test the same feature more than **twice** total.
- Avoid tiny numeric boxes; keep splits coarse and interpretable.

THRESHOLD GUIDELINES
- Use thresholds close to quantiles (q20, q40, q60, q80).
- **Include a "quantile_hint"** at each non-leaf node.

OUTPUT FORMAT (STRICT JSON array of EXACTLY {N_LLM_TREES} trees; NO commentary outside JSON):
[
  {{
    "tree_id": 1,
    "target": "{sp_code}_BIN",
    "max_depth": {LLM_TREE_MAX_DEPTH},
    "root": {{
      "feature": "<RWQ|ALT|FFP|BIO1>", "op": "<=", "value": <number>, "quantile_hint": "<q20|q40|q60|q80>",
      "left":  {{ "leaf": <0_or_1> }},
      "right": {{
        "feature": "<RWQ|ALT|FFP|BIO1>", "op": ">", "value": <number>, "quantile_hint": "<q20|q40|q60|q80>",
        "left":  {{ "leaf": <0_or_1> }},
        "right": {{ "leaf": <0_or_1> }}
      }}
    }}
  }}
]

ALLOWED ops: "<=", ">", "<", ">=".

VALIDATION CHECKLIST (verify before outputting):
- depth ≤ {LLM_TREE_MAX_DEPTH}
- roots cycle through different features across trees
- thresholds within SAFE [min, max]
- quantile_hint present at every non-leaf node
- no micro-rectangles; respect monotone tendencies unless data contradict
""".strip()


def make_referee_prompt(sp_code: str) -> str:
    return f"""
You are a JSON referee. You will receive a JSON array of shallow decision trees (depth ≤ {LLM_TREE_MAX_DEPTH})
for {sp_code}_BIN using ONLY features: {', '.join(PREDICTORS)}. Your task:

1) **Validate & Repair** schema:
   - Must parse as list of objects, each with a "root".
   - Nodes: feature/op/value + left/right children. Leaves: {{"leaf": 0 or 1}}.
   - ALLOWED ops: "<=", ">", "<", ">=".
   - If a node has feature/op/value but missing child → create leaf with other class.

2) **Enforce constraints**:
   - depth ≤ {LLM_TREE_MAX_DEPTH}, features only in {{{', '.join(PREDICTORS)}}}.
   - Clip out-of-range thresholds to [min, max] from the companion prompt.
   - Remove duplicate trees (identical root + thresholds + leaves).

3) **Quantile hints**:
   - If "quantile_hint" missing on a non-leaf → add best guess (q20/q40/q60/q80).
   - No free-text commentary.

4) **Output**: Return ONLY the repaired JSON array (no extra text).
""".strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--species", type=str, default=None, choices=list(SPECIES.keys()))
    args = parser.parse_args()

    if not EXCEL_FILE.exists():
        print(f"ERROR: {EXCEL_FILE} not found. Place NETWORK.xlsx in data/")
        sys.exit(1)

    df_raw = load_excel(str(EXCEL_FILE))
    df = clean_decimal_commas(df_raw)

    species_list = [args.species] if args.species else list(SPECIES.keys())
    PROMPTS_DIR.mkdir(parents=True, exist_ok=True)

    for sp_code in species_list:
        sp = SPECIES[sp_code]
        print(f"\n{'='*60}")
        print(f"Generating prompts for {sp['full_name']} ({sp_code})")

        X, y = build_species_frame(df, sp)
        n1, n0 = int(y.sum()), int((y == 0).sum())
        print(f"  Samples: {len(y)} (presence={n1}, absence={n0})")

        gen_prompt = make_generation_prompt(sp_code, sp["full_name"], X, y)
        ref_prompt = make_referee_prompt(sp_code)

        gen_path = PROMPTS_DIR / f"paper_llm_trees_{sp_code}_prompt.txt"
        ref_path = PROMPTS_DIR / f"paper_llm_trees_{sp_code}_referee.txt"

        gen_path.write_text(gen_prompt, encoding="utf-8")
        ref_path.write_text(ref_prompt, encoding="utf-8")

        print(f"  → {gen_path.name}")
        print(f"  → {ref_path.name}")

    print(f"\nAll prompts saved to {PROMPTS_DIR}/")
    print("Next: paste each prompt into GPT-4 (temp ~0.7–0.9), save output as")
    print("  llm_trees/paper_llm_trees_{CODE}.json")


if __name__ == "__main__":
    main()
