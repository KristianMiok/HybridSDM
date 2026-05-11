#!/usr/bin/env python3
"""
generate_llm_trees_v2.py — Improved automated LLM tree generation.

Key improvements over v1:
  1. Over-generate: produce 150 candidate trees (3x target), keep best 50
  2. Multi-temperature: low (0.4) for stable trees, high (0.9) for diversity
  3. Per-tree evaluation: score each tree on training data, drop poor performers
  4. Iterative refinement: round 2 prompt includes top-performing trees from round 1
  5. Data-informed priors: DT splits from training data guide the LLM prompt

Setup:
  export OPENAI_API_KEY="sk-..."

Usage:
  python generate_llm_trees_v2.py                 # all species
  python generate_llm_trees_v2.py --species AUT   # one species
  python generate_llm_trees_v2.py --dry-run       # preview only
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
    N_FOLDS, RANDOM_STATE, DT_MAX_DEPTH, DT_MIN_SAMPLES_LEAF, KNN_K,
)
from utils import (
    load_excel, clean_decimal_commas, build_species_frame,
    load_llm_trees, predict_llm_tree, llm_ensemble_predict,
)

# ══════════════════════════════════════════════
# V2 CONFIGURATION
# ══════════════════════════════════════════════
N_TARGET_TREES = 50          # final ensemble size
N_OVERGENERATE = 150         # generate this many candidates
BATCH_SIZE = 25              # trees per API call
TEMPERATURES = [0.4, 0.7, 0.9]  # mix of conservative and creative
MIN_TREE_F1 = 0.40           # drop trees worse than this on training data
REFINEMENT_ROUNDS = 2        # round 1 = base, round 2 = guided by best from round 1
TOP_K_FOR_REFINEMENT = 5     # show this many top trees in round 2 prompt


# ══════════════════════════════════════════════
# ECOLOGICAL PRIORS (species-specific)
# ══════════════════════════════════════════════
SPECIES_PRIORS = {
    "AUT": [
        "Austropotamobius torrentium is a cold-water species found in clean, fast-flowing montane streams.",
        "ALT: prefers moderate altitudes (200–800 m); declines at very low or very high elevations.",
        "RWQ: strongly associated with high water quality (higher RWQ values indicate worse quality, so LOWER RWQ favors presence).",
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
        "BIO1: tolerates a wide temperature range; may favor warmer conditions relative to native crayfish.",
        "FFP: low flash-flood potential (calm, larger rivers) strongly favors presence.",
    ],
}


# ══════════════════════════════════════════════
# PROMPT BUILDING
# ══════════════════════════════════════════════

def compute_stats(X: pd.DataFrame) -> Dict[str, dict]:
    stats = {}
    for c in PREDICTORS:
        s = pd.to_numeric(X[c], errors="coerce").dropna()
        if s.empty:
            continue
        stats[c] = {
            "min": float(s.min()),
            "max": float(s.max()),
            "median": float(s.quantile(0.5)),
            "quantiles": {f"q{int(q*100):02d}": float(s.quantile(q)) for q in QUANTILES},
        }
    return stats


def compute_class_conditional_stats(X: pd.DataFrame, y: np.ndarray) -> str:
    """Compute predictor stats split by class — gives LLM much better guidance."""
    lines = []
    for c in PREDICTORS:
        vals_pres = X.loc[y == 1, c].dropna()
        vals_abs = X.loc[y == 0, c].dropna()
        if vals_pres.empty or vals_abs.empty:
            continue
        lines.append(
            f"- {c}: Presence mean={vals_pres.mean():.2f} (q25={vals_pres.quantile(0.25):.2f}, "
            f"q75={vals_pres.quantile(0.75):.2f}) | "
            f"Absence mean={vals_abs.mean():.2f} (q25={vals_abs.quantile(0.25):.2f}, "
            f"q75={vals_abs.quantile(0.75):.2f})"
        )
    return "\n".join(lines)


def get_dt_splits(X: pd.DataFrame, y: np.ndarray) -> str:
    """Train a DT and extract its splits as hints for the LLM."""
    dt = DecisionTreeClassifier(
        max_depth=DT_MAX_DEPTH,
        min_samples_leaf=DT_MIN_SAMPLES_LEAF,
        random_state=RANDOM_STATE,
    )
    dt.fit(X, y)
    rules = export_text(dt, feature_names=PREDICTORS)
    return rules


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


def build_prompt(sp_code: str, sp_name: str, X: pd.DataFrame, y: np.ndarray,
                 n_trees: int, round_num: int = 1, top_trees_json: str = None,
                 pure: bool = False) -> str:
    """Build generation prompt. If pure=True, omit DT splits and class-conditional stats."""
    stats = compute_stats(X)
    stats_block = format_stats(stats)
    priors = SPECIES_PRIORS.get(sp_code, [])
    priors_block = "\n".join(f"- {p}" for p in priors)
    n1, n0 = int(y.sum()), int((y == 0).sum())
    class_ratio = f"presence={n1} | true_absence={n0} (total={len(y)})"

    # Data-informed sections (omitted in pure mode)
    class_stats_section = ""
    dt_section = ""
    if not pure:
        class_stats = compute_class_conditional_stats(X, y)
        dt_rules = get_dt_splits(X, y)
        class_stats_section = f"""
CLASS-CONDITIONAL STATISTICS (use these to place thresholds where classes separate):
{class_stats}
"""
        dt_section = f"""
DATA-DRIVEN DECISION TREE (for reference — a DT trained on this data found these splits):
{dt_rules}
"""

    refinement_section = ""
    if round_num > 1 and top_trees_json:
        refinement_section = f"""
GUIDANCE FROM PREVIOUS ROUND:
The following trees performed best on held-out data. Generate NEW trees that explore
SIMILAR threshold ranges but with DIFFERENT structures and feature combinations.
Do NOT copy these trees exactly — use them as inspiration for where good thresholds lie.

Top-performing trees from round 1:
{top_trees_json}

"""

    return f"""You are an aquatic ecologist building a species distribution model.
Produce EXACTLY {n_trees} shallow decision trees (depth ≤ {LLM_TREE_MAX_DEPTH}) to predict
{sp_name} ({sp_code}_BIN) presence (1) vs true absence (0) in Romanian rivers.

OPTIMIZE FOR **Macro-F1** (not accuracy). Class balance: {class_ratio}.

ALLOWED FEATURES (ONLY): {', '.join(PREDICTORS)}

ECOLOGICAL PRIORS:
{priors_block}
{class_stats_section}
OVERALL PREDICTOR RANGES:
{stats_block}
{dt_section}
{refinement_section}DIVERSITY & QUALITY CONSTRAINTS:
- Cycle root features across trees (use each of [{', '.join(PREDICTORS)}] roughly equally).
- Per tree, test at most 2 different features.
- Place thresholds where presence and absence distributions SEPARATE.
- Avoid thresholds that put nearly all data on one side of the split.
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
# API CALLS
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
    raise ValueError("Could not parse JSON from response")


# ══════════════════════════════════════════════
# TREE VALIDATION & REPAIR
# ══════════════════════════════════════════════

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
        if feat in stats and "quantile_hint" not in node and val is not None:
            rng = stats[feat]["max"] - stats[feat]["min"]
            if rng > 0:
                pct = (val - stats[feat]["min"]) / rng
                hints = [(0.3, "q20"), (0.5, "q40"), (0.7, "q60"), (1.0, "q80")]
                node["quantile_hint"] = next(h for t, h in hints if pct < t)
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
# PER-TREE EVALUATION
# ══════════════════════════════════════════════

def score_individual_trees(trees: list, X: pd.DataFrame, y: np.ndarray) -> List[Tuple[dict, float]]:
    """Score each tree individually on the data. Returns (tree, macro_f1) pairs sorted descending."""
    scored = []
    for tree in trees:
        y_pred = predict_llm_tree(X, tree)
        # Handle degenerate predictions (all same class)
        if len(np.unique(y_pred)) == 1:
            f1 = 0.0
        else:
            f1 = f1_score(y, y_pred, average="macro")
        scored.append((tree, f1))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


def score_ensemble_cv(trees: list, X: pd.DataFrame, y: np.ndarray) -> float:
    """Quick 3-fold CV score for an ensemble of trees."""
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    f1s = []
    for tr_idx, te_idx in skf.split(X, y):
        X_te = X.iloc[te_idx]
        y_te = y[te_idx]
        _, p = llm_ensemble_predict(X_te, trees)
        y_pred = (p >= 0.5).astype(int)
        if len(np.unique(y_pred)) == 1:
            f1s.append(0.0)
        else:
            f1s.append(f1_score(y_te, y_pred, average="macro"))
    return np.mean(f1s)


# ══════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════

def generate_trees_v2(sp_code: str, sp_name: str, X: pd.DataFrame, y: np.ndarray,
                      dry_run: bool = False, pure: bool = False) -> List[dict]:

    stats = compute_stats(X)
    suffix = "_pure" if pure else ""
    audit_dir = OUTPUTS_DIR / sp_code / f"llm_audit_v2{suffix}"
    audit_dir.mkdir(parents=True, exist_ok=True)

    if pure:
        print(f"  [PURE MODE] No DT splits or class-conditional stats in prompt")

    all_candidates = []

    for round_num in range(1, REFINEMENT_ROUNDS + 1):
        print(f"\n  ── Round {round_num}/{REFINEMENT_ROUNDS} ──")

        # For round 2, include top trees from round 1 as guidance
        top_trees_json = None
        if round_num > 1 and all_candidates:
            scored = score_individual_trees(all_candidates, X, y)
            top_trees = [t for t, _ in scored[:TOP_K_FOR_REFINEMENT]]
            top_trees_json = json.dumps(top_trees, indent=2)
            print(f"  Using top {TOP_K_FOR_REFINEMENT} trees from round 1 as guidance")

        # Calculate batches for this round
        trees_this_round = N_OVERGENERATE // REFINEMENT_ROUNDS
        n_batches = max(1, trees_this_round // BATCH_SIZE)

        for batch_idx in range(n_batches):
            n_this = min(BATCH_SIZE, trees_this_round - batch_idx * BATCH_SIZE)
            if n_this <= 0:
                break

            # Cycle through temperatures
            temp = TEMPERATURES[batch_idx % len(TEMPERATURES)]
            print(f"  Batch {batch_idx+1}/{n_batches} (n={n_this}, temp={temp})...", end=" ", flush=True)

            prompt = build_prompt(sp_code, sp_name, X, y, n_this,
                                  round_num=round_num, top_trees_json=top_trees_json,
                                  pure=pure)

            # Save prompt
            prompt_path = audit_dir / f"r{round_num}_batch{batch_idx+1}_prompt.txt"
            prompt_path.write_text(prompt, encoding="utf-8")

            if dry_run:
                print(f"[DRY RUN] ({len(prompt)} chars)")
                continue

            # API call with retries
            for attempt in range(1, 4):
                try:
                    raw = call_openai(prompt, temperature=temp)
                    batch_trees = extract_json(raw)
                    # Save raw response
                    (audit_dir / f"r{round_num}_batch{batch_idx+1}_raw.txt").write_text(raw, encoding="utf-8")
                    break
                except Exception as e:
                    print(f"attempt {attempt} failed: {e}")
                    if attempt == 3:
                        batch_trees = []
                    time.sleep(2 ** attempt)

            # Validate & repair
            valid = []
            for t in batch_trees:
                t = repair_tree(t, stats)
                if is_valid_tree(t):
                    valid.append(t)
            print(f"received {len(batch_trees)}, valid {len(valid)}")
            all_candidates.extend(valid)

    if dry_run:
        print(f"\n  [DRY RUN] Prompts saved to {audit_dir}/")
        return []

    # ── Deduplicate ──
    all_candidates = deduplicate(all_candidates)
    print(f"\n  Total unique candidates: {len(all_candidates)}")

    # ── Score individual trees ──
    scored = score_individual_trees(all_candidates, X, y)

    # Drop trees below minimum threshold
    good = [(t, f1) for t, f1 in scored if f1 >= MIN_TREE_F1]
    print(f"  Trees with Macro-F1 ≥ {MIN_TREE_F1}: {len(good)}/{len(scored)}")

    if len(good) < 10:
        print(f"  ⚠ Very few good trees. Relaxing threshold to keep top {N_TARGET_TREES}.")
        good = scored[:N_TARGET_TREES]

    # ── Greedy ensemble selection ──
    # Start with the best tree, greedily add trees that improve ensemble CV score
    # Stop when no more trees improve performance (adaptive size)
    print(f"  Greedy ensemble selection (max={N_TARGET_TREES})...")
    selected = [good[0][0]]
    remaining = [t for t, _ in good[1:]]

    best_score = score_ensemble_cv(selected, X, y)
    peak_score = best_score
    peak_size = 1
    no_improve_count = 0
    MAX_NO_IMPROVE = 5  # stop after 5 consecutive non-improving additions

    print(f"    1 tree: ensemble F1={best_score:.4f}")

    while len(selected) < N_TARGET_TREES and remaining and no_improve_count < MAX_NO_IMPROVE:
        best_add = None
        best_new_score = best_score

        for i, candidate in enumerate(remaining):
            trial = selected + [candidate]
            s = score_ensemble_cv(trial, X, y)
            if s > best_new_score:
                best_new_score = s
                best_add = i

        if best_add is not None:
            selected.append(remaining.pop(best_add))
            best_score = best_new_score
            no_improve_count = 0
            if best_score > peak_score:
                peak_score = best_score
                peak_size = len(selected)
        else:
            # No tree improves — count toward stopping
            no_improve_count += 1
            selected.append(remaining.pop(0))

        if len(selected) % 10 == 0:
            current_score = score_ensemble_cv(selected, X, y)
            print(f"    {len(selected)} trees: ensemble F1={current_score:.4f}")

    # Trim back to peak if we overshot
    if peak_size < len(selected):
        print(f"  Peak was at {peak_size} trees (F1={peak_score:.4f}), trimming back.")
        selected = selected[:peak_size]

    # Final ensemble score
    final_score = score_ensemble_cv(selected, X, y)
    print(f"  Final ensemble ({len(selected)} trees): CV Macro-F1={final_score:.4f}")

    # ── Assign IDs ──
    for i, tree in enumerate(selected, 1):
        tree["tree_id"] = i

    # ── Save metadata ──
    meta = {
        "species": sp_code,
        "model": OPENAI_MODEL,
        "temperatures": TEMPERATURES,
        "n_candidates_generated": len(all_candidates),
        "n_above_threshold": len(good),
        "n_selected": len(selected),
        "ensemble_cv_f1": round(final_score, 4),
        "min_tree_f1_threshold": MIN_TREE_F1,
        "refinement_rounds": REFINEMENT_ROUNDS,
        "timestamp": datetime.now().isoformat(),
    }
    (audit_dir / "generation_metadata_v2.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )

    # Save individual tree scores
    tree_scores = [{"tree_id": i+1, "individual_f1": f1}
                   for i, (_, f1) in enumerate(scored[:len(selected)])]
    (audit_dir / "tree_individual_scores.json").write_text(
        json.dumps(tree_scores, indent=2), encoding="utf-8"
    )

    return selected


def main():
    parser = argparse.ArgumentParser(description="Generate LLM trees v2 (improved)")
    parser.add_argument("--species", type=str, default=None, choices=list(SPECIES.keys()))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--pure", action="store_true",
                        help="Pure mode: LLM sees only aggregate stats + priors, no DT splits or class-conditional stats")
    args = parser.parse_args()

    if not EXCEL_FILE.exists():
        print(f"ERROR: {EXCEL_FILE} not found.")
        sys.exit(1)
    if not args.dry_run and not OPENAI_API_KEY:
        print("ERROR: OPENAI_API_KEY not set. export OPENAI_API_KEY='sk-...'")
        sys.exit(1)

    df = clean_decimal_commas(load_excel(str(EXCEL_FILE)))
    species_list = [args.species] if args.species else list(SPECIES.keys())

    LLM_TREES_DIR.mkdir(parents=True, exist_ok=True)

    mode_label = "PURE" if args.pure else "FULL"

    for sp_code in species_list:
        sp = SPECIES[sp_code]
        print(f"\n{'='*70}")
        print(f"V2 Tree Generation [{mode_label}]: {sp['full_name']} ({sp_code})")
        print(f"{'='*70}")

        X, y = build_species_frame(df, sp)
        n1, n0 = int(y.sum()), int(len(y) - y.sum())
        print(f"  Data: {len(y)} samples (presence={n1}, absence={n0})")

        trees = generate_trees_v2(sp_code, sp["full_name"], X, y,
                                   dry_run=args.dry_run, pure=args.pure)

        if trees:
            suffix = "_pure" if args.pure else ""
            out_path = LLM_TREES_DIR / f"paper_llm_trees_{sp_code}{suffix}.json"
            out_path.write_text(json.dumps(trees, indent=2), encoding="utf-8")
            print(f"\n  ✓ Saved {len(trees)} trees to {out_path}")

    if not args.dry_run:
        print(f"\nDone. Next: python run_cv.py")


if __name__ == "__main__":
    main()