#!/usr/bin/env python3
"""
generate_llm_trees.py — Automated LLM tree generation via OpenAI API.

Replaces the manual workflow of copy-pasting prompts into ChatGPT.
For each species:
  1. Builds the prompt (same as generate_prompts.py)
  2. Calls OpenAI API to generate trees
  3. Applies referee/repair logic automatically
  4. Validates all trees against schema
  5. Saves final JSON to llm_trees/

Full audit trail: saves raw API responses, prompts used, and repair logs.

Setup:
  export OPENAI_API_KEY="sk-..."
  pip install openai

Usage:
  python generate_llm_trees.py                 # all species
  python generate_llm_trees.py --species AUT   # one species
  python generate_llm_trees.py --dry-run       # just show prompts, don't call API
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

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    EXCEL_FILE, LLM_TREES_DIR, PROMPTS_DIR, OUTPUTS_DIR,
    SPECIES, PREDICTORS, FEATURE_DESCRIPTIONS,
    N_LLM_TREES, LLM_TREE_MAX_DEPTH, QUANTILES,
    OPENAI_API_KEY, OPENAI_MODEL, OPENAI_TEMPERATURE,
    OPENAI_N_BATCHES, OPENAI_MAX_RETRIES,
)
from utils import load_excel, clean_decimal_commas, build_species_frame


# ──────────────────────────────────────────────
# Ecological priors per species (same as generate_prompts.py)
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


# ══════════════════════════════════════════════
# PROMPT BUILDING
# ══════════════════════════════════════════════

def compute_stats(X: pd.DataFrame) -> Dict[str, dict]:
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
            + f", median={st['median']:.3g}, max={st['max']:.3g}"
        )
    return "\n".join(lines)


def build_generation_prompt(sp_code: str, sp_name: str, X: pd.DataFrame, y: np.ndarray,
                            n_trees: int) -> str:
    """Build the generation prompt for a batch of trees."""
    stats = compute_stats(X)
    stats_block = format_stats(stats)
    priors = SPECIES_PRIORS.get(sp_code, [])
    priors_block = "\n".join(f"- {p}" for p in priors)
    roots_hint = ", ".join(PREDICTORS)
    n1, n0 = int(y.sum()), int((y == 0).sum())
    class_ratio = f"presence={n1} | true_absence={n0} (total={len(y)})"

    return f"""You are an aquatic ecologist. Produce EXACTLY {n_trees} shallow, human-auditable decision trees
(depth ≤ {LLM_TREE_MAX_DEPTH}) to predict {sp_name} ({sp_code}_BIN) presence (1) vs true absence (0) for Romanian rivers.

OPTIMIZE FOR
- **Macro-F1** (not accuracy). Class balance: {class_ratio}.
- Prefer thresholds near **empirical quantiles** for stability.

ALLOWED FEATURES (ONLY): {', '.join(PREDICTORS)}

ECOLOGICAL PRIORS (soft — can be overridden if data strongly suggest otherwise):
{priors_block}

SAFE AGGREGATE STATS (anchor thresholds near quantiles):
{stats_block}

DIVERSITY & SIMPLICITY CONSTRAINTS
- Each tree MUST use a DIFFERENT root feature (cycle across [{roots_hint}]).
- Per tree, do not test the same feature more than twice total.
- Avoid tiny numeric boxes; keep splits coarse and interpretable.

THRESHOLD GUIDELINES
- Use thresholds close to quantiles (q20, q40, q60, q80).
- Include a "quantile_hint" at each non-leaf node.

OUTPUT FORMAT (STRICT JSON array of EXACTLY {n_trees} trees; NO commentary outside JSON):
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
Return ONLY the JSON array. No markdown fences, no commentary."""


def build_referee_prompt(sp_code: str, stats: dict) -> str:
    """Build the referee/repair system prompt."""
    ranges = {}
    for c in PREDICTORS:
        st = stats.get(c)
        if st:
            ranges[c] = {"min": st["min"], "max": st["max"]}

    return f"""You are a JSON referee. Validate and repair a JSON array of decision trees for {sp_code}_BIN.

RULES:
1. Must be a valid JSON array of objects, each with "root".
2. Nodes: feature/op/value + left/right children. Leaves: {{"leaf": 0 or 1}}.
3. ALLOWED ops: "<=", ">", "<", ">=". Features: ONLY {', '.join(PREDICTORS)}.
4. depth ≤ {LLM_TREE_MAX_DEPTH}.
5. Clip thresholds to valid ranges: {json.dumps(ranges)}.
6. Add "quantile_hint" if missing on non-leaf nodes.
7. Remove exact duplicate trees.
8. Return ONLY the repaired JSON array, no extra text."""


# ══════════════════════════════════════════════
# OPENAI API CALLS
# ══════════════════════════════════════════════

def call_openai(prompt: str, system_prompt: str = "", temperature: float = 0.8,
                model: str = "gpt-4o") -> str:
    """Call OpenAI API and return the response text."""
    try:
        from openai import OpenAI
    except ImportError:
        print("ERROR: openai package not installed. Run: pip install openai")
        sys.exit(1)

    if not OPENAI_API_KEY:
        print("ERROR: OPENAI_API_KEY not set.")
        print("  Set it via: export OPENAI_API_KEY='sk-...'")
        print("  Or create a .env file in the project root.")
        sys.exit(1)

    client = OpenAI(api_key=OPENAI_API_KEY)

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=16000,
    )

    return response.choices[0].message.content


# ══════════════════════════════════════════════
# JSON PARSING & VALIDATION
# ══════════════════════════════════════════════

def extract_json_from_response(text: str) -> list:
    """Robustly extract JSON array from LLM response text."""
    # Clean smart quotes
    for old, new in [("\u201c", '"'), ("\u201d", '"'), ("\u201e", '"'),
                     ("\u201f", '"'), ("\u2018", "'"), ("\u2019", "'")]:
        text = text.replace(old, new)

    # Strip markdown fences
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text
        if text.lower().startswith("json\n"):
            text = text.split("\n", 1)[1]
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3]

    # Remove JS comments
    text = re.sub(r"(^|\s)//.*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)

    # Remove trailing commas
    text = re.sub(r",(\s*[\]\}])", r"\1", text)

    # Try direct parse
    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            return obj
        if isinstance(obj, dict) and "trees" in obj:
            return obj["trees"]
        return [obj]
    except json.JSONDecodeError:
        pass

    # Try finding array
    m = re.search(r"\[.*\]", text, flags=re.DOTALL)
    if m:
        try:
            return json.loads(re.sub(r",(\s*[\]\}])", r"\1", m.group(0)))
        except json.JSONDecodeError:
            pass

    raise ValueError("Could not extract valid JSON array from response.")


def validate_tree(tree: dict, stats: dict) -> Tuple[bool, List[str]]:
    """Validate a single tree. Returns (is_valid, list_of_issues)."""
    issues = []

    if not isinstance(tree, dict):
        return False, ["Not a dict"]
    if "root" not in tree:
        return False, ["Missing 'root'"]

    def check_node(node, depth=0):
        if "leaf" in node:
            if node["leaf"] not in (0, 1):
                issues.append(f"Invalid leaf value: {node['leaf']}")
            return

        if depth > LLM_TREE_MAX_DEPTH:
            issues.append(f"Depth {depth} exceeds max {LLM_TREE_MAX_DEPTH}")
            return

        feat = node.get("feature")
        op = node.get("op")
        val = node.get("value")

        if feat not in PREDICTORS:
            issues.append(f"Invalid feature: {feat}")
        if op not in ("<=", ">", "<", ">="):
            issues.append(f"Invalid op: {op}")
        if val is not None and feat in stats:
            if val < stats[feat]["min"] or val > stats[feat]["max"]:
                issues.append(f"Threshold {feat}={val} out of range [{stats[feat]['min']}, {stats[feat]['max']}]")

        for child_key in ("left", "right"):
            if child_key in node:
                check_node(node[child_key], depth + 1)
            else:
                issues.append(f"Missing '{child_key}' child")

    check_node(tree["root"])
    return len(issues) == 0, issues


def repair_tree(tree: dict, stats: dict) -> dict:
    """Attempt to repair common issues in a tree."""
    if "root" not in tree:
        return tree

    def repair_node(node):
        if "leaf" in node:
            node["leaf"] = int(bool(node["leaf"]))
            return node

        feat = node.get("feature")
        val = node.get("value")

        # Clip threshold
        if feat in stats and val is not None:
            node["value"] = max(stats[feat]["min"],
                               min(stats[feat]["max"], val))

        # Add missing quantile_hint
        if feat in stats and "quantile_hint" not in node and val is not None:
            rng = stats[feat]["max"] - stats[feat]["min"]
            if rng > 0:
                pct = (val - stats[feat]["min"]) / rng
                if pct < 0.3:
                    node["quantile_hint"] = "q20"
                elif pct < 0.5:
                    node["quantile_hint"] = "q40"
                elif pct < 0.7:
                    node["quantile_hint"] = "q60"
                else:
                    node["quantile_hint"] = "q80"

        # Ensure children exist
        for child_key in ("left", "right"):
            if child_key not in node:
                node[child_key] = {"leaf": 0}
            else:
                node[child_key] = repair_node(node[child_key])

        return node

    tree["root"] = repair_node(tree["root"])
    return tree


def deduplicate_trees(trees: list) -> list:
    """Remove exact duplicate trees based on structure hash."""
    seen = set()
    unique = []
    for t in trees:
        # Hash the tree structure (ignore tree_id)
        t_copy = {k: v for k, v in t.items() if k != "tree_id"}
        h = hashlib.md5(json.dumps(t_copy, sort_keys=True).encode()).hexdigest()
        if h not in seen:
            seen.add(h)
            unique.append(t)
    return unique


# ══════════════════════════════════════════════
# MAIN GENERATION PIPELINE
# ══════════════════════════════════════════════

def generate_trees_for_species(
    sp_code: str,
    sp_name: str,
    X: pd.DataFrame,
    y: np.ndarray,
    dry_run: bool = False,
) -> List[dict]:
    """Full pipeline: generate, validate, repair, deduplicate trees."""

    stats = compute_stats(X)
    trees_per_batch = N_LLM_TREES // OPENAI_N_BATCHES
    remainder = N_LLM_TREES % OPENAI_N_BATCHES

    # Audit trail directory
    audit_dir = OUTPUTS_DIR / sp_code / "llm_audit"
    audit_dir.mkdir(parents=True, exist_ok=True)

    all_trees = []
    all_raw_responses = []

    for batch_idx in range(OPENAI_N_BATCHES):
        n_this_batch = trees_per_batch + (1 if batch_idx < remainder else 0)
        print(f"\n  Batch {batch_idx+1}/{OPENAI_N_BATCHES}: requesting {n_this_batch} trees...")

        prompt = build_generation_prompt(sp_code, sp_name, X, y, n_this_batch)

        if dry_run:
            print(f"  [DRY RUN] Would send prompt ({len(prompt)} chars) to {OPENAI_MODEL}")
            # Save prompt for inspection
            prompt_path = audit_dir / f"batch_{batch_idx+1}_prompt.txt"
            prompt_path.write_text(prompt, encoding="utf-8")
            continue

        # Call API with retries
        raw_response = None
        for attempt in range(1, OPENAI_MAX_RETRIES + 1):
            try:
                print(f"    API call (attempt {attempt}/{OPENAI_MAX_RETRIES})...")
                raw_response = call_openai(
                    prompt=prompt,
                    system_prompt="You are an expert aquatic ecologist. Output ONLY valid JSON.",
                    temperature=OPENAI_TEMPERATURE,
                    model=OPENAI_MODEL,
                )
                batch_trees = extract_json_from_response(raw_response)
                print(f"    Received {len(batch_trees)} trees")
                break
            except Exception as e:
                print(f"    Attempt {attempt} failed: {e}")
                if attempt == OPENAI_MAX_RETRIES:
                    print(f"    ERROR: All retries failed for batch {batch_idx+1}")
                    batch_trees = []
                else:
                    time.sleep(2 ** attempt)  # exponential backoff

        # Save raw response
        if raw_response:
            raw_path = audit_dir / f"batch_{batch_idx+1}_raw_response.txt"
            raw_path.write_text(raw_response, encoding="utf-8")
            all_raw_responses.append(raw_response)

        # Save prompt
        prompt_path = audit_dir / f"batch_{batch_idx+1}_prompt.txt"
        prompt_path.write_text(prompt, encoding="utf-8")

        all_trees.extend(batch_trees)

    if dry_run:
        print(f"\n  [DRY RUN] Prompts saved to {audit_dir}/")
        return []

    print(f"\n  Total raw trees: {len(all_trees)}")

    # ── Validate & repair ──
    repaired_trees = []
    repair_log = []

    for i, tree in enumerate(all_trees):
        is_valid, issues = validate_tree(tree, stats)
        if not is_valid:
            repair_log.append(f"Tree {i+1}: {issues}")
            tree = repair_tree(tree, stats)
            is_valid_after, issues_after = validate_tree(tree, stats)
            if not is_valid_after:
                repair_log.append(f"  → Still invalid after repair: {issues_after}. SKIPPING.")
                continue
            else:
                repair_log.append(f"  → Repaired successfully.")
        repaired_trees.append(tree)

    print(f"  After validation: {len(repaired_trees)} valid trees")

    # ── Optionally call referee API for additional repair ──
    if len(repaired_trees) < N_LLM_TREES * 0.8:
        print(f"  ⚠ Low yield ({len(repaired_trees)}/{N_LLM_TREES}). "
              f"Calling referee API for repair...")
        try:
            referee_system = build_referee_prompt(sp_code, stats)
            referee_input = json.dumps(repaired_trees, indent=2)
            referee_response = call_openai(
                prompt=f"Repair this JSON:\n{referee_input}",
                system_prompt=referee_system,
                temperature=0.2,  # low temp for precise repair
                model=OPENAI_MODEL,
            )
            referee_trees = extract_json_from_response(referee_response)
            # Re-validate
            valid_referee = []
            for t in referee_trees:
                t = repair_tree(t, stats)
                ok, _ = validate_tree(t, stats)
                if ok:
                    valid_referee.append(t)
            if len(valid_referee) > len(repaired_trees):
                repaired_trees = valid_referee
                print(f"  Referee improved: {len(repaired_trees)} trees")
            # Save referee response
            (audit_dir / "referee_response.txt").write_text(referee_response, encoding="utf-8")
        except Exception as e:
            print(f"  Referee call failed: {e}")

    # ── Deduplicate ──
    final_trees = deduplicate_trees(repaired_trees)
    print(f"  After deduplication: {len(final_trees)} unique trees")

    # ── Assign clean tree_ids ──
    for i, tree in enumerate(final_trees, 1):
        tree["tree_id"] = i

    # ── Save repair log ──
    if repair_log:
        log_path = audit_dir / "repair_log.txt"
        log_path.write_text("\n".join(repair_log), encoding="utf-8")

    # ── Save generation metadata ──
    meta = {
        "species": sp_code,
        "model": OPENAI_MODEL,
        "temperature": OPENAI_TEMPERATURE,
        "n_batches": OPENAI_N_BATCHES,
        "n_requested": N_LLM_TREES,
        "n_raw": len(all_trees),
        "n_valid": len(repaired_trees),
        "n_final": len(final_trees),
        "timestamp": datetime.now().isoformat(),
        "predictors": PREDICTORS,
        "max_depth": LLM_TREE_MAX_DEPTH,
    }
    meta_path = audit_dir / "generation_metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return final_trees


def main():
    parser = argparse.ArgumentParser(description="Generate LLM trees via OpenAI API")
    parser.add_argument("--species", type=str, default=None, choices=list(SPECIES.keys()))
    parser.add_argument("--dry-run", action="store_true",
                        help="Build and save prompts without calling API")
    args = parser.parse_args()

    if not EXCEL_FILE.exists():
        print(f"ERROR: {EXCEL_FILE} not found. Place NETWORK.xlsx in data/")
        sys.exit(1)

    if not args.dry_run and not OPENAI_API_KEY:
        print("ERROR: OPENAI_API_KEY not set.")
        print("  export OPENAI_API_KEY='sk-...'")
        print("  Or use --dry-run to just generate prompts.")
        sys.exit(1)

    df = clean_decimal_commas(load_excel(str(EXCEL_FILE)))
    species_list = [args.species] if args.species else list(SPECIES.keys())

    LLM_TREES_DIR.mkdir(parents=True, exist_ok=True)

    for sp_code in species_list:
        sp = SPECIES[sp_code]
        print(f"\n{'='*70}")
        print(f"Generating LLM trees: {sp['full_name']} ({sp_code})")
        print(f"{'='*70}")

        X, y = build_species_frame(df, sp)
        n1, n0 = int(y.sum()), int(len(y) - y.sum())
        print(f"  Data: {len(y)} samples (presence={n1}, absence={n0})")

        trees = generate_trees_for_species(
            sp_code, sp["full_name"], X, y,
            dry_run=args.dry_run,
        )

        if trees:
            out_path = LLM_TREES_DIR / f"paper_llm_trees_{sp_code}.json"
            out_path.write_text(json.dumps(trees, indent=2), encoding="utf-8")
            print(f"\n  ✓ Saved {len(trees)} trees to {out_path}")
        elif not args.dry_run:
            print(f"\n  ⚠ No valid trees produced for {sp_code}!")

    if args.dry_run:
        print(f"\n[DRY RUN complete] Prompts saved to outputs/{{CODE}}/llm_audit/")
        print("Review prompts, then run without --dry-run to call the API.")
    else:
        print(f"\nAll done. Trees saved to {LLM_TREES_DIR}/")
        print("Next: python run_cv.py")


if __name__ == "__main__":
    main()
