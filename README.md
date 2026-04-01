# Hybrid DT–LLM Framework for Explainable Species Distribution Modelling

Reproducible experiments for the manuscript: *"Hybrid decision tree–language model framework for explainable species distribution modelling"*

## Species

| Code | Species | Context |
|------|---------|---------|
| AUT  | *Austropotamobius torrentium* | Well-studied native |
| ABI  | *Austropotamobius bihariensis* | Data-poor endemic |
| FXL  | *Faxonius limosus* | Invasive generalist |

## Predictors

All models use 4 ecologically relevant predictors:
- **RWQ** — Remote water quality index
- **ALT** — Altitude (m a.s.l.)
- **FFP** — Flash-flood potential proxy
- **BIO1** — Mean annual temperature (°C, WorldClim)

## Repository Structure

```
hybrid-sdm-experiments/
├── data/
│   └── NETWORK.xlsx                    # Main dataset (place here)
├── llm_trees/
│   ├── paper_llm_trees_AUT.json        # Full-mode LLM trees (data-informed)
│   ├── paper_llm_trees_ABI.json
│   ├── paper_llm_trees_FXL.json
│   ├── paper_llm_trees_AUT_pure.json   # Pure-mode LLM trees (priors only)
│   ├── paper_llm_trees_ABI_pure.json
│   └── paper_llm_trees_FXL_pure.json
├── outputs/
│   ├── AUT/
│   │   ├── cv5_summary.csv             # Model comparison table
│   │   ├── cv5_perfold.csv             # Per-fold metrics
│   │   ├── dt_rules_per_fold.txt       # DT stability analysis
│   │   ├── rf_feature_importances.csv
│   │   ├── shap/                       # SHAP plots & data
│   │   ├── llm_audit/                  # v1 API audit trail
│   │   └── llm_audit_v2/              # v2 API audit trail
│   ├── ABI/
│   ├── FXL/
│   └── comparison/
│       ├── species_comparison.csv
│       ├── predictor_correlations.csv
│       └── correlation_heatmap.png
├── src/
│   ├── config.py                       # All hyperparameters & paths
│   ├── utils.py                        # Data loading, LLM tree eval, metrics
│   ├── explore_data.py                 # Data exploration & diagnostics
│   ├── generate_prompts.py             # Build prompts (text files only)
│   ├── generate_llm_trees.py           # v1: basic API generation
│   ├── generate_llm_trees_v2.py        # v2: over-generate + greedy selection
│   ├── run_cv.py                       # 5-fold CV: DT + LLM + hybrids + RF
│   ├── run_shap.py                     # SHAP analysis for RF benchmark
│   └── legacy/                         # Original FXL-only scripts
├── requirements.txt
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
conda install -c conda-forge openai   # or: pip install openai
export OPENAI_API_KEY="sk-..."
```

Place `NETWORK.xlsx` in `data/`.

## Workflow

### Step 1: Explore data
```bash
python src/explore_data.py
```
Outputs: correlation matrix (ALT–BIO1 r = −0.93), class balance, predictor boxplots.

### Step 2: Generate LLM trees (v2 — recommended)
```bash
# Full mode (class-conditional stats + DT splits in prompt):
python src/generate_llm_trees_v2.py

# Pure mode (only ecological priors + aggregate stats — for ablation):
python src/generate_llm_trees_v2.py --pure

# Single species:
python src/generate_llm_trees_v2.py --species AUT
```

v2 improvements over v1:
- Over-generates 150 candidates, keeps best via greedy ensemble selection
- Multi-temperature (0.4, 0.7, 0.9) for diversity
- Class-conditional statistics guide threshold placement
- DT splits shown as reference (full mode only)
- Iterative refinement: round 2 is guided by top trees from round 1
- Adaptive ensemble size: stops adding trees when performance drops

### Step 3: Run cross-validation
```bash
python src/run_cv.py            # using full-mode trees
python src/run_cv.py --pure     # using pure-mode trees (ablation)
python src/run_cv.py --species AUT
```

### Step 4: SHAP analysis (requires shap package)
```bash
python src/run_shap.py
```

### What the LLM sees vs. does NOT see

**Sees:** aggregate statistics, quantiles, class balance, ecological priors, (full mode: class-conditional means, DT splits)

**Does NOT see:** individual data points, site coordinates, raw occurrence records

The greedy ensemble selection evaluates candidate trees against training data — this is analogous to hyperparameter tuning, not data leakage.

## Models Evaluated

| Model | Type | Interpretable? |
|-------|------|---------------|
| DT(d=2) | Shallow decision tree | Full |
| LLM(N-tree) | LLM-generated ensemble | Full |
| AND / OR / k-veto / soft-veto / blend | Hybrid DT+LLM variants | Full |
| stacked(logistic) | Logistic meta-model | Semi |
| RF(500) | Random Forest benchmark | Black-box |

## Reproducibility

- Fixed random seeds (42) for CV splits and model training
- Every API prompt, raw response, and repair action saved in `llm_audit_v2/`
- Generation metadata (model, temperature, timestamps, tree scores) logged as JSON
- Same 5-fold CV structure used for all models within each species
- Commit `llm_trees/*.json` and reuse for all downstream analyses
- LLM outputs are non-deterministic; committed trees are the canonical set
