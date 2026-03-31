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
│   └── NETWORK.xlsx              # Main dataset (place here)
├── llm_trees/
│   ├── paper_llm_trees_AUT.json  # Auto-generated LLM trees per species
│   ├── paper_llm_trees_ABI.json
│   └── paper_llm_trees_FXL.json
├── prompts/                       # Auto-generated LLM prompts (text files)
├── outputs/
│   ├── AUT/
│   │   ├── cv5_summary.csv
│   │   ├── cv5_perfold.csv
│   │   ├── dt_rules_per_fold.txt
│   │   ├── rf_feature_importances.csv
│   │   ├── shap/                  # SHAP plots & data
│   │   └── llm_audit/            # Full API audit trail
│   ├── ABI/
│   ├── FXL/
│   └── comparison/
│       ├── species_comparison.csv
│       ├── predictor_correlations.csv
│       └── correlation_heatmap.png
├── src/
│   ├── config.py                  # All hyperparameters & paths
│   ├── utils.py                   # Data loading, LLM tree eval, metrics
│   ├── explore_data.py            # Data exploration & diagnostics
│   ├── generate_prompts.py        # Build prompts (text files only)
│   ├── generate_llm_trees.py      # Automated API generation + validation
│   ├── run_cv.py                  # Main CV: DT + LLM + hybrids + RF
│   ├── run_shap.py                # SHAP analysis for RF benchmark
│   └── legacy/                    # Original FXL-only scripts
├── requirements.txt
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
pip install openai
export OPENAI_API_KEY="sk-..."
```

Place `NETWORK.xlsx` in `data/`.

## Workflow

### Step 1: Explore data
```bash
cd src
python explore_data.py
```

### Step 2: Generate LLM trees (automated via API)
```bash
python generate_llm_trees.py --dry-run   # preview prompts
python generate_llm_trees.py             # call API for all species
python generate_llm_trees.py --species AUT
```

### Step 3: Run cross-validation
```bash
python run_cv.py
```

### Step 4: SHAP analysis
```bash
python run_shap.py
```

## Reproducibility

- Fixed random seeds for CV and model training
- Every API prompt, raw response, and repair action saved in `llm_audit/`
- Generation metadata (model, temperature, timestamps) logged as JSON
- Same 5-fold CV structure used for all models within each species
- Commit the generated `llm_trees/*.json` files and reuse for all analyses

## Models Evaluated

| Model | Type | Interpretable? |
|-------|------|---------------|
| DT(d=2) | Shallow decision tree | Full |
| LLM(N-tree) | LLM-generated ensemble | Full |
| AND / OR / k-veto / soft-veto / blend | Hybrid variants | Full |
| stacked(logistic) | Logistic meta-model | Semi |
| RF(500) | Random Forest benchmark | Black-box |
