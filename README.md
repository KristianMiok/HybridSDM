# Interpretable by design: language model-derived ecological rules for species distribution modelling
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
├── data/
│   └── NETWORK.xlsx                    # Occurrence dataset (Satmari et al. 2023)
├── llm_trees/
│   ├── paper_llm_trees_{CODE}.json     # LLM trees (data-informed prompt)
│   └── paper_llm_trees_{CODE}_pure.json # LLM trees (priors-only prompt)
├── outputs/
│   ├── {CODE}/
│   │   ├── cv5_summary.csv             # Model comparison
│   │   ├── cv5_perfold.csv             # Per-fold metrics
│   │   ├── dt_rules_per_fold.txt       # DT stability analysis
│   │   ├── shap/                       # SHAP plots and data
│   │   └── llm_audit_v2/              # Full API audit trail
│   └── comparison/                     # Cross-species summaries
├── src/
│   ├── config.py                       # All hyperparameters and paths
│   ├── utils.py                        # Data loading, tree evaluation, metrics
│   ├── explore_data.py                 # Data exploration and diagnostics
│   ├── generate_llm_trees.py           # v1: basic API generation
│   ├── generate_llm_trees_v2.py        # v2: over-generate + greedy selection
│   ├── run_cv.py                       # 5-fold CV: DT + LLM + hybrids + RF
│   ├── run_shap.py                     # SHAP analysis for RF benchmark
│   └── export_figure_data.py           # Export CSV data for figures
├── requirements.txt
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
conda install -c conda-forge openai
export OPENAI_API_KEY="sk-..."
```

Place `NETWORK.xlsx` in `data/`.

## Workflow

### 1. Explore data
```bash
python src/explore_data.py
```

### 2. Generate LLM trees via API
```bash
python src/generate_llm_trees_v2.py              # data-informed prompt
python src/generate_llm_trees_v2.py --pure        # priors-only prompt (ablation)
python src/generate_llm_trees_v2.py --species AUT  # single species
```

### 3. Run cross-validation
```bash
python src/run_cv.py              # using data-informed trees
python src/run_cv.py --pure       # using priors-only trees
```

### 4. SHAP analysis
```bash
python src/run_shap.py
```

### 5. Export figure data
```bash
python src/export_figure_data.py
```

## What the LLM sees

The LLM receives **aggregate statistics** (quantiles, class-conditional means, class balance), **ecological priors**, and optionally **DT splits** as reference. It never sees individual data points, site coordinates, or raw occurrence records. The greedy ensemble selection evaluates candidates against training data, analogous to hyperparameter tuning.

## Reproducibility

- Fixed random seed (42) for all CV splits and model training
- Every API prompt, raw response, and repair action saved in `llm_audit_v2/`
- Generation metadata (model, temperature, timestamps) logged as JSON
- LLM outputs are non-deterministic; committed `llm_trees/*.json` files are the canonical set used for all reported results

## Data

Occurrence data from: Satmari, A. et al. 2023. Headwater refuges: Flow protects *Austropotamobius* crayfish from *Faxonius limosus* invasion. *NeoBiota* 89: 71–94. doi: [10.17632/5vg35hc58m.3](https://doi.org/10.17632/5vg35hc58m.3)

## License

Code: MIT. Data: see original publication.
