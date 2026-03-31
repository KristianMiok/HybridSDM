#!/usr/bin/env python3
"""
Shared configuration for hybrid DT-LLM SDM experiments.
All species use the same predictors, CV scheme, and evaluation metrics.
"""

from pathlib import Path

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
LLM_TREES_DIR = PROJECT_ROOT / "llm_trees"
PROMPTS_DIR = PROJECT_ROOT / "prompts"

EXCEL_FILE = DATA_DIR / "NETWORK.xlsx"

# ──────────────────────────────────────────────
# Species definitions
# ──────────────────────────────────────────────
SPECIES = {
    "AUT": {
        "code": "AUT",
        "full_name": "Austropotamobius torrentium",
        "prez_col": "AUT_PREZ",
        "trueabs_col": "AUT_TRUEABS",
        "context": "well-studied native",
    },
    "ABI": {
        "code": "ABI",
        "full_name": "Austropotamobius bihariensis",
        "prez_col": "AUB_PREZ",       # NOTE: column in data is AUB, not ABI
        "trueabs_col": "AUB_TRUEABS",
        "context": "data-poor endemic",
    },
    "FXL": {
        "code": "FXL",
        "full_name": "Faxonius limosus",
        "prez_col": "FXL_PREZ",
        "trueabs_col": "FXL_TRUEABS",
        "context": "invasive generalist",
    },
}

# ──────────────────────────────────────────────
# Predictors (fixed across all species)
# ──────────────────────────────────────────────
PREDICTORS = ["RWQ", "ALT", "FFP", "BIO1"]

FEATURE_DESCRIPTIONS = {
    "RWQ":  "Remote water quality index (dimensionless; ~0–4)",
    "ALT":  "Altitude (m a.s.l.)",
    "FFP":  "Flash-flood potential proxy",
    "BIO1": "Mean annual temperature (°C, WorldClim)",
}

# ──────────────────────────────────────────────
# Cross-validation
# ──────────────────────────────────────────────
N_FOLDS = 5
RANDOM_STATE = 42
KNN_K = 10  # for KNN imputation

# ──────────────────────────────────────────────
# Decision tree baseline
# ──────────────────────────────────────────────
DT_MAX_DEPTH = 2
DT_MIN_SAMPLES_LEAF = 5

# ──────────────────────────────────────────────
# Random Forest baseline (black-box benchmark)
# ──────────────────────────────────────────────
RF_N_ESTIMATORS = 500
RF_MAX_DEPTH = None        # unlimited
RF_MIN_SAMPLES_LEAF = 5
RF_MAX_FEATURES = "sqrt"

# ──────────────────────────────────────────────
# Hybrid grids
# ──────────────────────────────────────────────
AND_TAU_LIST = [0.50, 0.60, 0.70, 0.80]
SOFT_VETO_THETAS = [0.5, 0.6, 0.7]
SOFT_VETO_ALPHAS = [0.25, 0.5, 0.75]
SOFT_BLEND_WEIGHTS = [0.50, 0.55, 0.60, 0.625, 0.65, 0.675, 0.70]

# ──────────────────────────────────────────────
# LLM prompt generation
# ──────────────────────────────────────────────
N_LLM_TREES = 50
LLM_TREE_MAX_DEPTH = 2
QUANTILES = [0.05, 0.20, 0.40, 0.50, 0.60, 0.80, 0.95]

# ──────────────────────────────────────────────
# OpenAI API (for automated LLM tree generation)
# ──────────────────────────────────────────────
import os
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")  # set via env var or .env file
OPENAI_MODEL = "gpt-4o"           # model to use for tree generation
OPENAI_TEMPERATURE = 0.8          # moderate creativity for diversity
OPENAI_N_BATCHES = 3              # generate N_LLM_TREES across this many API calls
OPENAI_MAX_RETRIES = 3            # retries on API or parse failure
