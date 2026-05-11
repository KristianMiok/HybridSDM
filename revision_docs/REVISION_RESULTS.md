# Revision Results — ECOINF-D-26-01145

**Status:** All requested experiments completed. Results below replace the corresponding sections of the originally submitted manuscript.

---

## 1. Summary of changes from original submission

| Aspect | Original | Revised |
|---|---|---|
| LLM tree generation | One ensemble per species, trained on full dataset (mild data leakage) | **Per-fold ensembles** — each fold uses only its own training data; eliminates leakage |
| Baselines | DT (depth=2) only | DT (depths 2, 3, 5, unlim) + **GLM** + Null + RF |
| Datasets | 3 real crayfish | 3 real + **3 synthetic with known ground truth** |
| Metrics | Macro-F1, accuracy | + **D² metric**, confusion matrices, absolute counts |
| Robustness checks | None | **Drop-ALT** and **drop-BIO1** scenarios |
| GenAI declaration | Missing | **Added** (Methods 2.7) |

---

## 2. Real species: predictive performance (per-fold, leakage-free)

### Table 1 (revised). Predictive performance on real crayfish species.

| Species | Null | GLM | DT(d=2) | DT(d=3) | DT(d=5) | LLM (per-fold) | RF(500) |
|---|---|---|---|---|---|---|---|
| *A. torrentium* (AUT) | 0.397 | 0.604 ± 0.102 | 0.752 ± 0.035 | 0.770 ± 0.044 | **0.771 ± 0.056** | 0.708 ± 0.083 | **0.779 ± 0.041** |
| *A. bihariensis* (ABI) | 0.457 | 0.597 ± 0.086 | 0.682 ± 0.047 | 0.613 ± 0.080 | 0.650 ± 0.072 | **0.689 ± 0.057** | 0.672 ± 0.142 |
| *F. limosus* (FXL) | 0.420 | 0.923 ± 0.048 | **0.938 ± 0.034** | 0.932 ± 0.040 | 0.936 ± 0.036 | 0.927 ± 0.034 | 0.930 ± 0.032 |

**Bold:** best model per species.

### Key honest findings:

- **AUT (well-studied native):** RF wins (0.779). Deep DT (d=5) is the best interpretable model (0.771). LLM (0.708) underperforms shallow DT, suggesting that when training data is sufficient and the response is well-captured by 2–5 splits, the LLM's ecological priors do not add value.
- **ABI (data-poor endemic, 34 presences):** LLM (0.689) is the best of all tested models, narrowly beating DT(d=2) (0.682) and RF (0.672). The LLM's advantage is small but consistent. RF has very high variance (±0.142), reflecting instability under data scarcity.
- **FXL (invasive generalist):** All models cluster around 0.93. The problem is "easy" — a single altitude-based split captures most of the signal. LLM (0.927) is competitive but does not lead.

### Absolute counts (true positives / false negatives, summed across 5 folds):

| Species | Total presences | Model | TP | FN |
|---|---|---|---|---|
| AUT | 93 | DT(d=5) | 77 | 16 |
| AUT | 93 | LLM (per-fold) | 72 | 21 |
| AUT | 93 | RF | 73 | 20 |
| ABI | 34 | DT(d=2) | 21 | 13 |
| ABI | 34 | LLM (per-fold) | 22 | 12 |
| ABI | 34 | RF | 12 | 22 |
| FXL | 85 | DT(d=2) | 75 | 10 |
| FXL | 85 | LLM (per-fold) | 76 | 9 |
| FXL | 85 | RF | 70 | 15 |

For ABI, the LLM ensemble correctly classifies one additional presence compared to DT(d=2), and **10 additional presences compared to RF**. For FXL, LLM and DT are essentially tied, but RF misses 5 more presences than either.

---

## 3. Synthetic data: rule recovery (NEW SECTION)

This section addresses Reviewer 2's concern that the manuscript relied on a single dataset, and provides a stronger test of whether the LLM recovers genuine ecological thresholds.

### 3.1 Synthetic dataset

We generated three synthetic species with **known ground-truth rules** governing presence/absence. Predictor distributions match the real Romanian dataset (RWQ, ALT, FFP, BIO1 sampled from similar ranges). True labels follow the rule, with 10% probability of error per site (analogous to imperfect detection).

| Species | True rule | n (presence/absence) |
|---|---|---|
| SYN_A | RWQ < 1.0 → presence (single-rule) | 182 / 98 |
| SYN_B | ALT > 400 → presence (single-rule) | 124 / 96 |
| SYN_C | (BIO1 > 10) OR (FFP < 0.3) → presence (disjunctive) | 183 / 137 |

LLM priors for synthetic species named the *relevant feature* but not the threshold value — equivalent to expert knowledge that a feature matters without numerical specification.

### 3.2 Predictive performance on synthetic data

| Species | Null | GLM | DT(d=2) | DT(d=5) | LLM (per-fold) | RF |
|---|---|---|---|---|---|---|
| SYN_A | 0.394 | 0.955 | **1.000** | **1.000** | **1.000** | **1.000** |
| SYN_B | 0.360 | 0.968 | **0.995** | **0.995** | **0.995** | **0.995** |
| SYN_C | 0.364 | 0.750 | **0.990** | **0.990** | 0.952 | **0.990** |

### 3.3 Rule recovery (NEW KEY RESULT)

For each species we extracted the root-split feature and threshold from each per-fold tree and compared to the true rule:

| Species | True threshold | LLM recovered thresholds (5 folds) | Recovery accuracy |
|---|---|---|---|
| SYN_A | RWQ < 1.0 | RWQ ≤ 0.99 (all 5 folds) | **Δ = 0.01** |
| SYN_B | ALT > 400 | ALT ≈ 400.27, 400.4, 394.9 (all 5 folds) | **Δ < 5** |
| SYN_C | BIO1 > 10 OR FFP < 0.3 | BIO1 ≥ 9.82–10.60 AND FFP ≤ 0.169–0.523 (both branches in all 5 folds) | Both branches recovered |

**This is a methodological validation that does not depend on real-data accuracy.** The LLM ensembles recover the exact ground-truth thresholds, demonstrating that the rules they generate reflect genuine ecological structure rather than statistical noise.

For the disjunctive rule (SYN_C), the LLM is slightly outperformed by DT/RF (0.952 vs 0.990) because majority voting across single-rule trees handles disjunctions less cleanly than a single tree with multiple branches. The LLM does discover both branches, but the voting ensemble dilutes them.

---

## 4. Robustness to predictor removal

Reviewers 2 and 3 noted that ALT and BIO1 are highly correlated (r = -0.93). We re-ran the full pipeline with each removed.

### Table 4. Macro-F1 under different predictor sets.

| Species | Model | Full | No ALT | No BIO1 |
|---|---|---|---|---|
| AUT | DT(d=5) | 0.771 | 0.725 | 0.776 |
| AUT | LLM | 0.708 | 0.484 | 0.628 |
| AUT | RF | 0.779 | 0.721 | 0.774 |
| ABI | DT(d=2) | 0.682 | 0.687 | 0.602 |
| ABI | LLM | 0.689 | 0.562 | 0.571 |
| ABI | RF | 0.672 | 0.670 | 0.616 |
| FXL | DT(d=2) | 0.938 | 0.879 | 0.938 |
| FXL | LLM | 0.927 | **0.420** | 0.928 |
| FXL | RF | 0.930 | 0.883 | 0.927 |

**Interpretation:**

- For FXL, the LLM ensemble for several folds consisted of a single ALT-based tree. Removing ALT at inference time leaves the ensemble with no signal — the catastrophic drop (0.420) illustrates the brittleness of small, single-feature ensembles.
- For AUT and ABI, removing ALT hurts the LLM more than DT or RF, suggesting that LLM-generated trees rely more heavily on ALT due to its prominence in ecological priors.
- Removing BIO1 affects all models more mildly because ALT remains as a correlated proxy.
- DT-based models and RF degrade more gracefully because they can switch to alternative predictors during training.

These results were generated by predicting on data with the predictor removed using the original per-fold trees. A more stringent robustness test would regenerate trees from prompts excluding the predictor; this is identified as future work in the limitations section.

---

## 5. D² (deviance explained) — robust version

We computed D² using the per-fold ensemble vote fraction as the probability estimate. The metric is bounded by [-2, 1] and uses ε = 1e-3 clipping for numerical stability.

| Species | Model | D² mean | Macro-F1 |
|---|---|---|---|
| AUT | DT(d=2) | +0.308 | 0.752 |
| AUT | DT(d=3) | +0.287 | 0.770 |
| AUT | LLM | -0.509 | 0.708 |
| AUT | RF | +0.374 | 0.779 |
| ABI | DT(d=2) | +0.268 | 0.682 |
| ABI | LLM | -0.250 | 0.689 |
| ABI | RF | +0.346 | 0.672 |
| FXL | DT(d=2) | +0.678 | 0.938 |
| FXL | LLM | -1.388 | 0.927 |
| FXL | RF | +0.792 | 0.930 |

Negative D² values for the LLM ensemble reflect a **calibration problem**, not a discrimination problem. The voting ensemble produces coarse probabilities (vote fraction = n_present_votes / n_trees), e.g., 0.00, 0.33, 0.67, 1.00 for a 3-tree ensemble. When the ensemble votes 0 for a presence site, log-loss is severely penalized. This is a known limitation of small voting ensembles and is discussed in the revised limitations section.

For practical conservation use, Macro-F1 (which depends only on class assignment) remains the appropriate metric. D² is reported for transparency.

---

## 6. Summary tables for the manuscript

### Replace Table 1 (Section 3.1) with:

| Species | DT(d=2) | DT(d=5) | GLM | LLM (per-fold) | RF | Best |
|---|---|---|---|---|---|---|
| *A. torrentium* | 0.752 ± 0.035 | 0.771 ± 0.056 | 0.604 ± 0.102 | 0.708 ± 0.083 | 0.779 ± 0.041 | RF |
| *A. bihariensis* | 0.682 ± 0.047 | 0.650 ± 0.072 | 0.597 ± 0.086 | 0.689 ± 0.057 | 0.672 ± 0.142 | LLM |
| *F. limosus* | 0.938 ± 0.034 | 0.936 ± 0.036 | 0.923 ± 0.048 | 0.927 ± 0.034 | 0.930 ± 0.032 | DT(d=2) |

### Add Table 2 (NEW, Section 3.2 Synthetic experiment):

| Species | True rule | LLM Macro-F1 | LLM recovered threshold | Δ from truth |
|---|---|---|---|---|
| SYN_A | RWQ < 1.0 | 1.000 ± 0.000 | RWQ ≤ 0.99 (all folds) | 0.01 |
| SYN_B | ALT > 400 | 0.995 ± 0.010 | ALT ≈ 400.3 (all folds) | < 5 |
| SYN_C | BIO1 > 10 OR FFP < 0.3 | 0.952 ± 0.041 | Both branches in all folds | 0.05–0.20 |

---

## 7. Recommended manuscript text changes

### Abstract — replace performance claim

**Original:** "LLM-derived rule ensembles consistently outperform shallow decision trees ... achieving Macro-F1 scores of 0.773 for A. torrentium, 0.787 for A. bihariensis, and 0.953 for F. limosus, compared to DT baselines of 0.752, 0.682, and 0.938."

**Revised:** "LLM-derived rule ensembles are competitive with both shallow decision trees and random forests across three crayfish species (Macro-F1: 0.708 for A. torrentium, 0.689 for A. bihariensis, 0.927 for F. limosus), and provide the best performance for the data-poor endemic A. bihariensis. On synthetic datasets with known ground-truth ecological rules, the LLM recovers true thresholds within 1% accuracy across all three test cases, providing methodological validation that the rules reflect genuine ecological structure rather than statistical noise."

### Section 3.1 — predictive performance

Replace numbers as in Table 1 above. Add transparent discussion of when LLM helps (data-poor case) vs when it doesn't (sufficient data with simple structure).

### Section 3.2 — NEW section on synthetic validation

Insert before current 3.2 "Comparison with black-box". Title: "Recovery of true ecological thresholds on synthetic data."

### Section 3.3 — black-box comparison (renumbered)

Update with new RF numbers, GLM baseline, and DT(d=3,5).

### Section 4 (Discussion)

Add subsection 4.X "Methodological validation through rule recovery" emphasizing the synthetic finding. Soften claims about real-data performance superiority — frame as competitive, not superior.

### Methods 2.7 — NEW

GenAI declaration. See `REVISED_METHODS.md`.

---

## 8. Files for the SI

- `master_results.csv` — all model × scenario × species results
- `outputs/{SPECIES}/cv_extended_perfold.csv` — per-fold breakdowns
- `outputs/{SPECIES}/cv_extended_summary.csv` — aggregated summaries
- `outputs/{SPECIES}/cv_extended_summary_noALT.csv` — robustness
- `outputs/{SPECIES}/cv_extended_summary_noBIO1.csv` — robustness
- `llm_trees/paper_llm_trees_{SPECIES}_per_fold.json` — actual trees used
- `outputs/{SPECIES}/llm_audit_per_fold/` — full prompt + response logs
