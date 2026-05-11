# Revised Methods Section — for direct insertion into the manuscript

This file contains the rewritten Methods section incorporating all reviewer-requested changes. Lucian: paste these subsections into the manuscript replacing the corresponding originals.

---

## 2. Methods

### 2.1 Study system, data, and predictors

*[Keep existing text — only minor edits indicated below.]*

**Edit needed in 2.1:** Add the following sentence after the predictor list:

> "Although ALT and BIO1 are strongly correlated (Pearson r = −0.93), we retained both because they capture mechanistically distinct effects on crayfish ecology (topographic position vs thermal regime). We additionally report robustness checks with each predictor removed in turn (Section 3.5)."

---

### 2.2 Modelling framework

#### 2.2.1 Interpretable baselines: decision trees

We trained decision trees at four depths (2, 3, 5, and unlimited) using scikit-learn's `DecisionTreeClassifier` with `min_samples_leaf = 5` and a fixed random seed. The shallow depth-2 baseline retains the original manuscript's hardest constraint on interpretability (≤ 4 terminal rules), while the deeper trees provide stronger predictive benchmarks against which to evaluate the LLM ensembles. Reviewer 2 noted that depth-2 alone may be too constrained to serve as a fair benchmark for SDM applications; the deeper variants address this concern.

#### 2.2.2 Knowledge-driven component: LLM-derived rule ensembles (PER-FOLD)

We used OpenAI's GPT-4o (accessed via the API in March–May 2026) to generate ensembles of shallow decision trees expressing ecological reasoning. **Critically, in this revision the entire generation and selection pipeline is nested inside the cross-validation loop**, so that each fold's LLM ensemble is constructed using only that fold's training data. This eliminates the data leakage identified by Reviewer 2 in the original submission.

The procedure for each of the 5 outer folds is:

1. **Partition.** Split the data into 80% training and 20% test (stratified on presence/absence). The test fold is held out from every subsequent step.
2. **Prompt construction (from training data only).** Compute from the training set:
   - Aggregate quantiles (q05, q20, q40, q50, q60, q80, q95) for each predictor
   - Class-conditional means and IQRs for presence vs absence
   - DT splits from a shallow decision tree fitted on the training fold
   - Class balance (count of presences and absences)
   These are combined with short natural-language ecological priors (e.g., "Austropotamobius torrentium prefers cool montane streams; lower RWQ values favour presence") into a structured prompt. No individual occurrence record, no site coordinates, and no test data are exposed to the LLM.
3. **Candidate generation.** The LLM is queried in 4 batches of 20 trees each at temperatures {0.4, 0.7, 0.9} (cycling), producing 80 candidates per fold. Each tree is constrained to depth ≤ 2 and uses only the four allowed predictors.
4. **Validation and repair.** Each candidate tree is checked for schema compliance (required fields, allowed operators, predictor names, threshold ranges). Trees with minor errors are repaired automatically (e.g., out-of-range thresholds clipped to empirical min–max); irreparable trees are discarded. Exact duplicates are removed.
5. **Scoring.** Each valid candidate is scored on the training data using Macro-F1. Trees scoring below 0.40 are discarded.
6. **Ensemble selection.** Starting from the best-scoring tree, we greedily add candidates that maximally improve the ensemble's 3-fold inner-CV Macro-F1 on the training data, stopping when 5 consecutive additions fail to improve performance or when the ensemble reaches 15 trees.
7. **Evaluation.** The selected ensemble is evaluated exactly once on the held-out test fold. The ensemble prediction for a site is the fraction of trees voting for presence.

All prompts, raw API responses, and selection metadata are archived in `outputs/{SPECIES}/llm_audit_per_fold/`.

Per-fold ensemble sizes varied with ecological complexity: AUT 5–7 trees, ABI 1–4 trees, FXL 1–4 trees. For FXL, several folds produced single-tree ensembles based on altitude alone, reflecting the species' simple distributional pattern.

#### 2.2.3 Hybrid integration

*[Keep existing description of soft-veto, AND-presence, soft-blend, and stacked variants. No methodological change needed.]*

---

### 2.3 White-box and black-box benchmark models

To address Reviewer 3's recommendation for an additional interpretable benchmark, we fitted **generalized linear models (GLM)** as logistic regressions on standardized predictors using scikit-learn's `LogisticRegression` with default L2 regularization and a fixed random seed. The GLM provides a continuous-response interpretable alternative to threshold-based trees.

To quantify the accuracy cost of full transparency, we fitted **random forests (RF)** with 500 trees, no depth limit, minimum 5 samples per leaf, and the square root of the number of features per split. Out-of-bag scores were recorded as an internal consistency check.

To support reviewer concerns about benchmark realism, we additionally evaluated a **null model** that always predicts the majority (absence) class, providing a floor against which all models are compared.

SHAP values (TreeExplainer; Lundberg et al. 2017) were computed for RF to allow direct comparison between the data-driven feature importances and the LLM-derived rules.

---

### 2.4 Evaluation

All models were evaluated using stratified 5-fold cross-validation with a fixed random seed (42). We report:
- **Macro-F1** — primary metric, weights presence and absence classes equally under class imbalance
- **Accuracy** — secondary metric
- **D² (deviance explained)** — added per Reviewer 3's recommendation; computed as 1 − (model log-loss / null log-loss), with ε = 1e-3 clipping for numerical stability, bounded to [-2, 1]
- **Absolute confusion-matrix counts** (TP, FP, FN, TN) aggregated across the 5 folds

Bootstrap 95% confidence intervals (1000 resamples) are reported for the LLM vote fractions in Figure 1.

To assess the contribution of data-informed prompting, we additionally evaluated LLM ensembles generated in a **'pure' mode** where the prompt contained only aggregate predictor statistics and ecological priors but no class-conditional statistics or data-driven decision tree splits. Comparing pure and full LLM ensembles quantifies the added value of grounding the LLM's ecological knowledge in data summaries.

---

### 2.5 Implementation

*[Keep existing description with the following replacements:]*

Replace the GitHub URL line with:

> "All code is available in a public GitHub repository (https://github.com/KristianMiok/HybridSDM), released under an open-source licence. The repository includes every prompt sent to the LLM, every raw response received, validation and repair logs, fold-level CV splits with their seeds, and full generation metadata (model version, temperature, timestamps) — sufficient for byte-for-byte reproducibility of all per-fold ensembles and all reported metrics."

---

### 2.6 Synthetic data validation (NEW)

To address Reviewer 2's concern that the original manuscript relied on a single dataset, and to provide a stronger test of whether the LLM recovers genuine ecological thresholds, we generated three **synthetic species datasets with known ground-truth rules**. Predictor distributions (RWQ, ALT, FFP, BIO1) were drawn from log-normal, gamma, and exponential distributions tuned to match the empirical ranges in the Romanian dataset. Presence labels were then assigned according to species-specific rules:

- **SYN_A:** *presence iff* RWQ < 1.0 (single-rule, balanced; 182/98)
- **SYN_B:** *presence iff* ALT > 400 m (single-rule, balanced; 124/96)
- **SYN_C:** *presence iff* (BIO1 > 10 °C) OR (FFP < 0.3) (disjunctive, imbalanced; 183/137)

For each rule-match, the assigned probability of presence was 0.90; for non-matches, 0.10 — corresponding to roughly 10% noise (analogous to imperfect detection in real surveys). The ecological priors provided to the LLM named the relevant feature(s) for each species but **did not specify the threshold value** — equivalent to expert knowledge that a feature matters, without numerical specification.

The full pipeline (per-fold generation, validation, ensemble selection, CV evaluation) was applied identically to the real and synthetic datasets. For the synthetic data we additionally extracted the **root-split feature and threshold** from each per-fold tree and compared to the true ground-truth rule, providing a direct test of rule recovery accuracy that does not depend on predictive performance.

---

### 2.7 Use of generative AI (NEW)

This study used OpenAI's GPT-4o (accessed via API, March–May 2026) as an integral component of the methodology: GPT-4o was prompted to generate candidate decision-tree rules from aggregate predictor statistics and ecological priors. The LLM did not access any individual occurrence record or test data. All prompts, raw API responses, and selection metadata are archived in the project repository to ensure full reproducibility.

Generative AI tools (ChatGPT, Claude) were also used in a limited assistive role during manuscript preparation — for text editing of grammar and clarity, and for code commenting — under the authors' direct supervision. All scientific claims, ecological interpretations, data analyses, and conclusions are the authors' own work. No content was published without author review and approval.
