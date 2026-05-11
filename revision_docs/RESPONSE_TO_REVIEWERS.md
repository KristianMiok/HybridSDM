# Response to Reviewers — ECOINF-D-26-01145

**Manuscript:** Interpretable by design: language model-derived ecological rules for species distribution modelling

**Authors:** Miok, Škrlj, Robnik-Šikonja, Pârvulescu

---

## Editor's overall comment

> All reviewers agree that the topic of the analysis is worthy of publication. However, they raise a number of issues that would need to be addressed... There are a few methodological problems that have been identified that may lead to bias or self-perpetuating conclusions. Additional analysis or simulations would help strengthen inferences.

**Response:** We thank the editor and reviewers for their constructive critiques. We have substantially restructured the paper to address all major concerns. The most important changes are:

1. **Per-fold LLM tree generation** — eliminates the mild data leakage Reviewer 2 identified
2. **GLM baseline added** — requested by all three reviewers
3. **Deeper decision tree baselines** (depths 3, 5, unlimited) — addresses Reviewer 2's "weak benchmark" objection
4. **Synthetic dataset with known ground-truth rules** — addresses Reviewer 2's "single dataset" and "rule recovery" concerns; Reviewer 3's "simulated data" recommendation
5. **Honest reframing of results** — LLM is no longer claimed to "outperform" black-box; it is "competitive" overall and "best" only for the data-poor species
6. **D² metric, null model, confusion matrices, absolute counts** — all requested by Reviewer 3
7. **GenAI declaration** — added per Reviewer 2
8. **Robustness to predictor removal** (drop ALT, drop BIO1) — requested by Reviewers 2 and 3
9. **Improved metadata and reproducibility** in the GitHub repository

The revised paper now contains two complementary contributions: (1) a methodological validation via synthetic rule recovery, and (2) an honest performance comparison on real species.

---

## Reviewer #1 (Falk Huettmann)

### General philosophical framing

> [Authors] got lost in 'understanding' or that it would help us... Authors offer here their own inference and inference scenario, but hardly more. It's subjective, hardly so meaningful or generalizable then.

**Response:** We thank Dr. Huettmann for his thoughtful comments on the broader epistemology of interpretation in machine learning. We respect his position — drawn from Breiman's "two cultures" tradition — that inference should derive from predictions rather than from the inner structure of black-box models. We agree that no parsimonious explanation captures all dimensions of complex ecological systems, and we have substantially softened our claims throughout.

However, we believe the goal of producing **inherently interpretable** models (Rudin 2019) is distinct from the post-hoc interpretation of black-box models that Breiman cautioned against. The rules our framework generates are not approximations of an underlying random forest — they are the model itself. They make explicit, falsifiable claims about ecological thresholds that can be tested against new field data.

We have:
- Softened the abstract and discussion, removing assertive language like "no black box needed"
- Added explicit acknowledgment that any rule set is one approximation among many possible
- Cited Fernández-Delgado et al. (2014) and Humphries et al. (2018) as recommended
- Added a synthetic-data validation that demonstrates the rules are not arbitrary — they recover known ground-truth thresholds

### Specific request: confidence intervals on Figure 2

> Figure 2: I assume it should show 95% confidence intervals.

**Response:** Figure 2 (now Figure 1 in the revision) has been updated with bootstrap 95% confidence intervals on the boxplot whiskers.

### Specific request: FAIR / ISO metadata

> Authors should make this more clear though, specifically the repeatable runs of LLMs, e.g. with time stamps.

**Response:** The repository now includes for every LLM call:
- The exact prompt text (saved as `batch_N_prompt.txt`)
- The raw API response
- Timestamp, model version (gpt-4o), temperature, and selection metadata
- Per-fold seed information

These are stored in `outputs/{SPECIES}/llm_audit_per_fold/fold_{N}/`. We have added a FAIR-compliance section to the README and a Data Availability statement noting all artifacts are versioned in the GitHub repository.

### Citations added

- Fernández-Delgado, M., et al. (2014). Do we need hundreds of classifiers to solve real world classification problems? *Journal of Machine Learning Research*, 15(1), 3133-3181.
- Humphries, G. R. W., et al. (2018). Machine Learning for Ecology and Sustainable Natural Resource Management. Springer.
- Rudin, C. (2019). Stop explaining black box machine learning models for high stakes decisions and use interpretable models instead. *Nature Machine Intelligence*, 1, 206-215. [already cited]

---

## Reviewer #2 (most extensive critique)

### Main concern: definition of "interpretability" is too narrow

> 'interpretability' is too narrowly defined, and equating interpretability with simple threshold rules

**Response:** We agree this needed clarification. We have added a paragraph in the Introduction explicitly defining what we mean by "interpretable" in this work — namely, models whose predictions can be traced to a small set of **explicit threshold rules** that practitioners can verify, communicate, and challenge with new data. We explicitly note that GLMs and GAMs are also legitimately interpretable in a different sense (continuous response shapes), and we now include GLM as a baseline.

The rule-based form we adopt is one specific kind of interpretability that is particularly suited to **conservation decision-making**, where threshold-based reasoning is common in management protocols and legal definitions of critical habitat.

### Critical concern: data leakage / circularity

> The LLM appears non-independent from the decision tree comparison model, so the performance comparison is spurious... How can you integrate two results, when the second result already relies on the first result?

**Response:** This was a genuine methodological weakness in the original submission, and we are grateful to the reviewer for identifying it. We have completely redesigned the LLM tree generation procedure to be **per-fold**:

- For each of the 5 cross-validation folds, we now generate a fresh ensemble of LLM trees using **only that fold's training data**
- Class-conditional statistics, DT splits, and any data summaries shown to the LLM are computed from training data alone
- The test fold is held out from every stage of generation and selection

This eliminates the leakage. The honest cost is that LLM Macro-F1 dropped from 0.77/0.79/0.95 (real data, original) to 0.71/0.69/0.93 (per-fold). The data-poor ABI case remains the strongest result — the LLM ensemble still outperforms RF — but we no longer claim the LLM consistently outperforms black-box approaches. The revised abstract and Section 3.1 reflect this honestly.

We have also added a synthetic-data validation (Section 3.2) that does not depend on real-data performance and demonstrates the LLM recovers true ecological thresholds.

### Critical concern: weak benchmark (DT depth=2)

> The decision tree of depth two benchmark is weak and very constrained — it makes the LLM look good in comparison to a realistic SDM.

**Response:** We have added decision tree baselines at depths 3, 5, and unlimited. These often perform as well as or better than the LLM ensemble:

| Species | LLM | DT(d=2) | DT(d=3) | DT(d=5) |
|---|---|---|---|---|
| AUT | 0.708 | 0.752 | 0.770 | **0.771** |
| ABI | **0.689** | 0.682 | 0.613 | 0.650 |
| FXL | 0.927 | **0.938** | 0.932 | 0.936 |

The revised manuscript now states clearly that deeper decision trees are often the best interpretable choice for well-studied species; the LLM's advantage appears specifically under data scarcity (ABI).

### Critical concern: single dataset

> This model is demonstrated on one data set which may have quirks that influence the results. To demonstrate a novel modelling approach like this, I'd expect both a simulated test set and real test data set.

**Response:** We have added **three synthetic species** with known ground-truth rules:
- SYN_A: presence iff RWQ < 1.0 (single-rule, balanced)
- SYN_B: presence iff ALT > 400 (single-rule, balanced)
- SYN_C: presence iff (BIO1 > 10) OR (FFP < 0.3) (disjunctive, imbalanced)

Critically, this lets us evaluate **whether the LLM recovers the true rules**, not just whether it predicts well. Results (Section 3.2 of the revised paper):

- For SYN_A, all 5 per-fold LLM ensembles recover RWQ ≤ 0.99 (Δ from true threshold: 0.01)
- For SYN_B, all 5 ensembles recover ALT ≈ 400.3 (Δ < 5 from true 400)
- For SYN_C, all 5 ensembles recover both branches of the disjunction

We did not have access to a second real SDM dataset with comparable predictors and conservation context for this revision, but the synthetic experiment provides a more rigorous test of rule recovery than any single additional real dataset would have.

### Concern: two highly correlated predictors retained

> Two variables with extremely high correlation are retained... could make coefficient estimates unstable, confuse variable importance, and reduce model transferability.

**Response:** We have added two robustness checks (Section 3.5 of the revised paper): re-running every model with ALT removed, then with BIO1 removed. Results:
- DT and RF degrade gracefully (4–6% drop)
- LLM ensembles degrade more (especially FXL, where the per-fold ensemble was a single ALT-based tree)
- GLM is least affected (relies on correlations rather than thresholds)

We added an explicit limitation noting that LLM ensembles built around dominant predictors are brittle if those predictors are unavailable at inference time. We retained ALT and BIO1 in the main analysis because they capture distinct mechanisms (topographic vs thermal), but we now report the no-ALT and no-BIO1 robustness checks transparently.

### Concern: post-hoc XAI mentioned twice in Introduction

> SHAP is mentioned, then mentioned again as if for the first time. This is awkward.

**Response:** Fixed. The two redundant mentions of SHAP/LIME in the Introduction have been consolidated into one paragraph.

### Concern: prompt construction "answers" the model

> "we provided the split rules from a shallow decision tree trained on the same data as a reference for empirically supported thresholds." This sounds like you gave the model the answers.

**Response:** We have separated the contribution of "data-informed" vs "priors-only" prompts via ablation. In the priors-only condition (no class-conditional stats, no DT splits), the LLM relies only on aggregate quantiles and ecological priors. Section 4.5 of the revised paper compares both modes and quantifies the contribution of each information source. Per-fold generation also means the DT splits provided to the LLM in fold k were computed on fold k's training data, never on test data.

### Concern: lack of uncertainty estimation

> method limitations such as uncertainty estimation are not mentioned

**Response:** We added an explicit limitation about uncertainty (Section 4.5) and now report:
- D² (deviance explained) alongside Macro-F1
- Bootstrap 95% CIs on Figure 1
- Confusion matrices (SI)
- A note that the ensemble vote fraction is a coarse uncertainty estimate and that proper calibration is future work

### Concern: GenAI declaration missing

> 'declaration of generative AI use' is requested by this publisher but missing

**Response:** Added (new Methods Section 2.7):

> *"This study used OpenAI's GPT-4o (accessed via API, March–May 2026) as an integral component of the methodology: GPT-4o was prompted to generate candidate decision-tree rules from aggregate predictor statistics and ecological priors. The LLM did not access any individual occurrence record or test data. All prompts, raw API responses, and selection metadata are archived in the project repository to ensure full reproducibility. Generative AI tools were also used in a limited assistive role during manuscript preparation (text editing, code commenting) under the authors' supervision; all scientific claims, analyses, and conclusions are the authors' own."*

### Concern: SHAP shows same ecological drivers — but the LLM saw the DT splits?

> But you said that the LLM sees the DT splits?

**Response:** In the revised per-fold pipeline, the LLM only sees the DT splits computed on the training fold's data — these are observable to any human ecologist with access to the same training data. The convergence between SHAP-derived RF feature importance, DT splits, and LLM rules in Section 3.4 reflects that they all detect the same dominant ecological gradients (RWQ, ALT), not that the LLM was given the answer. We have clarified this in the revised Section 3.4.

### Concern: rule generation "without seeing the data"

> [Line 184] How can the model know which are the top performing trees without knowledge of the occurrence data? Presumably local assessment is done between rounds

**Response:** Clarified in Methods 2.2.2: the LLM generates candidate trees from aggregate stats and priors, but the **selection** of which trees enter the final ensemble uses training-data F1 scores. We now describe the pipeline explicitly as having two stages: generation (priors + aggregates only) and selection (data-driven). This is analogous to hyperparameter tuning in standard ML — the form of the rule is hypothesized from priors, and empirical performance decides whether to include it. The LLM still never sees individual records.

---

## Reviewer #3 (most constructive)

### Add a GLM baseline

> I would strongly recommend adding a "white box" approach as a second benchmark (e.g. a glm)

**Response:** Done. GLM (logistic regression with standardized predictors) is now in all results tables. Honest finding: GLM is the weakest of the interpretable models for all three real species (Macro-F1 of 0.60, 0.60, 0.92) but is competitive on synthetic SYN_A and SYN_B. We now report this honestly.

### Drop ALT and check robustness

> As a minimal compromise I would recommend running the whole workflow with removing elevation as predictor and comparing how robust the results are.

**Response:** Done — see Section 3.5 of the revised paper. We also ran the dual check (drop BIO1) for completeness. Results show LLM ensembles degrade more than DT or RF when ALT is removed, particularly for FXL where the per-fold ensemble depended heavily on ALT. We acknowledge this as a limitation and discuss it in Section 4.5.

### Confusion matrices in SI

> I would suggest to provide the resulting confusion matrices as supporting information.

**Response:** Done. All confusion matrices are in `outputs/{SPECIES}/cv_extended_perfold.csv` and `cv_extended_summary.csv` with per-fold and aggregated TP/FP/FN/TN counts. A formatted SI table is included as `revision_docs/SI_Table_confusion_matrices.csv`.

### Hold-out test set

> I would recommend a test where you keep some hold-out data that the whole workflow has never seen

**Response:** Per-fold tree generation now achieves this within the cross-validation structure — for each fold, the test data is held out from prompt construction, tree generation, ensemble selection, and DT training. Each of the 5 test folds serves as a held-out test set for that fold's generated trees. We agree this is not identical to a single canonical test set, but it is methodologically stronger because every site is tested exactly once on an LLM ensemble it never contributed to.

### D² metric

> In addition to the F1 macro score I would include a metric that provides a quantitative measure of the distance between prediction and observation (standardized deviance or cross-entropy, D2)

**Response:** D² is now reported for all models. We note that LLM ensembles produce coarse vote-fraction probabilities (n/N) and therefore have poorer D² than DT or RF, despite competitive Macro-F1. This is now flagged as a limitation in Section 4.5.

### Null model

> compare model prediction against a null model that directly uses prevalence as a prediction

**Response:** Null model (predict majority class) added to all results tables. Macro-F1 is 0.40 / 0.46 / 0.42 for the three species — all real models substantially exceed this.

### Translate F1 gains to absolute counts

> translating the percentage performance gain in F1 score to absolute numbers of presence points and absence points that were correctly predicted

**Response:** Done in Section 3.1 and in absolute-count tables. For example: for ABI, the LLM ensemble correctly classifies 22 of 34 presences across the 5 folds, compared to 12 of 34 for RF — an additional **10 presences correctly identified** by switching from RF to LLM, with comparable absence accuracy.

### Univariate scatter plots / partial dependence plots

> I would strongly recommend a more transparent visualization of data

**Response:** Univariate boxplots of each predictor by presence/absence are now in SI Figure S1. Partial dependence plots for the RF model are in SI Figure S2.

### Clarify CV nesting

> Some aspects of the modelling procedure were not 100% clear

**Response:** Methods 2.2.2 has been substantially rewritten to make the pipeline explicit:
1. For each of 5 outer folds, partition data into training (4/5) and test (1/5)
2. From training data only, compute aggregate stats, class-conditional stats, and DT splits
3. Build a prompt and have the LLM generate 80 candidate trees
4. Score each candidate on the training data; keep candidates with F1 ≥ 0.4
5. Greedy forward selection (using 3-fold inner CV on training data) builds the ensemble
6. The final ensemble is evaluated **once** on the held-out test fold

The test fold contributes nothing to generation, scoring, or selection.

### Number of selected trees varies

> I understood that the resulting model is (in this application case) an ensemble of 1-9 decision trees... if it was only 1 tree, would this be covered in the shallow DT?

**Response:** Yes, single-tree ensembles do occur (especially for FXL, where one altitude split captures most of the signal). When this happens, the LLM ensemble is functionally equivalent to a single shallow tree — but a *different* one from the data-driven DT. We have added an explanation and a worked example for one such case. Per-fold ensemble sizes are reported in SI Table S2.

### Other minor concerns

> Line 138/139: what does it mean, ambiguous or contradictory records?

**Response:** Clarified — these are sites with conflicting presence/absence labels from different surveys. We retained only sites with unambiguous status from Satmari et al. (2023).

> Line 169: I am not familiar with the term "safe summary statistics"

**Response:** Replaced with "aggregate summary statistics" throughout. We meant statistics that do not expose individual records — quantiles and means, not raw values.

> Line 173: define the exact meaning of "soft priors"

**Response:** Defined in Methods 2.2.2: "soft ecological priors" are short natural-language statements describing each species' general habitat preferences (e.g., "prefers cool montane streams"), without numerical thresholds. They guide the LLM toward biologically plausible rules but do not specify the answer.

> L 196: Why did you choose a minimum performance threshold of 0.4?

**Response:** The Macro-F1 = 0.4 floor was chosen as slightly above the prevalence-based null baseline (0.40–0.46 across species) to discard trees that essentially predict a single class. We have noted this rationale in the revised Methods.

> L 242: with these low number of data points doing 5-fold cross-validation

**Response:** We agree this is tight for ABI. We retained 5-fold CV for comparability with the original submission but note that bootstrap CIs (now reported) capture the additional uncertainty.

---

## Summary of all changes

| Change | Section affected | Status |
|---|---|---|
| Per-fold tree generation | Methods 2.2.2, Results 3.1, all tables | Done |
| GLM baseline | Methods 2.3, all results tables | Done |
| Deeper DT baselines (d=3, 5, unlim) | Methods 2.2.1, Results | Done |
| Synthetic dataset + rule recovery | NEW Methods 2.6, NEW Results 3.2 | Done |
| Drop-ALT robustness | NEW Section 3.5 | Done |
| Drop-BIO1 robustness | Section 3.5 | Done |
| D² metric | Methods 2.4, Results | Done |
| Null model | Methods 2.4, Results | Done |
| Confusion matrices | SI | Done |
| Absolute counts in text | Section 3.1 | Done |
| GenAI declaration | NEW Methods 2.7 | Done |
| Fernández-Delgado, Humphries citations | Intro, Discussion | Done |
| Definition of "interpretability" | Intro | Done |
| Bootstrap 95% CIs | Figure 1 | Done |
| FAIR metadata + reproducibility | Data Availability, README | Done |
| Softened "no black box needed" framing | Throughout | Done |
| Univariate predictor plots | SI Figure S1 | Done |
| RF partial dependence plots | SI Figure S2 | Done |
