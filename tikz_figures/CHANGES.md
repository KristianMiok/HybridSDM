# Figure changes from original submission to revision

All figures now use **per-fold ensemble** data (each fold uses its own LLM trees, eliminating data leakage).

## Figure 1 (was Figure 2): Explanation cards — `fig2_explanation_cards_AUT.tex`
- Updated to use new per-fold ensembles
- Card 1: site 112546 (TP, vote 0.80 — was 0.78 with global ensemble)
- Card 2: site 120154 (FP, vote 0.00 — LLM now even more strongly rejects)
- Card 3: site 125941 (moderate TP, vote 0.60 — replaces site 174044)

## Figure 2 (was Figure 3): Boxplot vote vs DT — `fig3_boxplot_AUT.tex`
- DT=1 median dropped from 0.89 to 0.71 (per-fold ensembles are smaller/noisier)
- DT=0 q75 rose slightly from 0.00 to 0.14
- Sample sizes unchanged (n=117, n=156)

## Figure 3A (was Figure 4A): DT split frequency — `fig4a_dt_splits_AUT.tex`
- RWQ ≤ 1.07 now appears in 4 folds (was 5 in global)
- ALT > 118.78 in 3 folds
- Other splits unchanged
- Per-fold ensemble produces same stability pattern: RWQ and ALT dominant

## Figure 3B (was Figure 4B): LLM rule activation — `fig4b_llm_rules_AUT.tex`
- COMPLETELY DIFFERENT structure now
- Per-fold ensembles have only 31 total trees vs hundreds in global ensemble
- Each rule appears 1-2 times (was 90-170 times in global)
- Top rule "RWQ ≤ 1.07 → 1" still leads but with frequency 2 instead of 166
- Rules color-coded (green = presence, red = absence)
- Demonstrates that despite smaller ensemble, dominant ecological patterns recur across folds

## Figure 4 (was Figure 5): RF Gini importance — `fig5_rf_importance_all.tex`
- UNCHANGED — RF importances do not depend on LLM pipeline
- Three species comparison: RWQ dominant for AUT, balanced for ABI, ALT dominant for FXL
