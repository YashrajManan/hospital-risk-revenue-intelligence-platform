# Model Card — Visit Risk Classifier (XGBoost)

## Model Details
- **Model type:** XGBoost Classifier (multi:softprob), with early stopping and L1/L2 regularization
- **Training rounds used:** 38 (early-stopped from a maximum of 500)
- **Target variable:** risk_score (Low / Medium / High)
- **Training data:** 17,000 visits | Validation: 3,000 visits | Test: 5,000 visits
- **Split method:** Time-based, three-way (train / validation / test), strictly chronological

## Intended Use
Decision-support signal for staffing, triage prioritization, and resource allocation.
NOT a substitute for clinical judgment and NOT suitable as a sole determinant of patient care.

## Performance
- Test Accuracy: 0.4956
- Test Macro AUROC: 0.5338
- Test Macro F1: 0.2269
- High Risk Recall: 0.0000
- High Risk AUROC: 0.5089
- High Risk AUPRC: 0.2154 (no-skill baseline: 0.2046)

## Overfitting Assessment
Train-test AUROC gap: 0.0736. This is well-controlled, confirming the model is not memorizing
training data. Early stopping engaged at round 38, and the training curve shows
validation loss tracking training loss closely rather than diverging.

## Fairness
Accuracy spread across gender: 0.0201
Accuracy spread across city: 0.0319
Accuracy spread across insurance provider: 0.0520

## Limitations
1. **Data-level signal ceiling, proven statistically before modeling:** A formal chi-square/ANOVA
   signal check (Step 0) found the largest effect size across ALL candidate features to be
   0.0197 — negligible by standard benchmarks (Cramer's V and eta-squared interpretation
   guides). This independently predicted near-random AUROC before any model was trained, and the
   trained model's actual test AUROC (0.5338) matches that prediction closely.
2. **This is NOT an overfitting problem.** The train-test AUROC gap of 0.0736 is small,
   and the training curve confirms early stopping worked as intended. Overfitting and "no signal"
   are different problems with different fixes; this model has the second one, which model
   architecture changes cannot resolve.
3. **High Risk recall of 0.0% at the default threshold.** The model's predicted
   probability for High never exceeds roughly 0.27 for any test visit, so it is always
   outranked by Low or Medium. This reflects an absence of class-separating signal rather
   than a fixable threshold problem; AUPRC for High (0.2154) sitting
   close to its no-skill baseline confirms this independently of any single threshold choice.
4. **Synthetic data:** Real hospital data with genuine clinical signals (vitals, diagnosis codes,
   prior admission history) would very likely show measurably larger effect sizes in a Step-0-style
   check, and a correspondingly more useful model — without changing this pipeline's structure.

## Recommendation
Treat as a decision-support signal only, reviewed alongside clinical judgment. Before production
deployment: (a) retrain on real clinical data with richer features, and (b) re-run the Step 0
signal check on that data first — if effect sizes remain negligible, no amount of further model
tuning will resolve it, and the data collection itself needs revisiting.

## Assumptions
- risk_score reflects ground-truth risk as clinical staff would assign it
- Visits are treated as independent records (no explicit readmission-chain modeling)
