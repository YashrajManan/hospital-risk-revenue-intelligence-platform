# Model Card — Claim Outcome Classifier (XGBoost)

## Model Details
- **Model type:** XGBoost Classifier (multi:softprob), early stopping, L1/L2 regularization
- **Training rounds used:** 152 (early-stopped from a maximum of 500)
- **Target variable:** claim_status (Paid / Pending / Rejected)
- **Class imbalance handling:** SMOTE (training data only, post-split)
- **Training data:** 17,000 claims | Validation: 3,000 | Test: 5,000
- **Split method:** Time-based, three-way (train / validation / test), strictly chronological

## Intended Use
Secondary screening signal to flag claims likely to be Rejected or left Pending BEFORE
submission, for human billing-team review. NOT intended to auto-reject or auto-approve
claims without review.

## Performance
- Test Accuracy: 0.5656
- Test Macro AUROC: 0.5713
- Test Macro F1: 0.3051
- Rejected Recall: 0.1525
- Rejected AUROC: 0.6484
- Rejected AUPRC: 0.2111 (no-skill baseline: 0.1456)

## Overfitting Assessment
Train-test AUROC gap: 0.0659. Well-controlled; early stopping engaged at round
152, with validation loss tracking training loss rather than diverging.

## Fairness
Accuracy spread across gender: 0.0087
Accuracy spread across city: 0.0534
Accuracy spread across insurance provider: 0.0402

## Limitations
1. **Data-level signal ceiling, proven statistically before modeling.** The largest effect size
   found across ALL candidate features — INCLUDING the correctly leakage-fixed
   provider_rejection_rate — was 0.0142 (negligible). This independently predicted
   near-random AUROC, and the trained model's actual test AUROC (0.5713) matches.
2. **This is NOT an overfitting problem.** Train-test AUROC gap of 0.0659 is small;
   the leakage fix and SMOTE were both correctly scoped to avoid contaminating evaluation.
3. **Rejected recall of 15.2%.** The model's predicted probability for Rejected
   never exceeds 0.60 for any test claim, so it is consistently outranked —
   this reflects absent class-separating signal, not a fixable threshold choice.
4. **Synthetic data:** Real claims data with procedure codes, documentation-completeness flags,
   and specific denial reason codes would very likely show measurably larger effect sizes.

## Recommendation
Use as a secondary screening signal alongside existing billing review, not as an automated
gatekeeper. Before production use: (a) retrain on real claims data with richer features, and
(b) re-run the signal check on that data — if effect sizes remain negligible, further model
tuning will not resolve it; the feature set itself needs revisiting.

## Assumptions
- claim_status reflects the final, true outcome of each claim as adjudicated by the insurer
- Each billing record is independent (no explicit claim-resubmission-chain modeling)
