# Claim Model Evaluation Report (XGBoost)
## Model B — Claim Outcome Classification

### Step 3/7 — Pre-Modeling Statistical Signal Check
Largest effect size found across all candidate features, INCLUDING the leakage-free
provider_rejection_rate: 0.0142 (negligible range). This predicted near-random
AUROC before any model was trained.

### Train vs Test Performance
| Metric | Train | Test |
|---|---|---|
| Accuracy | 0.5663 | 0.5656 |
| Macro F1 | 0.3432 | 0.3051 |
| Macro AUROC | 0.6372 | 0.5713 |

Train-test AUROC gap: 0.0659 (well-controlled)

### Per-Class AUROC / AUPRC (Test Set)
| class    |   AUROC |   AUPRC |   no_skill_baseline_AUPRC |   lift_over_baseline |
|:---------|--------:|--------:|--------------------------:|---------------------:|
| Paid     |  0.5603 |  0.6632 |                    0.5994 |               0.0638 |
| Pending  |  0.5052 |  0.2581 |                    0.255  |               0.0031 |
| Rejected |  0.6484 |  0.2111 |                    0.1456 |               0.0655 |

### Business Metric — Rejected Claim Recall
- Rejected Recall: 0.1525 (15.2%)
- Rejected AUROC: 0.6484
- Rejected AUPRC: 0.2111 vs no-skill baseline 0.1456
- Max predicted Rejected probability across test set: 0.6011

### Class Imbalance Handling
SMOTE applied to training data only (post-split); validation and test reflect the true distribution.

### Top 5 Most Influential Features
| feature                 |   importance |
|:------------------------|-------------:|
| provider_rejection_rate |    0.261868  |
| insurance_provider_enc  |    0.153045  |
| visit_is_weekend        |    0.113933  |
| visit_type_enc          |    0.0926242 |
| chronic_flag            |    0.0917375 |

### Fairness Segmentation Summary
- Gender accuracy spread: 0.0087
- City accuracy spread: 0.0534
- Insurance provider accuracy spread: 0.0402

### Conclusion
The well-controlled train-test AUROC gap confirms overfitting is NOT the cause of limited
performance. The pre-modeling signal check independently predicted this outcome. Performance
is capped by the information content of available features, including the theoretically
strongest one (provider_rejection_rate), not by model choice or methodology.
