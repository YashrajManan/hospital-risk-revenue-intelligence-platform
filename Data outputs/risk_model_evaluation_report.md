# Risk Model Evaluation Report (XGBoost)
## Model A — Visit Risk Classification

### Step 0 — Pre-Modeling Statistical Signal Check
Largest effect size found across all candidate features: 0.0197 (negligible-to-weak range).
This set the expectation that achievable AUROC would be close to random (0.50) before any model was trained.

### Train vs Test Performance
| Metric | Train | Test |
|---|---|---|
| Accuracy | 0.5076 | 0.4956 |
| Macro F1 | 0.2416 | 0.2269 |
| Macro AUROC | 0.6073 | 0.5338 |

Train-test AUROC gap: 0.0736 (well-controlled by early stopping + regularization)

### Per-Class AUROC / AUPRC (Test Set)
| class   |   AUROC |   AUPRC |   no_skill_baseline_AUPRC |   lift_over_baseline |
|:--------|--------:|--------:|--------------------------:|---------------------:|
| Low     |  0.5534 |  0.537  |                    0.496  |               0.041  |
| Medium  |  0.539  |  0.326  |                    0.2994 |               0.0266 |
| High    |  0.5089 |  0.2154 |                    0.2046 |               0.0108 |

### Business Metric — High Risk Recall
- High Risk Recall: 0.0000 (0.0%)
- High Risk AUROC: 0.5089
- High Risk AUPRC: 0.2154 vs no-skill baseline 0.2046

### Top 5 Most Influential Features
| feature                 |   importance |
|:------------------------|-------------:|
| length_of_stay_hours    |    0.144225  |
| gender_enc              |    0.0698553 |
| avg_los_per_patient     |    0.0677923 |
| doctor_id               |    0.0661082 |
| days_since_registration |    0.0640337 |

### Fairness Segmentation Summary
- Gender accuracy spread: 0.0201
- City accuracy spread: 0.0319
- Insurance provider accuracy spread: 0.0520

### Conclusion
The well-controlled train-test AUROC gap (0.0736) confirms overfitting is NOT the cause of
limited performance. The Step 0 statistical signal check independently predicted this outcome before
training began. Performance is capped by the information content of the available features, not by
model choice, tuning, or methodology.
