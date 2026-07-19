# Governance and Compliance Document
## Hospital Operations & Revenue Risk Intelligence Platform
## Phase 6 — Monitoring, Drift Detection, and Governance

**Document version:** 1.0
**Generated:** 2026-07-02 18:21 UTC
**Covers:** Visit Risk Classification Model (Model A) and Claim Outcome Classification Model (Model B)

---

## 1. System Overview

This AI system consists of two XGBoost classification models deployed via a
FastAPI service on Hugging Face Spaces (Docker container, Python 3.12). The
system predicts:
- **Model A:** Visit Risk Level (Low / Medium / High) to support clinical
  triage and staffing decisions
- **Model B:** Insurance Claim Outcome (Paid / Pending / Rejected) to support
  proactive billing review before submission

**Live API:** https://yashrajy-hospital-risk-claim-api.hf.space
**Source code:** https://huggingface.co/spaces/Yashrajy/hospital-risk-claim-api

---

## 2. Known Limitations

### 2.1 Weak Predictive Signal (Confirmed by Pre-Modeling Statistical Test)
Before any model was trained, a formal statistical signal check (chi-square /
Cramer's V for categorical features, ANOVA / eta-squared for numeric features)
found that BOTH target variables (risk_score and claim_status) show near-zero
statistical association with all available features in this dataset:
- Largest effect size found across all features: ~0.02 (negligible range)
- This independently predicted near-random model performance before training

**Observed test-set performance:**
- Model A (risk): Macro AUROC = 0.53 (barely above 0.50 random baseline)
- Model B (claim): Macro AUROC = 0.57; Rejected-class AUROC = 0.65

This is a **data-level limitation**, not a modeling or code deficiency. The
pipeline (leakage-free features, time-based splits, proper imbalance handling,
early stopping) is built correctly and would benefit substantially from real
clinical data with genuine feature-target associations.

### 2.2 High Risk Recall = 0% at Default Threshold (Model A)
At the default classification threshold, Model A's predicted probability for
"High" never exceeds 0.27 for any record. This means it never predicts "High"
at the default threshold. This reflects absent signal, not a fixable threshold
issue — the AUPRC for High sits barely above the no-skill baseline.

### 2.3 Synthetic Dataset
Both models were trained on a fully synthetic dataset generated for this
capstone project. Real hospital data with genuine clinical signals (vitals,
diagnosis codes, prior admissions, lab results, specific ICD-10 codes) would
very likely show measurably larger effect sizes and produce a substantially
more useful model.

### 2.4 Single-Record Inference Only
The current API accepts one record per request. Batch endpoints for
high-volume scenarios are not implemented. For processing thousands of records
simultaneously, the models should be called programmatically in parallel or a
batch endpoint should be added to main.py.

---

## 3. Assumptions

1. `risk_score` in the training data reflects the ground-truth risk level a
   clinical professional would assign — i.e. it is not a derived or proxy label
2. `claim_status` reflects the final, true adjudication outcome for each claim
3. Each visit record is independent — no explicit modeling of readmission chains
   or sequential patient encounters
4. `registration_date` is always on or before `visit_date` in real data (the
   synthetic dataset violates this for ~48.6% of records — documented in Phase 2
   and handled by clipping `days_since_registration` at 0)
5. Insurance providers in production will always be one of the four trained
   categories (CareOne, HealthPlus, MediCareX, SecureLife). A new insurer not
   seen during training would have its label-encoded value set to -1 and its
   rejection rate set to the training-period average — both are documented
   fallback behaviors, not errors.
6. `billed_amount_is_outlier` is derived from IQR bounds computed on the TRAINING
   set (lower: -13,640.97, upper: 53,621.49). These bounds are hardcoded in the
   API and will need updating if the billed amount distribution shifts significantly.

---

## 4. Monitoring Results Summary (Phase 6 Analysis)

### 4.1 Data Validation
- Validation checks run on 5,000 test-period records
- Checks covering: 31 fields across missing values, numeric ranges, and unseen categories
- Validation failures found in: 0 checks
- See `drift_summary.csv` for full results

### 4.2 Feature Drift
- Features monitored: 15
- Features with detected drift: 2
- Drift detection method: KS test (numeric features, threshold: p < 0.05 AND KS > 0.10)
  and Chi-square test (categorical features, threshold: p < 0.05)
- See `drift_summary.csv` for per-feature results

### 4.3 Prediction Drift
- Both models' prediction distributions were compared across two equal time
  windows within the test period
- Large instability in prediction distributions would indicate the model is
  behaving inconsistently and requires investigation

### 4.4 Audit Log
- 10,000 prediction records logged in `monitoring_audit_log.csv`
- Fields: timestamp, model_name, model_version, input_feature_hash, prediction, visit_id
- Input features are hashed (SHA-256, first 16 characters) to protect patient privacy

---

## 5. Retraining Strategy

### When to retrain:
Retraining should be triggered by ANY of the following conditions:

| Trigger | Threshold | Action |
|---|---|---|
| Feature drift detected | KS statistic > 0.10 AND p < 0.05 for any key feature | Investigate cause, retrain if structural change confirmed |
| Prediction drift detected | >5 percentage point shift in any class fraction between windows | Investigate immediately; retrain if shift persists |
| Validation failure rate | >1% of incoming records fail any validation check | Update validation rules or retrain on data including new categories |
| Business performance | Rejected-claim recall drops below 10% (from current 15.25%) | Retrain with new data |
| New data available | >6 months of new real operational data collected | Schedule retraining run |

### Retraining procedure:
1. Run the Phase 3 signal check (chi-square/ANOVA) on the new data first.
   If effect sizes remain negligible, retraining will not help — the data
   collection process itself needs revisiting.
2. If signal is present, run `build_features.py` (Phase 2) on the new data.
3. Re-run `02_risk_model.ipynb` and `03_claim_model.ipynb` (Phase 3/4) on
   the updated feature table.
4. Evaluate both models against the Phase 4 criteria (AUROC, AUPRC, recall
   for the critical class, fairness segmentation).
5. Update `MODEL_VERSION` in `app/main.py` (e.g. `xgboost-v2-phase3`).
6. Rebuild and redeploy the Docker image on Hugging Face Spaces.
7. Monitor the new model's drift metrics for at least two weeks before
   decommissioning the previous version.

### Who is responsible for retraining:
The data science team (model developers) is responsible for triggering and
executing retraining. The hospital IT/operations team is responsible for
flagging data quality issues and notifying when new data volumes are sufficient.
A retraining decision should not be made by the API or the monitoring system
automatically — human review of the trigger conditions is required before
any model replacement.

---

## 6. Compliance Notes

- **No real patient data** was used in training or evaluation. All data is
  synthetic and generated for this educational capstone project.
- **Privacy:** Patient-identifiable information is not stored in prediction
  logs. Input features are represented by a SHA-256 hash only.
- **Transparency:** Every API response includes an explicit `warning` field
  disclosing the model's known reliability limitations, so no user of the
  API can be unaware of them.
- **Auditability:** All predictions are logged with timestamp, model version,
  and input hash, enabling post-hoc review of any prediction.
- **Human oversight required:** Both models are explicitly designed as
  decision-SUPPORT tools. Neither is intended to make autonomous clinical
  or financial decisions without human review.
