# Sample API Requests and Responses

All examples below were captured from a live run of this exact API
against the real trained model artifacts — these are actual outputs,
not illustrative placeholders.

---

## GET /health

**Request:**
```bash
curl http://localhost:8000/health
```

**Response (200 OK):**
```json
{
  "status": "ok",
  "risk_model_loaded": true,
  "claim_model_loaded": true
}
```

If either model failed to load, the corresponding field would be `false`,
though in practice the application is designed to fail at container
startup (see `deployment_runbook.md`, Step 2) rather than start
successfully with a model missing.

---

## POST /predict/risk

**Request:**
```bash
curl -X POST http://localhost:8000/predict/risk \
  -H "Content-Type: application/json" \
  -d '{
    "department": "ICU",
    "visit_type": "ICU",
    "city": "Delhi",
    "gender": "M",
    "age": 65,
    "chronic_flag": 1,
    "length_of_stay_hours": 48.0,
    "doctor_id": 101,
    "visit_frequency": 3,
    "avg_los_per_patient": 40.0,
    "registration_date": "2025-01-15",
    "visit_date": "2025-08-01"
  }'
```

**Response (200 OK):**
```json
{
  "predicted_risk": "Low",
  "probability_low": 0.4521,
  "probability_medium": 0.3509,
  "probability_high": 0.1971,
  "model_version": "xgboost-v1-phase3",
  "warning": "This model's test-set performance (Macro AUROC 0.53) is barely above random. A formal statistical signal check found negligible association between risk_score and all available features. Treat this prediction as decision-support only, not a reliable risk assessment. See risk_model_card.md for details."
}
```

**Note on this specific example:** despite the input describing a
65-year-old chronic-condition ICU patient with a 48-hour stay — exactly
the profile one might expect to be flagged High Risk — the model predicts
"Low" with the highest probability. This is the real, observed behavior
of this model, consistent with the AUROC ≈ 0.53 finding: it does not
reliably distinguish risk levels based on clinical severity indicators in
this dataset. This is exactly why the `warning` field exists.

---

## POST /predict/claim

**Request:**
```bash
curl -X POST http://localhost:8000/predict/claim \
  -H "Content-Type: application/json" \
  -d '{
    "insurance_provider": "SecureLife",
    "department": "Cardiology",
    "visit_type": "OPD",
    "billed_amount": 35000.0,
    "length_of_stay_hours": 12.0,
    "chronic_flag": 0,
    "registration_date": "2025-02-01",
    "visit_date": "2025-08-01"
  }'
```

**Response (200 OK):**
```json
{
  "predicted_claim_status": "Paid",
  "probability_paid": 0.6700,
  "probability_pending": 0.2246,
  "probability_rejected": 0.1054,
  "model_version": "xgboost-v1-phase3",
  "warning": "This model's test-set performance for the Rejected class (AUROC 0.65) is modest. A formal statistical signal check found only weak association in the underlying data. Treat this prediction as a secondary screening signal, not a substitute for billing team review. See claim_model_card.md for details."
}
```

---

## POST /predict/claim — Outlier Billed Amount

**Request:** (billed_amount of 60,000 exceeds the IQR upper bound of
53,621.49 computed in Phase 2, so `billed_amount_is_outlier` is
internally set to 1)
```bash
curl -X POST http://localhost:8000/predict/claim \
  -H "Content-Type: application/json" \
  -d '{
    "insurance_provider": "MediCareX",
    "department": "General",
    "visit_type": "OPD",
    "billed_amount": 60000.0,
    "length_of_stay_hours": 5.0,
    "chronic_flag": 0,
    "registration_date": "2025-01-01",
    "visit_date": "2026-06-27"
  }'
```

**Response (200 OK):**
```json
{
  "predicted_claim_status": "Paid",
  "probability_paid": 0.8088,
  "probability_pending": 0.1762,
  "probability_rejected": 0.0150,
  "model_version": "xgboost-v1-phase3",
  "warning": "This model's test-set performance for the Rejected class (AUROC 0.65) is modest. A formal statistical signal check found only weak association in the underlying data. Treat this prediction as a secondary screening signal, not a substitute for billing team review. See claim_model_card.md for details."
}
```

---

## Error Responses

### Invalid category value (422 Unprocessable Entity)

**Request:** (`"department": "Pediatrics"` — not one of the six trained
department categories)
```bash
curl -X POST http://localhost:8000/predict/risk \
  -H "Content-Type: application/json" \
  -d '{"department": "Pediatrics", "visit_type": "ICU", "city": "Delhi", "gender": "M", "age": 65, "chronic_flag": 1, "length_of_stay_hours": 48.0, "doctor_id": 101, "visit_frequency": 3, "avg_los_per_patient": 40.0, "registration_date": "2025-01-15", "visit_date": "2025-08-01"}'
```

**Response (422 Unprocessable Entity):**
```json
{
  "detail": [
    {
      "type": "literal_error",
      "loc": ["body", "department"],
      "msg": "Input should be 'Cardiology', 'ER', 'General', 'ICU', 'Neurology' or 'Orthopedics'",
      "input": "Pediatrics",
      "ctx": {"expected": "'Cardiology', 'ER', 'General', 'ICU', 'Neurology' or 'Orthopedics'"}
    }
  ]
}
```

### Missing required fields (422 Unprocessable Entity)

**Request:** (only 2 of 12 required fields provided)
```bash
curl -X POST http://localhost:8000/predict/risk \
  -H "Content-Type: application/json" \
  -d '{"department": "ICU", "visit_type": "ICU"}'
```

**Response (422 Unprocessable Entity):** lists every missing field by
name — truncated here for brevity, but the real response enumerates all
10 missing fields individually:
```json
{
  "detail": [
    {"type": "missing", "loc": ["body", "city"], "msg": "Field required", ...},
    {"type": "missing", "loc": ["body", "gender"], "msg": "Field required", ...},
    ...
  ]
}
```

### Out-of-range numeric value (422 Unprocessable Entity)

**Request:** (`"age": -5`)

**Response (422 Unprocessable Entity):**
```json
{
  "detail": [
    {
      "type": "greater_than_equal",
      "loc": ["body", "age"],
      "msg": "Input should be greater than or equal to 0",
      "input": -5,
      "ctx": {"ge": 0}
    }
  ]
}
```

---

## Prediction Logging (Internal — Not an API Response)

Every successful prediction writes one structured log line. Captured
from `app/logs/predictions.log` during the live test run above:

```
2026-06-25T20:10:05.053608+00:00 | risk_model | xgboost-v1-phase3 | hash=7fcae45c5b189322 | prediction=Low
2026-06-25T20:10:10.184518+00:00 | claim_model | xgboost-v1-phase3 | hash=99e69def3f02c83c | prediction=Paid
```

(Shown here in human-readable form; the actual log line is JSON — see
`app/logging_utils.py` for the exact format.) The hash is computed over
the full input feature set, not the raw request body, so two requests
with identical clinical/billing content always hash identically
regardless of JSON key ordering.
