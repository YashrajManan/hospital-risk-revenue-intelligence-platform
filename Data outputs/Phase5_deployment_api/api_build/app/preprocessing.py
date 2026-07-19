"""
preprocessing.py

Converts raw API request fields into the exact encoded feature vector each
model expects, in the exact column order from feature_schema.json. This
logic intentionally mirrors the notebooks (02_risk_model.ipynb,
03_claim_model.ipynb) step for step — the API must do precisely what
training did, or predictions will be silently wrong.

Two engineered features depend on the CURRENT date relative to a fixed
historical date, not on anything the notebooks needed at training time:
  - days_since_registration  (visit_date - registration_date)
  - visit_month / visit_day_of_week / visit_is_weekend / visit_quarter
      (all derived from visit_date)

These are computed here exactly as Phase 2's build_features.py did.
"""

import pandas as pd
from typing import Any


def encode_categorical(encoders: dict, column: str, value: str) -> int:
    """
    Applies a fitted LabelEncoder the same way the notebooks did at
    evaluation time: if the value was seen during training, encode it
    normally; if not (a category the model has never seen), map it to -1
    rather than raising an error. This matches the exact unseen-category
    handling documented in feature_schema.json.
    """
    encoder = encoders[column]
    if value in encoder.classes_:
        return int(encoder.transform([value])[0])
    return -1


def build_risk_features(request_data: dict, encoders: dict) -> pd.DataFrame:
    """
    Builds the 15-column feature DataFrame the risk model expects, in the
    exact order specified by feature_schema_risk_model.json's
    final_feature_order.
    """
    visit_date = pd.Timestamp(request_data["visit_date"])
    registration_date = pd.Timestamp(request_data["registration_date"])

    # Same clipping logic as Phase 2: a visit can never precede registration.
    # If raw input violates this, we clip at 0 rather than feed the model
    # a negative value it never saw during training.
    days_since_registration = max((visit_date - registration_date).days, 0)

    row = {
        "department_enc": encode_categorical(encoders, "department", request_data["department"]),
        "visit_type_enc": encode_categorical(encoders, "visit_type", request_data["visit_type"]),
        "city_enc": encode_categorical(encoders, "city", request_data["city"]),
        "gender_enc": encode_categorical(encoders, "gender", request_data["gender"]),
        "age": request_data["age"],
        "chronic_flag": request_data["chronic_flag"],
        "length_of_stay_hours": request_data["length_of_stay_hours"],
        "doctor_id": request_data["doctor_id"],
        "visit_frequency": request_data["visit_frequency"],
        "avg_los_per_patient": request_data["avg_los_per_patient"],
        "days_since_registration": days_since_registration,
        "visit_month": visit_date.month,
        "visit_day_of_week": visit_date.dayofweek,
        "visit_is_weekend": int(visit_date.dayofweek in (5, 6)),
        "visit_quarter": visit_date.quarter,
    }

    final_feature_order = [
        "department_enc", "visit_type_enc", "city_enc", "gender_enc",
        "age", "chronic_flag", "length_of_stay_hours", "doctor_id",
        "visit_frequency", "avg_los_per_patient", "days_since_registration",
        "visit_month", "visit_day_of_week", "visit_is_weekend", "visit_quarter"
    ]

    return pd.DataFrame([row])[final_feature_order]


def build_claim_features(
    request_data: dict, encoders: dict, provider_rejection_rates: pd.Series,
    outlier_bounds: dict
) -> pd.DataFrame:
    """
    Builds the 13-column feature DataFrame the claim model expects, in the
    exact order specified by feature_schema_claim_model.json's
    final_feature_order.

    provider_rejection_rates: a pandas Series indexed by insurance_provider,
        loaded from claim_model_provider_rejection_rate.joblib. This is the
        TRAINING-PERIOD-ONLY rate computed in the notebook to avoid leakage
        — at inference time we look it up rather than recompute it.

    outlier_bounds: precomputed IQR bounds for billed_amount (see
        OUTLIER_BOUNDS in main.py for where these literal values come from).
    """
    visit_date = pd.Timestamp(request_data["visit_date"])
    registration_date = pd.Timestamp(request_data["registration_date"])
    days_since_registration = max((visit_date - registration_date).days, 0)

    insurer = request_data["insurance_provider"]
    if insurer in provider_rejection_rates.index:
        rejection_rate = float(provider_rejection_rates[insurer])
    else:
        # Unseen insurer at inference time: fall back to the overall
        # training-period average, same defensive handling used in the notebook.
        rejection_rate = float(provider_rejection_rates.mean())

    billed_amount = request_data["billed_amount"]
    is_outlier = int(
        billed_amount < outlier_bounds["lower"] or billed_amount > outlier_bounds["upper"]
    )

    row = {
        "insurance_provider_enc": encode_categorical(encoders, "insurance_provider", insurer),
        "department_enc": encode_categorical(encoders, "department", request_data["department"]),
        "visit_type_enc": encode_categorical(encoders, "visit_type", request_data["visit_type"]),
        "billed_amount": billed_amount,
        "length_of_stay_hours": request_data["length_of_stay_hours"],
        "chronic_flag": request_data["chronic_flag"],
        "days_since_registration": days_since_registration,
        "visit_month": visit_date.month,
        "visit_day_of_week": visit_date.dayofweek,
        "visit_is_weekend": int(visit_date.dayofweek in (5, 6)),
        "visit_quarter": visit_date.quarter,
        "provider_rejection_rate": rejection_rate,
        "billed_amount_is_outlier": is_outlier,
    }

    final_feature_order = [
        "insurance_provider_enc", "department_enc", "visit_type_enc",
        "billed_amount", "length_of_stay_hours", "chronic_flag",
        "days_since_registration", "visit_month", "visit_day_of_week",
        "visit_is_weekend", "visit_quarter", "provider_rejection_rate",
        "billed_amount_is_outlier"
    ]

    return pd.DataFrame([row])[final_feature_order]
