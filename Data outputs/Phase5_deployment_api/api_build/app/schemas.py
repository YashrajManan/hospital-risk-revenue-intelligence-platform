"""
schemas.py

Pydantic models defining the exact request and response shape for every
API endpoint. FastAPI uses these to validate incoming requests automatically
and to generate the interactive API docs (visible at /docs).

If a request doesn't match these field names, types, or constraints,
FastAPI rejects it with a 422 error BEFORE our prediction code ever runs —
this is the "request and response schema validation" required by the
Phase 5 brief.
"""

from pydantic import BaseModel, Field
from typing import Literal
from datetime import date


# ─────────────────────────────────────────────────────────────────────────
# MODEL A — Visit Risk Classification
# ─────────────────────────────────────────────────────────────────────────

class RiskPredictionRequest(BaseModel):
    """
    Input fields for a single visit risk prediction.

    These are the RAW business fields a hospital system would actually have
    on hand for an upcoming or in-progress visit — NOT the encoded/derived
    features the model needs internally. The API is responsible for doing
    that translation (see app/preprocessing.py), exactly like the notebook did.
    """
    department: Literal["Cardiology", "ER", "General", "ICU", "Neurology", "Orthopedics"] = Field(
        ..., description="Hospital department for this visit"
    )
    visit_type: Literal["OPD", "ER", "ICU"] = Field(
        ..., description="Type of visit"
    )
    city: Literal["Bangalore", "Chennai", "Delhi", "Hyderabad", "Mumbai", "Pune"] = Field(
        ..., description="Patient's city"
    )
    gender: Literal["M", "F"] = Field(
        ..., description="Patient gender"
    )
    age: int = Field(..., ge=0, le=130, description="Patient age in years")
    chronic_flag: int = Field(..., ge=0, le=1, description="1 if patient has a chronic condition, else 0")
    length_of_stay_hours: float = Field(..., ge=0, description="Expected or actual length of stay, in hours")
    doctor_id: int = Field(..., description="Attending doctor's ID")
    visit_frequency: int = Field(..., ge=1, description="Total number of visits this patient has had historically")
    avg_los_per_patient: float = Field(..., ge=0, description="This patient's average length of stay across all visits, in hours")
    registration_date: date = Field(..., description="Date the patient registered with the hospital")
    visit_date: date = Field(..., description="Date of this visit")

    class Config:
        json_schema_extra = {
            "example": {
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
            }
        }


class RiskPredictionResponse(BaseModel):
    """Output of a risk prediction: the predicted class plus full probability breakdown."""
    predicted_risk: Literal["Low", "Medium", "High"]
    probability_low: float
    probability_medium: float
    probability_high: float
    model_version: str
    warning: str | None = Field(
        default=None,
        description="Populated if the prediction has known reliability concerns (see model card)"
    )


# ─────────────────────────────────────────────────────────────────────────
# MODEL B — Claim Outcome Classification
# ─────────────────────────────────────────────────────────────────────────

class ClaimPredictionRequest(BaseModel):
    """Input fields for a single claim outcome prediction."""
    insurance_provider: Literal["CareOne", "HealthPlus", "MediCareX", "SecureLife"] = Field(
        ..., description="Insurance provider for this claim"
    )
    department: Literal["Cardiology", "ER", "General", "ICU", "Neurology", "Orthopedics"] = Field(
        ..., description="Hospital department for this visit"
    )
    visit_type: Literal["OPD", "ER", "ICU"] = Field(
        ..., description="Type of visit"
    )
    billed_amount: float = Field(..., ge=0, description="Total amount billed for this visit")
    length_of_stay_hours: float = Field(..., ge=0, description="Length of stay, in hours")
    chronic_flag: int = Field(..., ge=0, le=1, description="1 if patient has a chronic condition, else 0")
    registration_date: date = Field(..., description="Date the patient registered with the hospital")
    visit_date: date = Field(..., description="Date of this visit")

    class Config:
        json_schema_extra = {
            "example": {
                "insurance_provider": "SecureLife",
                "department": "Cardiology",
                "visit_type": "OPD",
                "billed_amount": 35000.0,
                "length_of_stay_hours": 12.0,
                "chronic_flag": 0,
                "registration_date": "2025-02-01",
                "visit_date": "2025-08-01"
            }
        }


class ClaimPredictionResponse(BaseModel):
    """Output of a claim outcome prediction: predicted class plus full probability breakdown."""
    predicted_claim_status: Literal["Paid", "Pending", "Rejected"]
    probability_paid: float
    probability_pending: float
    probability_rejected: float
    model_version: str
    warning: str | None = None


# ─────────────────────────────────────────────────────────────────────────
# Health check
# ─────────────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: Literal["ok"]
    risk_model_loaded: bool
    claim_model_loaded: bool
