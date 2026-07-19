"""
main.py

FastAPI service for the Hospital Operations & Revenue Risk Intelligence
Platform. Serves both trained models from Phase 3/4:

  - Model A: Visit Risk Classification  (risk_model_xgb.joblib)
  - Model B: Claim Outcome Classification (claim_model_xgb.joblib)

Endpoints:
  GET  /health          - health check, confirms both models are loaded
  POST /predict/risk    - predict visit risk (Low / Medium / High)
  POST /predict/claim   - predict claim outcome (Paid / Pending / Rejected)

Every prediction is logged with a timestamp, model version, and a hash of
the input features, per the Phase 5 brief's logging requirement (see
app/logging_utils.py).

IMPORTANT — read before trusting this API's output in any real decision:
Both models were evaluated in Phase 4 and found to have very limited
predictive power (Macro AUROC ~0.53-0.57, barely above the 0.50 random
baseline). This was traced to a near-total absence of statistical signal
in the underlying data (see the Phase 3 notebooks' Step 0 signal checks),
not a flaw in this API or the model training process. Both /predict
responses include a `warning` field surfacing this. See risk_model_card.md
and claim_model_card.md for full detail. This service should be treated as
a decision-SUPPORT signal only, not an autonomous decision-maker.
"""

import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from app.schemas import (
    RiskPredictionRequest, RiskPredictionResponse,
    ClaimPredictionRequest, ClaimPredictionResponse,
    HealthResponse,
)
from app.preprocessing import build_risk_features, build_claim_features
from app.logging_utils import log_prediction, setup_logging

# ─────────────────────────────────────────────────────────────────────────
# Paths and constants
# ─────────────────────────────────────────────────────────────────────────

MODELS_DIR = Path(__file__).parent / "models"

# IQR outlier bounds for billed_amount, computed in Phase 2 from the full
# training dataset. Hardcoded here (rather than recomputed at request time)
# because outlier bounds are a property of the TRAINING distribution, not
# of any single incoming request — recomputing per-request would be both
# wrong and impossible (a single row has no IQR).
BILLED_AMOUNT_OUTLIER_BOUNDS = {"lower": -13640.97, "upper": 53621.49}

MODEL_VERSION = "xgboost-v1-phase3"

RISK_CLASS_NAMES = ["Low", "Medium", "High"]
CLAIM_CLASS_NAMES = ["Paid", "Pending", "Rejected"]

RISK_MODEL_WARNING = (
    "This model's test-set performance (Macro AUROC 0.53) is barely above random. "
    "A formal statistical signal check found negligible association between risk_score "
    "and all available features. Treat this prediction as decision-support only, not "
    "a reliable risk assessment. See risk_model_card.md for details."
)
CLAIM_MODEL_WARNING = (
    "This model's test-set performance for the Rejected class (AUROC 0.65) is modest. "
    "A formal statistical signal check found only weak association in the underlying "
    "data. Treat this prediction as a secondary screening signal, not a substitute for "
    "billing team review. See claim_model_card.md for details."
)

logger = logging.getLogger("hospital_api")

# ─────────────────────────────────────────────────────────────────────────
# Model loading — done once at startup, kept in memory for the app's lifetime
# ─────────────────────────────────────────────────────────────────────────

ml_artifacts: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan handler: loads all model artifacts into memory once,
    when the container starts, rather than on every request. This is the
    standard pattern for ML model serving — model loading is comparatively
    slow (disk I/O + deserialization) and must not happen per-request.
    """
    setup_logging()
    logger.info("Loading model artifacts...")

    try:
        ml_artifacts["risk_model"] = joblib.load(MODELS_DIR / "risk_model_xgb.joblib")
        ml_artifacts["risk_encoders"] = joblib.load(MODELS_DIR / "risk_model_label_encoders.joblib")
        ml_artifacts["claim_model"] = joblib.load(MODELS_DIR / "claim_model_xgb.joblib")
        ml_artifacts["claim_encoders"] = joblib.load(MODELS_DIR / "claim_model_label_encoders.joblib")
        ml_artifacts["provider_rejection_rates"] = joblib.load(
            MODELS_DIR / "claim_model_provider_rejection_rate.joblib"
        )
        logger.info("All model artifacts loaded successfully.")
    except FileNotFoundError as e:
        # Fail loudly and immediately at startup rather than on the first
        # request — a missing model file should never surface as a 500
        # error to whoever happens to call the API first.
        logger.error(f"Failed to load model artifacts: {e}")
        raise RuntimeError(
            f"Could not load required model artifact: {e}. "
            f"Confirm all .joblib files are present in {MODELS_DIR}."
        ) from e

    yield  # application runs here

    logger.info("Shutting down, clearing model artifacts from memory.")
    ml_artifacts.clear()


app = FastAPI(
    title="Hospital Operations & Revenue Risk Intelligence Platform API",
    description=(
        "Serves two classification models: Visit Risk (Low/Medium/High) and "
        "Claim Outcome (Paid/Pending/Rejected). See /health for service status "
        "and /docs for interactive API documentation."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ─────────────────────────────────────────────────────────────────────────
# Health check endpoint
# ─────────────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["Operations"])
async def health_check():
    """
    Health check endpoint, per the Phase 5 brief's requirement for one.
    Returns 200 with model-loaded status whenever the service is up.
    A monitoring system (Phase 6) should poll this endpoint and alert if
    either model fails to report as loaded.
    """
    return HealthResponse(
        status="ok",
        risk_model_loaded="risk_model" in ml_artifacts,
        claim_model_loaded="claim_model" in ml_artifacts,
    )


# ─────────────────────────────────────────────────────────────────────────
# Model A — Visit Risk Prediction
# ─────────────────────────────────────────────────────────────────────────

@app.post("/predict/risk", response_model=RiskPredictionResponse, tags=["Predictions"])
async def predict_risk(request: RiskPredictionRequest):
    """
    Predicts visit risk level (Low / Medium / High) from visit and patient
    details. See RiskPredictionRequest for required fields and
    RISK_MODEL_WARNING above for this model's known reliability limits.
    """
    if "risk_model" not in ml_artifacts:
        raise HTTPException(status_code=503, detail="Risk model is not loaded. Check /health.")

    request_data = request.model_dump()

    try:
        features = build_risk_features(request_data, ml_artifacts["risk_encoders"])
    except Exception as e:
        # A malformed feature build is a client-input problem in practice
        # (e.g. a date field that doesn't parse the way we expect), not a
        # server crash — surface it as a 400, not a 500.
        logger.error(f"Feature build failed for risk prediction: {e}")
        raise HTTPException(status_code=400, detail=f"Could not process input: {e}")

    proba = ml_artifacts["risk_model"].predict_proba(features)[0]
    predicted_idx = int(proba.argmax())

    response = RiskPredictionResponse(
        predicted_risk=RISK_CLASS_NAMES[predicted_idx],
        probability_low=float(proba[0]),
        probability_medium=float(proba[1]),
        probability_high=float(proba[2]),
        model_version=MODEL_VERSION,
        warning=RISK_MODEL_WARNING,
    )

    log_prediction(
        model_name="risk_model",
        model_version=MODEL_VERSION,
        input_features=request_data,
        prediction=response.predicted_risk,
    )

    return response


# ─────────────────────────────────────────────────────────────────────────
# Model B — Claim Outcome Prediction
# ─────────────────────────────────────────────────────────────────────────

@app.post("/predict/claim", response_model=ClaimPredictionResponse, tags=["Predictions"])
async def predict_claim(request: ClaimPredictionRequest):
    """
    Predicts claim outcome (Paid / Pending / Rejected) from billing and
    visit details. See ClaimPredictionRequest for required fields and
    CLAIM_MODEL_WARNING above for this model's known reliability limits.
    """
    if "claim_model" not in ml_artifacts:
        raise HTTPException(status_code=503, detail="Claim model is not loaded. Check /health.")

    request_data = request.model_dump()

    try:
        features = build_claim_features(
            request_data,
            ml_artifacts["claim_encoders"],
            ml_artifacts["provider_rejection_rates"],
            BILLED_AMOUNT_OUTLIER_BOUNDS,
        )
    except Exception as e:
        logger.error(f"Feature build failed for claim prediction: {e}")
        raise HTTPException(status_code=400, detail=f"Could not process input: {e}")

    proba = ml_artifacts["claim_model"].predict_proba(features)[0]
    predicted_idx = int(proba.argmax())

    response = ClaimPredictionResponse(
        predicted_claim_status=CLAIM_CLASS_NAMES[predicted_idx],
        probability_paid=float(proba[0]),
        probability_pending=float(proba[1]),
        probability_rejected=float(proba[2]),
        model_version=MODEL_VERSION,
        warning=CLAIM_MODEL_WARNING,
    )

    log_prediction(
        model_name="claim_model",
        model_version=MODEL_VERSION,
        input_features=request_data,
        prediction=response.predicted_claim_status,
    )

    return response


# ─────────────────────────────────────────────────────────────────────────
# Generic error handler — never leak raw stack traces to API callers
# ─────────────────────────────────────────────────────────────────────────

@app.exception_handler(Exception)
async def generic_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "An internal error occurred. This has been logged for investigation."},
    )
