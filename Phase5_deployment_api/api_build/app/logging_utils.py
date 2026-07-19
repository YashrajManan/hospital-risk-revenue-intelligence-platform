"""
logging_utils.py

Implements the Phase 5 requirement: "Log predictions with timestamp,
model version, and input feature hash."

Design choice — we hash the input features rather than logging them in
full. This serves two purposes:
  1. Privacy: avoids writing potentially sensitive patient/billing details
     into plaintext logs.
  2. Auditability: the hash still lets you verify, later, whether two
     requests had identical inputs (e.g. to debug a "same input, different
     output" report) without needing to store the raw values.

Logs are written to stdout (captured by Docker's logging driver) AND to a
local file at /app/logs/predictions.log, so they survive container restarts
if that path is mounted as a volume. This sets up exactly what Phase 6
(Monitoring, Drift Detection, and Governance) will read from.
"""

import hashlib
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

LOG_DIR = Path(__file__).parent.parent / "logs"
PREDICTION_LOG_FILE = LOG_DIR / "predictions.log"


def setup_logging():
    """
    Configures the root logger to write to both stdout (for `docker logs`)
    and a rotating-by-restart file. Called once at application startup.
    """
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(PREDICTION_LOG_FILE),
        ],
    )


def hash_features(input_features: dict) -> str:
    """
    Produces a stable SHA-256 hash of the input feature dictionary.

    json.dumps with sort_keys=True ensures the same feature values always
    hash identically regardless of dict insertion order (Python dicts
    preserve insertion order, which would otherwise make two logically
    identical requests hash differently).

    default=str handles non-JSON-native types (e.g. date objects from
    Pydantic) by stringifying them before hashing.
    """
    canonical_json = json.dumps(input_features, sort_keys=True, default=str)
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()[:16]


def log_prediction(model_name: str, model_version: str, input_features: dict, prediction: str) -> None:
    """
    Writes one structured log line per prediction request, satisfying the
    Phase 5 requirement to log timestamp, model version, and input feature
    hash. Also includes the prediction itself and which model served it,
    since a log entry with no outcome is of limited use for later drift
    analysis (Phase 6).
    """
    logger = logging.getLogger("hospital_api.predictions")

    log_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_name": model_name,
        "model_version": model_version,
        "input_feature_hash": hash_features(input_features),
        "prediction": prediction,
    }

    logger.info(json.dumps(log_entry))
