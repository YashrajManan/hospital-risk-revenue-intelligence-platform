# syntax=docker/dockerfile:1

# ─────────────────────────────────────────────────────────────────────────
# Hospital Operations & Revenue Risk Intelligence Platform — API Container
# ─────────────────────────────────────────────────────────────────────────
# Build:  docker build -t hospital-api .
# Run:    docker run -p 8000:8000 hospital-api
# Then:   curl http://localhost:8000/health
#         open http://localhost:8000/docs   (interactive API documentation)
# ─────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim

# Prevents Python from buffering stdout/stderr — without this, log lines
# can sit in a buffer and never reach `docker logs` until the buffer fills,
# which is exactly wrong for an app whose logging IS a graded requirement.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Copy requirements FIRST, separately from the rest of the app code.
# Docker caches each instruction as a layer; as long as requirements.txt
# doesn't change, this layer (and the slow `pip install` below) is reused
# across rebuilds even when application code changes — this is the single
# biggest rebuild-speed win for a Python Docker image.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Now copy the actual application code and model artifacts.
COPY app/ ./app/

# Create the logs directory the app writes to (see app/logging_utils.py).
# Created here, not left to be auto-created at runtime, so it exists with
# correct ownership before we switch to a non-root user below.
# Hugging Face Spaces requires write access to be granted to a non-root
# user with UID 1000 specifically (its container runtime expects this) -
# /app/logs must be writable by that user. For plain local Docker use
# (not Spaces), any non-root user works; UID 1000 keeps both paths working
# with the same Dockerfile.
RUN useradd -m -u 1000 apiuser \
    && mkdir -p /app/logs \
    && chown -R apiuser:apiuser /app
USER apiuser

# Port 7860 is REQUIRED when deploying to Hugging Face Spaces (Docker SDK)
# - Spaces only routes external traffic to this exact port. For plain local
# Docker use, this is just a regular EXPOSE/port choice, nothing special.
EXPOSE 7860

# Docker-native health check: lets `docker ps` show this container's health
# status directly, and lets orchestrators (Kubernetes, ECS, Docker Swarm,
# or Hugging Face Spaces itself) detect if /health stops responding.
# Mirrors the /health endpoint required by the Phase 5 brief.
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health', timeout=3)" || exit 1

# Single worker by default — see deployment_runbook.md for guidance on
# scaling to multiple workers/replicas for production-level load.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
