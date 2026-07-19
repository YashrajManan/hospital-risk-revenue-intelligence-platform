---
title: Hospital Risk and Claim Prediction API
emoji: 🏥
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
license: mit
---

# Hospital Operations & Revenue Risk Intelligence Platform — Prediction API

FastAPI service exposing two trained XGBoost models from the capstone
project's Phase 3/4:

- **Visit Risk Classification** — predicts Low / Medium / High risk for a
  hospital visit
- **Claim Outcome Classification** — predicts Paid / Pending / Rejected
  for an insurance claim

## Try it

Once this Space finishes building, open `/docs` on this Space's URL for
the interactive Swagger UI, or call the endpoints directly:

```bash
curl https://<your-space-url>/health
```

See `sample_requests.md` in this repository for full example requests and
real captured responses.

## Important — Model Reliability

Both models were evaluated rigorously in Phase 4 and found to have
**limited predictive power** (Macro AUROC 0.53–0.57, only slightly above
the 0.50 random baseline). A formal statistical signal check, run before
any model training, found negligible real-world association between the
target variables and the available features in this synthetic dataset.
This is documented transparently in `risk_model_card.md` and
`claim_model_card.md`. Every prediction response from this API includes
an explicit `warning` field stating this. **Treat all predictions as
decision-support signals only, not as reliable standalone assessments.**

## Architecture

See `deployment_runbook.md` for the full build/run/operate guide and
`app/` for the FastAPI source code.
