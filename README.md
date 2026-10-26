# Hospital Operations & Revenue Risk Intelligence Platform

A 6-phase healthcare business-analytics capstone on 25,000 real hospital visits from 5,000 patients across
6 departments, 4 insurance providers, and 6 cities: SQL analytics layer → exploratory data analysis and
data quality → claim-outcome and visit-risk classification models → model evaluation and explainability →
API deployment → drift detection and governance monitoring.

**Key real findings:** 2,332 high-value bills (>Rs 20,000) received zero insurance approval — a theoretical
~Rs 5.8 crore recovery opportunity, with ~Rs 2 crore realistically recoverable at a 35% resubmission
success rate. Insurer-level analysis identified one provider with the highest claim-rejection rate and
another as the slowest payer, both flagged as contract-level (not billing-process) issues. Cardiology
showed the lowest revenue-realization ratio of any department, pointing to a documentation/coding gap
worth a targeted audit.

**Two ML models, deployed live** at a Hugging Face Space API:
- **Claim outcome model** (Paid / Pending / Rejected) — recommended for production use as a pre-submission
  screen, meaningfully outperforming chance on the Rejected class specifically, the class that actually
  matters for revenue.
- **Visit risk model** (Low / Medium / High) — explicitly **not** recommended for clinical use yet: it
  struggled on the High Risk class because the available features have weak real statistical association
  with the risk labels in this dataset. Reported as an honest data limitation, not papered over as a model
  failure — the model would need real clinical signals (vitals, diagnosis codes, lab results, admission
  history) to be clinically usable.

Both models are monitored for data drift and prediction stability, with a governance document covering
limitations, retraining triggers, and compliance requirements, and a privacy-safe audit trail on every
prediction.
