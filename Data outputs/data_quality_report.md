# Data Quality Report — Phase 2
## Hospital Operations & Revenue Risk Intelligence Platform

### 1. Missing Value Analysis

| Column | Missing Count | Missing % | Root Cause |
|---|---|---|---|
| approved_amount | 0 (before imputation) | - | Claim not yet approved (Pending/Rejected status) |
| payment_days | 0 (before imputation) | - | Payment has not occurred yet (Pending/Rejected status) |
| length_of_stay_hours | - | - | No clear pattern; imputed with department-level median |

**Note:** Counts above reflect the dataset state captured at report-write time (after imputation in Step 7). See Step 3 of the notebook for pre-imputation missing counts and the relationship to claim_status.

### 2. Outlier Summary (IQR method)

| column               |   lower_bound |   upper_bound |   n_low_outliers |   n_high_outliers |
|:---------------------|--------------:|--------------:|-----------------:|------------------:|
| billed_amount        |   -13641      |    53621.5    |                0 |               373 |
| payment_days         |       -5.5    |       30.5    |                0 |               509 |
| length_of_stay_hours |      -16.0687 |       53.3413 |                0 |               256 |

### 3. Distribution Highlights

- 6 departments, 4 insurance providers, 6 cities, 3 visit types (OPD/ER/ICU)
- Risk score and claim status distributions vary meaningfully across departments and insurers — see Phase 1 SQL findings for exact figures (e.g., ICU has the highest High-Risk concentration, SecureLife has the highest rejection rate)

### 4. Feature Engineering Summary

New features created:
- `visit_frequency` — total visits per patient
- `avg_los_per_patient` — average length of stay per patient (computed post-imputation)
- `provider_rejection_rate` — historical rejection rate per insurer
- `days_since_registration` — days between patient registration and this visit
- `visit_month`, `visit_day_of_week`, `visit_is_weekend`, `visit_quarter` — time-based features
- `billed_amount_is_outlier`, `payment_days_is_outlier`, `length_of_stay_hours_is_outlier` — outlier flags

### 5. Imputation Strategy

| Column | Strategy | Justification |
|---|---|---|
| approved_amount | Fill with 0 | Not-yet-approved is distinct from approved-zero; mean would mislead |
| payment_days | Fill with -1 (sentinel) | Marks "no payment yet" as a distinct signal, not a short delay |
| length_of_stay_hours | Department-level median | Stay duration is strongly department-dependent |

### 6. Date Integrity Finding

`days_since_registration` was negative for 12,157 records (48.6%) because `visit_date` preceded `registration_date` in the raw data — a synthetic-data generation artifact, not a real-world possibility. These records are flagged via `registration_date_invalid` and the feature is clipped at 0. This flag itself is retained as a feature in case it carries any signal.

### 7. Known Limitation

`provider_rejection_rate` is calculated using the full dataset (including future records relative to any given visit). This creates a mild risk of information leakage in a strict time-series sense. Phase 3 will use a time-based train/test split — this feature should ideally be recalculated using only training-period data to avoid leakage. This is flagged here for transparency and revisited in Phase 3.

### 8. Output Files
- `model_table.csv` — final modeling dataset (25,000 rows x 32 columns)
- `build_features.py` — reusable feature engineering script
