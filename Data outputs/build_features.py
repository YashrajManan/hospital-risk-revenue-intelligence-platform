"""
build_features.py

Standalone feature engineering script for the Hospital Operations &
Revenue Risk Intelligence Platform capstone project.

Takes the Phase 1 SQL output (master_visit_view.csv) and produces the
Phase 3-ready modeling dataset (model_table.csv), applying the same
feature engineering and missing-value strategy used in Phase 2.

Usage:
    python build_features.py
"""

import pandas as pd


def load_data(input_path="/kaggle/working/master_visit_view.csv"):
    """Load the Phase 1 SQL output."""
    return pd.read_csv(input_path)


def engineer_features(df):
    """
    Apply all Phase 2 feature engineering steps to the input DataFrame.
    Returns the DataFrame with new columns added.
    """
    df = df.copy()

    # Convert date columns to datetime
    df["visit_date"]        = pd.to_datetime(df["visit_date"])
    df["registration_date"] = pd.to_datetime(df["registration_date"])
    df["billing_date"]      = pd.to_datetime(df["billing_date"])

    # Outlier flags using IQR rule
    for col in ["billed_amount", "payment_days", "length_of_stay_hours"]:
        clean = df[col].dropna()
        q1, q3 = clean.quantile(0.25), clean.quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        df[f"{col}_is_outlier"] = (df[col] < lower) | (df[col] > upper)

    # Feature: visit frequency per patient
    visit_frequency = df.groupby("patient_id")["visit_id"].count().rename("visit_frequency")
    df = df.merge(visit_frequency, on="patient_id", how="left")

    # Feature: provider rejection rate
    provider_rejection_rate = (
        df.groupby("insurance_provider")["claim_status"]
        .apply(lambda x: (x == "Rejected").mean())
        .rename("provider_rejection_rate")
    )
    df = df.merge(provider_rejection_rate, on="insurance_provider", how="left")

    # Feature: days since registration
    # Known data issue: some visit_dates precede registration_date in this synthetic
    # dataset. We flag these and clip the feature at 0 rather than allow negative values.
    df["days_since_registration"] = (df["visit_date"] - df["registration_date"]).dt.days
    df["registration_date_invalid"] = df["days_since_registration"] < 0
    df["days_since_registration"] = df["days_since_registration"].clip(lower=0)

    # Time-based features
    df["visit_month"]       = df["visit_date"].dt.month
    df["visit_day_of_week"] = df["visit_date"].dt.dayofweek
    df["visit_is_weekend"]  = df["visit_day_of_week"].isin([5, 6]).astype(int)
    df["visit_quarter"]     = df["visit_date"].dt.quarter

    # Handle missing values
    df["approved_amount"] = df["approved_amount"].fillna(0)
    df["payment_days"]    = df["payment_days"].fillna(-1)
    dept_median_los = df.groupby("department")["length_of_stay_hours"].transform("median")
    df["length_of_stay_hours"] = df["length_of_stay_hours"].fillna(dept_median_los)

    # Feature: average length of stay per patient (computed AFTER imputation)
    avg_los_per_patient = df.groupby("patient_id")["length_of_stay_hours"].mean().rename("avg_los_per_patient")
    df = df.merge(avg_los_per_patient, on="patient_id", how="left")

    return df


def save_data(df, output_path="/kaggle/working/model_table.csv"):
    """Save the final modeling dataset."""
    df.to_csv(output_path, index=False)
    print(f"Saved: {output_path} ({df.shape[0]:,} rows x {df.shape[1]} columns)")


if __name__ == "__main__":
    raw_df  = load_data()
    feat_df = engineer_features(raw_df)
    save_data(feat_df)
