"""Cleaning and feature rules shared by training (Airflow trainer pods) and
inference (API). One source of truth: whatever the model was trained on is
exactly what it receives in production (no training-serving skew)."""

from diabetes_features.cleaning import (
    CATEGORICAL_FEATURES,
    FEATURES,
    MEDICATIONS,
    NUMERIC_FEATURES,
    RAW_COLUMNS,
    TARGET,
    clean_features,
    eligible_mask,
    icd9_group,
    normalize_column_name,
    readmitted_within_30_days,
)

__all__ = [
    "CATEGORICAL_FEATURES",
    "FEATURES",
    "MEDICATIONS",
    "NUMERIC_FEATURES",
    "RAW_COLUMNS",
    "TARGET",
    "clean_features",
    "eligible_mask",
    "icd9_group",
    "normalize_column_name",
    "readmitted_within_30_days",
]
