"""Cleaning rules for the Diabetes 130-US hospitals dataset.

Decisions (see README): follow Strack et al. 2014 where it fits a predictive
goal — drop weight and payer_code, keep missing specialty/race as their own
category, group ICD-9 diagnoses into 9 clinical groups, binary target
"readmitted within 30 days", and exclude encounters that ended in death or
hospice (those patients cannot be readmitted).

Input is the raw data as text, exactly as in the CSV (column names already
normalized with `normalize_column_name`). Missing values arrive as "?", and
"None" in max_glu_serum / A1Cresult means "test not performed": it is a
category, never a missing value.
"""

import pandas as pd

# Every column of the source CSV, in file order, as stored in raw tables.
RAW_COLUMNS = [
    "encounter_id", "patient_nbr", "race", "gender", "age", "weight",
    "admission_type_id", "discharge_disposition_id", "admission_source_id",
    "time_in_hospital", "payer_code", "medical_specialty",
    "num_lab_procedures", "num_procedures", "num_medications",
    "number_outpatient", "number_emergency", "number_inpatient",
    "diag_1", "diag_2", "diag_3", "number_diagnoses",
    "max_glu_serum", "a1cresult",
    "metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride",
    "acetohexamide", "glipizide", "glyburide", "tolbutamide", "pioglitazone",
    "rosiglitazone", "acarbose", "miglitol", "troglitazone", "tolazamide",
    "examide", "citoglipton", "insulin", "glyburide_metformin",
    "glipizide_metformin", "glimepiride_pioglitazone",
    "metformin_rosiglitazone", "metformin_pioglitazone",
    "change", "diabetesmed", "readmitted",
]

TARGET = "readmitted_30d"

# Medications used in at least 100 encounters. The other 10 are almost
# constant ("No") and examide / citoglipton are fully constant.
MEDICATIONS = [
    "metformin", "repaglinide", "nateglinide", "glimepiride", "glipizide",
    "glyburide", "pioglitazone", "rosiglitazone", "acarbose", "insulin",
    "glyburide_metformin",
]

NUMERIC_FEATURES = [
    "age",  # bracket "[70-80)" -> midpoint 75: keeps the order of the brackets
    "time_in_hospital", "num_lab_procedures", "num_procedures",
    "num_medications", "number_outpatient", "number_emergency",
    "number_inpatient", "number_diagnoses",
]

CATEGORICAL_FEATURES = [
    "race", "gender",
    # Codes, not quantities: treated as categories.
    "admission_type_id", "discharge_disposition_id", "admission_source_id",
    "medical_specialty",
    "diag_1", "diag_2", "diag_3",          # grouped into 9 ICD-9 groups
    "max_glu_serum", "a1cresult",          # "None" = test not performed
    *MEDICATIONS,
    "change", "diabetesmed",
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Discharged as expired (11, 19, 20, 21) or to hospice (13, 14).
EXCLUDED_DISCHARGE_CODES = {"11", "13", "14", "19", "20", "21"}
MISSING = "?"
UNKNOWN = "Unknown"


def normalize_column_name(name: str) -> str:
    """CSV header -> SQL-friendly name: 'glyburide-metformin' -> 'glyburide_metformin',
    'A1Cresult' -> 'a1cresult'."""
    return name.strip().lower().replace("-", "_")


def icd9_group(code: str) -> str:
    """ICD-9 code -> one of the 9 groups used by Strack et al. 2014 (Table 2)."""
    code = (code or "").strip()
    if code in ("", MISSING):
        return UNKNOWN
    if code[0] in "VvEe":  # supplementary classifications
        return "Other"
    try:
        value = float(code)
    except ValueError:
        return UNKNOWN
    whole = int(value)
    if 390 <= value < 460 or whole == 785:
        return "Circulatory"
    if 460 <= value < 520 or whole == 786:
        return "Respiratory"
    if 520 <= value < 580 or whole == 787:
        return "Digestive"
    if whole == 250:
        return "Diabetes"
    if 800 <= value < 1000:
        return "Injury"
    if 710 <= value < 740:
        return "Musculoskeletal"
    if 580 <= value < 630 or whole == 788:
        return "Genitourinary"
    if 140 <= value < 240:
        return "Neoplasms"
    return "Other"


def _age_midpoint(bracket: str) -> float:
    """'[70-80)' -> 75.0"""
    low, high = bracket.strip("[)").split("-")
    return (int(low) + int(high)) / 2


def eligible_mask(raw: pd.DataFrame) -> pd.Series:
    """Rows the model is meant for: alive and not sent to hospice, with a valid gender."""
    return (
        ~raw["discharge_disposition_id"].astype(str).str.strip().isin(EXCLUDED_DISCHARGE_CODES)
        & (raw["gender"].astype(str).str.strip() != "Unknown/Invalid")
    )


def readmitted_within_30_days(raw: pd.DataFrame) -> pd.Series:
    """Binary target: 1 if readmitted in less than 30 days, else 0 ('>30' and 'NO')."""
    return (raw["readmitted"].astype(str).str.strip() == "<30").astype(int)


def clean_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Raw text columns -> model features (FEATURES, in that order).

    Used both by the training pipeline and by the API, on the same raw format,
    so the model always receives exactly what it was trained on. Rows are not
    filtered here: use `eligible_mask` for that.
    """
    text = raw[[c for c in FEATURES]].astype(str).apply(lambda column: column.str.strip())
    features = pd.DataFrame(index=raw.index)

    features["age"] = text["age"].map(_age_midpoint)
    for column in NUMERIC_FEATURES[1:]:
        features[column] = pd.to_numeric(text[column], errors="raise").astype("int64")

    for column in CATEGORICAL_FEATURES:
        values = text[column]
        if column.startswith("diag_"):
            values = values.map(icd9_group)
        else:
            values = values.replace({MISSING: UNKNOWN, "": UNKNOWN})
        features[column] = values

    return features[FEATURES]
