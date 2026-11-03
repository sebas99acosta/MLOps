import pandas as pd
import pytest

from diabetes_features import (
    FEATURES,
    RAW_COLUMNS,
    clean_features,
    eligible_mask,
    icd9_group,
    normalize_column_name,
    readmitted_within_30_days,
)

# One encounter, as text, exactly as it appears in the CSV / the raw table.
ROW = {
    "encounter_id": "2278392", "patient_nbr": "8222157", "race": "Caucasian",
    "gender": "Female", "age": "[70-80)", "weight": "?", "admission_type_id": "1",
    "discharge_disposition_id": "1", "admission_source_id": "7",
    "time_in_hospital": "3", "payer_code": "?", "medical_specialty": "?",
    "num_lab_procedures": "41", "num_procedures": "0", "num_medications": "13",
    "number_outpatient": "0", "number_emergency": "0", "number_inpatient": "1",
    "diag_1": "428", "diag_2": "250.01", "diag_3": "V45", "number_diagnoses": "9",
    "max_glu_serum": "None", "a1cresult": ">8",
    **{c: "No" for c in RAW_COLUMNS[24:47]},
    "insulin": "Up", "change": "Ch", "diabetesmed": "Yes", "readmitted": "<30",
}


def frame(**overrides):
    return pd.DataFrame([{**ROW, **overrides}])


@pytest.mark.parametrize("code, group", [
    ("428", "Circulatory"), ("459.9", "Circulatory"), ("785", "Circulatory"),
    ("460", "Respiratory"), ("786", "Respiratory"),
    ("250", "Diabetes"), ("250.83", "Diabetes"),
    ("820", "Injury"), ("715", "Musculoskeletal"), ("599", "Genitourinary"),
    ("153", "Neoplasms"), ("530", "Digestive"),
    ("V57", "Other"), ("E909", "Other"), ("276", "Other"),
    ("?", "Unknown"), ("", "Unknown"),
])
def test_icd9_groups_follow_strack_2014(code, group):
    assert icd9_group(code) == group


def test_column_names_are_normalized():
    assert normalize_column_name("glyburide-metformin") == "glyburide_metformin"
    assert normalize_column_name("A1Cresult") == "a1cresult"
    assert normalize_column_name("diabetesMed") == "diabetesmed"


def test_features_have_fixed_order_and_no_missing_values():
    features = clean_features(frame())
    assert list(features.columns) == FEATURES
    assert features.notna().all().all()


def test_none_means_not_tested_and_stays_a_category():
    features = clean_features(frame(max_glu_serum="None", a1cresult="None"))
    assert features.loc[0, "max_glu_serum"] == "None"
    assert features.loc[0, "a1cresult"] == "None"


def test_question_mark_becomes_unknown():
    features = clean_features(frame(race="?", medical_specialty="?", diag_1="?"))
    assert features.loc[0, "race"] == "Unknown"
    assert features.loc[0, "medical_specialty"] == "Unknown"
    assert features.loc[0, "diag_1"] == "Unknown"


def test_age_bracket_becomes_midpoint():
    assert clean_features(frame(age="[70-80)")).loc[0, "age"] == 75
    assert clean_features(frame(age="[0-10)")).loc[0, "age"] == 5


def test_codes_are_categorical_not_numeric():
    features = clean_features(frame(admission_type_id="6"))
    assert features.loc[0, "admission_type_id"] == "6"


@pytest.mark.parametrize("code", ["11", "13", "14", "19", "20", "21"])
def test_death_and_hospice_are_not_eligible(code):
    assert not eligible_mask(frame(discharge_disposition_id=code)).iloc[0]


def test_invalid_gender_is_not_eligible():
    assert not eligible_mask(frame(gender="Unknown/Invalid")).iloc[0]
    assert eligible_mask(frame()).iloc[0]


@pytest.mark.parametrize("label, target", [("<30", 1), (">30", 0), ("NO", 0)])
def test_binary_target(label, target):
    assert readmitted_within_30_days(frame(readmitted=label)).iloc[0] == target


def test_non_numeric_count_fails_loudly():
    with pytest.raises(ValueError):
        clean_features(frame(num_medications="abc"))
