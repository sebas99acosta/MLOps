"""Streamlit UI for the diabetes readmission API.

Talks only to the inference API (never to MLflow or the database): it shows
the model the API is serving, sends one encounter to /predict and shows the
result with the model version that produced it.
"""

import json
import os
from pathlib import Path

import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://api:8000")
TIMEOUT = 10

AGE_BRACKETS = [f"[{low}-{low + 10})" for low in range(0, 100, 10)]
MEDICATION_LEVELS = ["No", "Steady", "Up", "Down"]
MEDICATIONS = ["metformin", "repaglinide", "nateglinide", "glimepiride", "glipizide", "glyburide",
               "pioglitazone", "rosiglitazone", "acarbose", "insulin", "glyburide_metformin"]
RACES = ["Caucasian", "AfricanAmerican", "Hispanic", "Asian", "Other", "?"]
SPECIALTIES = ["?", "InternalMedicine", "Emergency/Trauma", "Family/GeneralPractice", "Cardiology",
               "Surgery-General", "Nephrology", "Orthopedics", "Radiologist", "Pulmonology", "Psychiatry"]
# The most frequent codes of IDS_mapping.csv.
ADMISSION_TYPES = {1: "1 · Emergency", 2: "2 · Urgent", 3: "3 · Elective", 5: "5 · Not available", 6: "6 · NULL"}
DISCHARGES = {1: "1 · Home", 2: "2 · Short-term hospital", 3: "3 · Skilled nursing facility",
              6: "6 · Home with health service", 18: "18 · NULL", 22: "22 · Rehab facility"}
ADMISSION_SOURCES = {7: "7 · Emergency room", 1: "1 · Physician referral", 2: "2 · Clinic referral",
                     4: "4 · Transfer from hospital", 6: "6 · Transfer from other facility", 17: "17 · NULL"}

# Example 1 is hand-made; the rest are real eligible encounters from Diabetes.csv
# (the first payloads of the Locust test).
EXAMPLES = json.loads((Path(__file__).parent / "examples.json").read_text())

st.set_page_config(page_title="Diabetes readmission", page_icon="🏥", layout="wide")


def api_get(path: str):
    try:
        response = requests.get(f"{API_URL}{path}", timeout=TIMEOUT)
        return response.json() if response.ok else None
    except requests.RequestException:
        return None


# --- Sidebar: the model being served. ---
with st.sidebar:
    st.header("Model in production")
    info = api_get("/model-info")
    if info is None:
        st.error(f"API not reachable or no model loaded ({API_URL}).")
    else:
        st.metric("Version", f"v{info['model_version']}", help=f"models:/{info['model_name']}@{info['model_alias']}")
        st.write(f"**Type:** {info['model_type']}")
        st.write(f"**Decision threshold:** {info['decision_threshold']:.3f}")
        metrics = info.get("metrics", {})
        if metrics:
            st.write("**Metrics (MLflow tags)**")
            labels = {"validation_pr_auc": "PR-AUC (val)", "test_pr_auc": "PR-AUC (test)",
                      "validation_recall": "Recall (val)", "validation_precision": "Precision (val)"}
            columns = st.columns(2)
            for i, (name, label) in enumerate(labels.items()):
                if name in metrics:
                    columns[i % 2].metric(label, f"{metrics[name]:.3f}")
        st.caption(f"Registered {info['registered_at']} · loaded {info['loaded_at']}")
        if info.get("promotion"):
            st.caption(f"Promotion: {info['promotion']}")
    if st.button("Refresh"):
        st.rerun()

st.title("🏥 Risk of readmission within 30 days")
st.write("Diabetic inpatient encounter (Diabetes 130-US hospitals dataset). Fill in the encounter and "
         "the API returns the probability of readmission in less than 30 days, using the model that "
         "Airflow promoted to `@champion` in MLflow.")

def describe(i: int) -> str:
    e = EXAMPLES[i]
    origin = "hand-made example" if i == 0 else "real encounter from the dataset"
    return (f"{i + 1}. {e['gender']} {e['age']}, primary diagnosis {e['diag_1']}, "
            f"{e['number_inpatient']} inpatient visits last year ({origin})")


# Picking an example refills the form: every widget's key includes the
# example number, so Streamlit creates it again with the example's values.
choice = st.selectbox("Load example values", range(len(EXAMPLES)), format_func=describe)
example, k = EXAMPLES[choice], f"-{choice}"


def select(label, options, field, column, **kwargs):
    """Selectbox preset to the example's value (added to the options if missing)."""
    options = list(options)
    if example[field] not in options:
        options.append(example[field])
    return column.selectbox(label, options, index=options.index(example[field]), key=field + k, **kwargs)


def number(label, low, high, field, column):
    return column.number_input(label, low, max(high, example[field]), example[field], key=field + k)


with st.form("encounter"):
    st.subheader("Patient")
    c1, c2, c3 = st.columns(3)
    race = select("Race", RACES, "race", c1)
    gender = select("Gender", ["Female", "Male"], "gender", c2)
    age = select("Age", AGE_BRACKETS, "age", c3)

    st.subheader("Admission")
    c1, c2, c3, c4 = st.columns(4)
    admission_type_id = select("Admission type", ADMISSION_TYPES, "admission_type_id", c1,
                               format_func=lambda v: ADMISSION_TYPES.get(v, str(v)))
    discharge_disposition_id = select("Discharge to", DISCHARGES, "discharge_disposition_id", c2,
                                      format_func=lambda v: DISCHARGES.get(v, str(v)))
    admission_source_id = select("Admission source", ADMISSION_SOURCES, "admission_source_id", c3,
                                 format_func=lambda v: ADMISSION_SOURCES.get(v, str(v)))
    medical_specialty = select("Admitting specialty", SPECIALTIES, "medical_specialty", c4,
                               help='"?" = not recorded')

    st.subheader("Stay and history")
    c1, c2, c3, c4 = st.columns(4)
    time_in_hospital = number("Days in hospital", 1, 14, "time_in_hospital", c1)
    num_lab_procedures = number("Lab procedures", 0, 150, "num_lab_procedures", c2)
    num_procedures = number("Other procedures", 0, 10, "num_procedures", c3)
    num_medications = number("Medications", 0, 100, "num_medications", c4)
    c1, c2, c3, c4 = st.columns(4)
    number_inpatient = number("Inpatient visits (past year)", 0, 30, "number_inpatient", c1)
    number_emergency = number("Emergency visits (past year)", 0, 80, "number_emergency", c2)
    number_outpatient = number("Outpatient visits (past year)", 0, 50, "number_outpatient", c3)
    number_diagnoses = number("Number of diagnoses", 1, 16, "number_diagnoses", c4)

    st.subheader("Diagnoses (ICD-9) and tests")
    c1, c2, c3, c4, c5 = st.columns(5)
    diag_1 = c1.text_input("Primary diagnosis", example["diag_1"], key="diag_1" + k,
                           help="e.g. 428 heart failure, 250.83 diabetes, V57")
    diag_2 = c2.text_input("Secondary diagnosis", example["diag_2"], key="diag_2" + k)
    diag_3 = c3.text_input("Additional diagnosis", example["diag_3"], key="diag_3" + k)
    max_glu_serum = select("Max glucose serum", ["None", "Norm", ">200", ">300"], "max_glu_serum", c4,
                           help='"None" = test not performed')
    a1cresult = select("A1C result", ["None", "Norm", ">7", ">8"], "a1cresult", c5,
                       help='"None" = test not performed')

    st.subheader("Diabetes medication")
    c1, c2 = st.columns(2)
    change = c1.radio("Medication changed?", ["Ch", "No"], horizontal=True, key="change" + k,
                      index=["Ch", "No"].index(example["change"]),
                      format_func=lambda v: "Yes" if v == "Ch" else "No")
    diabetesmed = c2.radio("Any diabetes medication?", ["Yes", "No"], horizontal=True, key="diabetesmed" + k,
                           index=["Yes", "No"].index(example["diabetesmed"]))
    medication_columns = st.columns(4)
    medications = {
        name: select(name.replace("_", "-"), MEDICATION_LEVELS, name, medication_columns[i % 4])
        for i, name in enumerate(MEDICATIONS)
    }

    submitted = st.form_submit_button("Predict", type="primary")

if submitted:
    encounter = {
        "race": race, "gender": gender, "age": age,
        "admission_type_id": admission_type_id, "discharge_disposition_id": discharge_disposition_id,
        "admission_source_id": admission_source_id, "medical_specialty": medical_specialty,
        "time_in_hospital": time_in_hospital, "num_lab_procedures": num_lab_procedures,
        "num_procedures": num_procedures, "num_medications": num_medications,
        "number_outpatient": number_outpatient, "number_emergency": number_emergency,
        "number_inpatient": number_inpatient, "number_diagnoses": number_diagnoses,
        "diag_1": diag_1, "diag_2": diag_2, "diag_3": diag_3,
        "max_glu_serum": max_glu_serum, "a1cresult": a1cresult,
        "change": change, "diabetesmed": diabetesmed, **medications,
    }
    try:
        response = requests.post(f"{API_URL}/predict", json=encounter, timeout=TIMEOUT)
    except requests.RequestException as error:
        st.error(f"Could not reach the API: {error}")
    else:
        if response.status_code == 422:
            st.error("The API rejected the input:")
            st.json(response.json()["detail"])
        elif not response.ok:
            st.error(f"API error {response.status_code}: {response.text}")
        else:
            result = response.json()
            high = result["prediction"] == 1
            (st.error if high else st.success)(
                f"**{'HIGH' if high else 'Low'} risk** of readmission within 30 days")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Probability", f"{result['probability']:.1%}")
            c2.metric("Threshold", f"{result['decision_threshold']:.1%}",
                      help="Above this probability the patient is flagged (tuned on validation: "
                           "max F2, at most 30% of patients flagged)")
            c3.metric("Model", f"v{result['model_version']}", help=f"{result['model_name']}@{result['model_alias']}")
            c4.metric("API latency", f"{result['latency_ms']:.0f} ms")
            st.progress(min(result["probability"], 1.0))
            st.caption(f"Request {result['request_id']} logged in inference.predictions")
