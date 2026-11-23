"""Streamlit UI for the diabetes readmission API.

Talks only to the inference API (never to MLflow or the database): it shows
the model the API is serving, sends one encounter to /predict and shows the
result with the model version that produced it.
"""

import os

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
            st.table({name.replace("_", " "): [f"{value:.3f}"] for name, value in metrics.items()})
        st.caption(f"Registered {info['registered_at']} · loaded {info['loaded_at']}")
        if info.get("promotion"):
            st.caption(f"Promotion: {info['promotion']}")
    if st.button("Refresh"):
        st.rerun()

st.title("🏥 Risk of readmission within 30 days")
st.write("Diabetic inpatient encounter (Diabetes 130-US hospitals dataset). Fill in the encounter and "
         "the API returns the probability of readmission in less than 30 days, using the model that "
         "Airflow promoted to `@champion` in MLflow.")

with st.form("encounter"):
    st.subheader("Patient")
    c1, c2, c3 = st.columns(3)
    race = c1.selectbox("Race", RACES)
    gender = c2.selectbox("Gender", ["Female", "Male"])
    age = c3.selectbox("Age", AGE_BRACKETS, index=7)

    st.subheader("Admission")
    c1, c2, c3, c4 = st.columns(4)
    admission_type_id = c1.selectbox("Admission type", list(ADMISSION_TYPES), format_func=ADMISSION_TYPES.get)
    discharge_disposition_id = c2.selectbox("Discharge to", list(DISCHARGES), format_func=DISCHARGES.get)
    admission_source_id = c3.selectbox("Admission source", list(ADMISSION_SOURCES),
                                       format_func=ADMISSION_SOURCES.get)
    medical_specialty = c4.selectbox("Admitting specialty", SPECIALTIES, help='"?" = not recorded')

    st.subheader("Stay and history")
    c1, c2, c3, c4 = st.columns(4)
    time_in_hospital = c1.number_input("Days in hospital", 1, 14, 5)
    num_lab_procedures = c2.number_input("Lab procedures", 0, 150, 45)
    num_procedures = c3.number_input("Other procedures", 0, 10, 1)
    num_medications = c4.number_input("Medications", 0, 100, 18)
    c1, c2, c3, c4 = st.columns(4)
    number_inpatient = c1.number_input("Inpatient visits (past year)", 0, 30, 2)
    number_emergency = c2.number_input("Emergency visits (past year)", 0, 80, 1)
    number_outpatient = c3.number_input("Outpatient visits (past year)", 0, 50, 0)
    number_diagnoses = c4.number_input("Number of diagnoses", 1, 16, 9)

    st.subheader("Diagnoses (ICD-9) and tests")
    c1, c2, c3, c4, c5 = st.columns(5)
    diag_1 = c1.text_input("Primary diagnosis", "428", help="e.g. 428 heart failure, 250.83 diabetes, V57")
    diag_2 = c2.text_input("Secondary diagnosis", "250.02")
    diag_3 = c3.text_input("Additional diagnosis", "403")
    max_glu_serum = c4.selectbox("Max glucose serum", ["None", "Norm", ">200", ">300"],
                                 help='"None" = test not performed')
    a1cresult = c5.selectbox("A1C result", ["None", "Norm", ">7", ">8"], index=3,
                             help='"None" = test not performed')

    st.subheader("Diabetes medication")
    c1, c2 = st.columns(2)
    change = c1.radio("Medication changed?", ["Ch", "No"], horizontal=True,
                      format_func=lambda v: "Yes" if v == "Ch" else "No")
    diabetesmed = c2.radio("Any diabetes medication?", ["Yes", "No"], horizontal=True)
    medication_columns = st.columns(4)
    medications = {
        name: medication_columns[i % 4].selectbox(name.replace("_", "-"), MEDICATION_LEVELS,
                                                  index=2 if name == "insulin" else 0)
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
