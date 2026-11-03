-- CLEAN layer: typed, cleaned features ready for training (written by the
-- preprocess pod with diabetes_features.clean_features). Only encounters the
-- model is meant for: deaths and hospice discharges are excluded.

CREATE TABLE IF NOT EXISTS clean.diabetes (
    encounter_id              TEXT     PRIMARY KEY REFERENCES raw.diabetes (encounter_id),
    patient_nbr               TEXT     NOT NULL,    -- not a feature: used to split by patient
    -- numeric features
    age                       REAL     NOT NULL,    -- bracket midpoint
    time_in_hospital          INTEGER  NOT NULL,
    num_lab_procedures        INTEGER  NOT NULL,
    num_procedures            INTEGER  NOT NULL,
    num_medications           INTEGER  NOT NULL,
    number_outpatient         INTEGER  NOT NULL,
    number_emergency          INTEGER  NOT NULL,
    number_inpatient          INTEGER  NOT NULL,
    number_diagnoses          INTEGER  NOT NULL,
    -- categorical features
    race                      TEXT     NOT NULL,
    gender                    TEXT     NOT NULL,
    admission_type_id         TEXT     NOT NULL,
    discharge_disposition_id  TEXT     NOT NULL,
    admission_source_id       TEXT     NOT NULL,
    medical_specialty         TEXT     NOT NULL,
    diag_1                    TEXT     NOT NULL,    -- ICD-9 group
    diag_2                    TEXT     NOT NULL,
    diag_3                    TEXT     NOT NULL,
    max_glu_serum             TEXT     NOT NULL,
    a1cresult                 TEXT     NOT NULL,
    metformin                 TEXT     NOT NULL,
    repaglinide               TEXT     NOT NULL,
    nateglinide               TEXT     NOT NULL,
    glimepiride               TEXT     NOT NULL,
    glipizide                 TEXT     NOT NULL,
    glyburide                 TEXT     NOT NULL,
    pioglitazone              TEXT     NOT NULL,
    rosiglitazone             TEXT     NOT NULL,
    acarbose                  TEXT     NOT NULL,
    insulin                   TEXT     NOT NULL,
    glyburide_metformin       TEXT     NOT NULL,
    change                    TEXT     NOT NULL,
    diabetesmed               TEXT     NOT NULL,
    -- target and bookkeeping
    readmitted_30d            SMALLINT NOT NULL CHECK (readmitted_30d IN (0, 1)),
    split                     TEXT     CHECK (split IN ('train', 'validation', 'test')),  -- set by assign_split
    batch_id                  INTEGER  NOT NULL,
    processed_at              TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS diabetes_split_idx ON clean.diabetes (split);
