-- RAW layer: the source file exactly as received, never overwritten.
-- Idempotent (IF NOT EXISTS): the DAG's create_tables task runs it on every run.

-- One row per batch and DAG run: which run loaded which slice of the file.
-- A retried run finds its own batch here instead of taking the next one.
CREATE TABLE IF NOT EXISTS raw.batch_log (
    batch_id       INTEGER     PRIMARY KEY,
    dag_run_id     TEXT        NOT NULL UNIQUE,
    source_file    TEXT        NOT NULL,
    first_row      INTEGER     NOT NULL,   -- 0-based position in the file
    rows_in_batch  INTEGER     NOT NULL,
    rows_inserted  INTEGER     NOT NULL,   -- < rows_in_batch only if rows were already loaded
    rows_valid     INTEGER,                -- filled in by quality_checks
    rows_rejected  INTEGER,
    loaded_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Every column of the CSV stays TEXT (casting is processing, not loading).
-- "?" and "None" are kept as they are.
CREATE TABLE IF NOT EXISTS raw.diabetes (
    encounter_id               TEXT NOT NULL,
    patient_nbr                TEXT,
    race                       TEXT,
    gender                     TEXT,
    age                        TEXT,
    weight                     TEXT,
    admission_type_id          TEXT,
    discharge_disposition_id   TEXT,
    admission_source_id        TEXT,
    time_in_hospital           TEXT,
    payer_code                 TEXT,
    medical_specialty          TEXT,
    num_lab_procedures         TEXT,
    num_procedures             TEXT,
    num_medications            TEXT,
    number_outpatient          TEXT,
    number_emergency           TEXT,
    number_inpatient           TEXT,
    diag_1                     TEXT,
    diag_2                     TEXT,
    diag_3                     TEXT,
    number_diagnoses           TEXT,
    max_glu_serum              TEXT,
    a1cresult                  TEXT,
    metformin                  TEXT,
    repaglinide                TEXT,
    nateglinide                TEXT,
    chlorpropamide             TEXT,
    glimepiride                TEXT,
    acetohexamide              TEXT,
    glipizide                  TEXT,
    glyburide                  TEXT,
    tolbutamide                TEXT,
    pioglitazone               TEXT,
    rosiglitazone              TEXT,
    acarbose                   TEXT,
    miglitol                   TEXT,
    troglitazone               TEXT,
    tolazamide                 TEXT,
    examide                    TEXT,
    citoglipton                TEXT,
    insulin                    TEXT,
    glyburide_metformin        TEXT,
    glipizide_metformin        TEXT,
    glimepiride_pioglitazone   TEXT,
    metformin_rosiglitazone    TEXT,
    metformin_pioglitazone     TEXT,
    change                     TEXT,
    diabetesmed                TEXT,
    readmitted                 TEXT,
    -- Load metadata required by the project.
    batch_id                   INTEGER     NOT NULL REFERENCES raw.batch_log (batch_id),
    load_timestamp             TIMESTAMPTZ NOT NULL DEFAULT now(),
    source_file                TEXT        NOT NULL,
    row_hash                   TEXT        NOT NULL,   -- md5 of the original CSV line
    -- loaded -> valid | rejected (quality_checks) -> processed | excluded (preprocess)
    status                     TEXT        NOT NULL DEFAULT 'loaded'
        CHECK (status IN ('loaded', 'valid', 'rejected', 'processed', 'excluded')),
    status_reason              TEXT,
    PRIMARY KEY (encounter_id)  -- one row per encounter: reloading is a no-op
);

CREATE INDEX IF NOT EXISTS diabetes_batch_status_idx ON raw.diabetes (batch_id, status);
