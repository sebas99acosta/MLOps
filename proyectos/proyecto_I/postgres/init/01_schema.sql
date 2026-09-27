-- Runs once, automatically, the first time the postgres-data container
-- starts on an empty volume (docker-entrypoint-initdb.d). Each stage of
-- the data (raw / processed / training) is its own Postgres schema.
--
-- Column names are lowercase snake_case: Postgres folds unquoted
-- identifiers to lowercase, so the API's "Elevation" etc. are mapped to
-- these names at ingestion time.

CREATE SCHEMA raw;
CREATE SCHEMA processed;
CREATE SCHEMA training;

-- Stage 1: exactly what the API returned. Every value stays TEXT (casting
-- is processing). row_hash = hash of the 13 original values; the primary
-- key makes duplicate rows a no-op via ON CONFLICT DO NOTHING.
CREATE TABLE raw.covertype (
    row_hash                           TEXT PRIMARY KEY,
    elevation                          TEXT,
    aspect                             TEXT,
    slope                              TEXT,
    horizontal_distance_to_hydrology   TEXT,
    vertical_distance_to_hydrology     TEXT,
    horizontal_distance_to_roadways    TEXT,
    hillshade_9am                      TEXT,
    hillshade_noon                     TEXT,
    hillshade_3pm                      TEXT,
    horizontal_distance_to_fire_points TEXT,
    wilderness_area                    TEXT,
    soil_type                          TEXT,
    cover_type                         TEXT,
    batch_number                       INTEGER     NOT NULL,
    run_id                             TEXT        NOT NULL,
    fetched_at                         TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- One row per DAG run: evidence of every collection without storing the
-- duplicate rows themselves.
CREATE TABLE raw.collection_log (
    run_id        TEXT PRIMARY KEY,
    batch_number  INTEGER     NOT NULL,
    rows_received INTEGER     NOT NULL,
    rows_new      INTEGER     NOT NULL,
    fetched_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Stage 2: typed and validated. The CHECK constraints document the valid
-- ranges and act as a safety net; the DAG filters invalid rows before
-- inserting, so they should never actually fire.
CREATE TABLE processed.covertype (
    row_hash                           TEXT PRIMARY KEY,
    elevation                          INTEGER NOT NULL,
    aspect                             INTEGER NOT NULL CHECK (aspect BETWEEN 0 AND 360),
    slope                              INTEGER NOT NULL,
    horizontal_distance_to_hydrology   INTEGER NOT NULL,
    vertical_distance_to_hydrology     INTEGER NOT NULL,
    horizontal_distance_to_roadways    INTEGER NOT NULL,
    hillshade_9am                      INTEGER NOT NULL CHECK (hillshade_9am  BETWEEN 0 AND 255),
    hillshade_noon                     INTEGER NOT NULL CHECK (hillshade_noon BETWEEN 0 AND 255),
    hillshade_3pm                      INTEGER NOT NULL CHECK (hillshade_3pm  BETWEEN 0 AND 255),
    horizontal_distance_to_fire_points INTEGER NOT NULL,
    wilderness_area                    TEXT    NOT NULL,
    soil_type                          TEXT    NOT NULL,
    cover_type                         SMALLINT NOT NULL,
    processed_at                       TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Stage 3: ready for training. Same typed columns plus a split that is
-- derived from row_hash, so a row never changes split as data grows.
CREATE TABLE training.covertype (
    row_hash                           TEXT PRIMARY KEY,
    elevation                          INTEGER NOT NULL,
    aspect                             INTEGER NOT NULL,
    slope                              INTEGER NOT NULL,
    horizontal_distance_to_hydrology   INTEGER NOT NULL,
    vertical_distance_to_hydrology     INTEGER NOT NULL,
    horizontal_distance_to_roadways    INTEGER NOT NULL,
    hillshade_9am                      INTEGER NOT NULL,
    hillshade_noon                     INTEGER NOT NULL,
    hillshade_3pm                      INTEGER NOT NULL,
    horizontal_distance_to_fire_points INTEGER NOT NULL,
    wilderness_area                    TEXT    NOT NULL,
    soil_type                          TEXT    NOT NULL,
    cover_type                         SMALLINT NOT NULL,
    split                              TEXT    NOT NULL CHECK (split IN ('train', 'test'))
);
