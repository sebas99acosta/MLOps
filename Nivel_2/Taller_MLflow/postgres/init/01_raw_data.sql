-- Runs once, automatically, the first time the postgres-data container
-- starts on an empty volume (docker-entrypoint-initdb.d). If the CSV is
-- missing, the COPY fails and the container does not start: no silent
-- "database without data".

CREATE SCHEMA raw;
CREATE SCHEMA processed;

-- Raw stage: the CSV exactly as delivered. Every value stays TEXT (casting
-- is processing). row_id keeps the file order, so the notebook can read the
-- rows in a stable order and always draw the same sample.
CREATE TABLE raw.covertype (
    row_id                             BIGSERIAL PRIMARY KEY,
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
    loaded_at                          TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Column list maps the CSV's columns in order; row_id and loaded_at are
-- filled in by their defaults.
COPY raw.covertype (
    elevation, aspect, slope,
    horizontal_distance_to_hydrology, vertical_distance_to_hydrology,
    horizontal_distance_to_roadways,
    hillshade_9am, hillshade_noon, hillshade_3pm,
    horizontal_distance_to_fire_points,
    wilderness_area, soil_type, cover_type
)
FROM '/data/covertype.csv' WITH (FORMAT csv, HEADER true);

-- Processed stage: typed and validated, written by the training notebook.
-- The CHECK constraints document the valid ranges and act as a safety net.
CREATE TABLE processed.covertype (
    row_id                             BIGINT  PRIMARY KEY REFERENCES raw.covertype (row_id),
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
