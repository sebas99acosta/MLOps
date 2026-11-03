-- INFERENCE LOGS layer: one row per /predict call, written by the API. Input
-- kept as received, so predictions can feed future retraining, monitoring and
-- audits.

CREATE TABLE IF NOT EXISTS inference.predictions (
    request_id      UUID        PRIMARY KEY,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    input           JSONB       NOT NULL,   -- raw fields received by the API
    prediction      SMALLINT    NOT NULL CHECK (prediction IN (0, 1)),
    probability     REAL        NOT NULL,   -- P(readmitted < 30 days)
    threshold       REAL        NOT NULL,   -- decision threshold of the model used
    model_name      TEXT        NOT NULL,
    model_version   TEXT        NOT NULL,
    model_alias     TEXT        NOT NULL,
    latency_ms      REAL        NOT NULL
);

CREATE INDEX IF NOT EXISTS predictions_created_at_idx ON inference.predictions (created_at);
