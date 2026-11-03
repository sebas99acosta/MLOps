"""raw.diabetes (rows that passed quality checks) -> clean.diabetes.

Incremental and idempotent: only rows with status 'valid' are read, and they
leave that status in the same transaction ('processed' or 'excluded'), so a
re-run never processes a row twice. Any error rolls everything back.
"""

import logging

import pandas as pd
from sqlalchemy import text

from diabetes_features import FEATURES, TARGET, clean_features, eligible_mask, readmitted_within_30_days
from trainer.common import db_engine, write_result

log = logging.getLogger("trainer.preprocess")

EXCLUSION_REASON = "death, hospice or invalid gender: not a readmission candidate"


def main() -> None:
    engine = db_engine()
    with engine.begin() as conn:
        raw = pd.read_sql(
            text("SELECT * FROM raw.diabetes WHERE status = 'valid' ORDER BY encounter_id"), conn
        )
        if raw.empty:
            log.info("No new valid rows to process.")
            write_result({"processed": 0, "excluded": 0})
            return

        eligible = eligible_mask(raw)
        kept = raw[eligible]

        clean = clean_features(kept)
        clean.insert(0, "encounter_id", kept["encounter_id"])
        clean.insert(1, "patient_nbr", kept["patient_nbr"])
        clean[TARGET] = readmitted_within_30_days(kept)
        clean["batch_id"] = kept["batch_id"]
        clean.to_sql("diabetes", conn, schema="clean", if_exists="append", index=False,
                     method="multi", chunksize=1000)

        conn.execute(
            text("UPDATE raw.diabetes SET status = 'processed', status_reason = NULL "
                 "WHERE encounter_id = ANY(:ids)"),
            {"ids": kept["encounter_id"].tolist()},
        )
        excluded_ids = raw.loc[~eligible, "encounter_id"].tolist()
        if excluded_ids:
            conn.execute(
                text("UPDATE raw.diabetes SET status = 'excluded', status_reason = :reason "
                     "WHERE encounter_id = ANY(:ids)"),
                {"ids": excluded_ids, "reason": EXCLUSION_REASON},
            )

    log.info("clean.diabetes: +%d rows (%d excluded), %d features", len(clean), len(excluded_ids), len(FEATURES))
    write_result({"processed": len(clean), "excluded": len(excluded_ids),
                  "positive_rate": round(float(clean[TARGET].mean()), 4)})


if __name__ == "__main__":
    main()
