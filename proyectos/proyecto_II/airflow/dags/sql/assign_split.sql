-- Train / validation / test = 70 / 15 / 15, decided by a hash of patient_nbr:
-- - by patient: all encounters of a patient land in the same split, so no
--   patient is seen in training and then scored in test (no leakage);
-- - stable: a patient's split never changes as new batches arrive, and an
--   already-assigned row is never touched again.
UPDATE clean.diabetes AS c
SET split = CASE
        WHEN s.bucket < 70 THEN 'train'
        WHEN s.bucket < 85 THEN 'validation'
        ELSE 'test'
    END
FROM (
    SELECT encounter_id,
           mod(('x' || substr(md5(patient_nbr), 1, 7))::bit(28)::int, 100) AS bucket
    FROM clean.diabetes
    WHERE split IS NULL
) AS s
WHERE c.encounter_id = s.encounter_id;
