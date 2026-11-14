-- SMQ020: smoked at least 100 cigarettes in life (1 yes, 2 no, 7 refused, 9 don't know).
-- SMQ040: do you now smoke (1 every day, 2 some days, 3 not at all; asked only if SMQ020 = 1).
-- Special codes 7 and 9 become NULL, never a number.
DROP VIEW IF EXISTS stg.smoking CASCADE;
CREATE VIEW stg.smoking AS
SELECT
    s.seqn::bigint AS seqn,
    s.cycle,
    CASE
        WHEN NULLIF(NULLIF(s.smq020, 7), 9) = 2 THEN 'never'
        WHEN NULLIF(NULLIF(s.smq020, 7), 9) = 1 AND NULLIF(NULLIF(s.smq040, 7), 9) = 3 THEN 'former'
        WHEN NULLIF(NULLIF(s.smq020, 7), 9) = 1 AND NULLIF(NULLIF(s.smq040, 7), 9) IN (1, 2)
            THEN 'current'
    END AS smoking_status
FROM raw.smq s;
