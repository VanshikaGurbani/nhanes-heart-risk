-- DIQ010: doctor told you have diabetes (1 yes, 2 no, 3 borderline, 7 refused, 9 don't know).
-- Insulin and diabetes medication questions are deliberately NOT used (treatment follows diagnosis).
DROP VIEW IF EXISTS stg.diabetes CASCADE;
CREATE VIEW stg.diabetes AS
SELECT
    q.seqn::bigint AS seqn,
    q.cycle,
    CASE NULLIF(NULLIF(q.diq010, 7), 9)
        WHEN 1 THEN 'yes' WHEN 2 THEN 'no' WHEN 3 THEN 'borderline'
    END AS diabetes
FROM raw.diq q;
