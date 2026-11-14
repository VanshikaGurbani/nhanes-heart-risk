-- Up to four auscultatory readings per person. We average the readings that exist.
-- A reading of 0 is not a physiological value here (it marks "no reading"), so it is excluded
-- before averaging. If no valid reading exists, the average is NULL.
DROP VIEW IF EXISTS stg.blood_pressure CASCADE;
CREATE VIEW stg.blood_pressure AS
SELECT
    p.seqn::bigint AS seqn,
    p.cycle,
    (SELECT AVG(v) FROM (VALUES (p.bpxsy1), (p.bpxsy2), (p.bpxsy3), (p.bpxsy4)) t(v) WHERE v > 0)
        AS sbp_mean,
    (SELECT AVG(v) FROM (VALUES (p.bpxdi1), (p.bpxdi2), (p.bpxdi3), (p.bpxdi4)) t(v) WHERE v > 0)
        AS dbp_mean
FROM raw.bpx p;
