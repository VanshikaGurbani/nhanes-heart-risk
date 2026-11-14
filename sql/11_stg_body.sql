-- Body measures from the mobile exam centre. Values are measured, so there are no
-- refused / don't know codes; missing means not measured.
DROP VIEW IF EXISTS stg.body CASCADE;
CREATE VIEW stg.body AS
SELECT
    b.seqn::bigint AS seqn,
    b.cycle,
    b.bmxbmi       AS bmi,
    b.bmxwaist     AS waist_cm,
    b.bmxht        AS height_cm,
    b.bmxwt        AS weight_kg
FROM raw.bmx b;
