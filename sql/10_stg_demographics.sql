-- Demographics and survey design variables. One row per person per cycle.
-- RIDAGEYR is top-coded by the CDC: 80 means "80 years or older".
-- RIDSTATR: 1 = interviewed only, 2 = interviewed and examined (labs and measurements exist).
-- Education and income are NOT carried into the model table (see docs/DESIGN.md).
DROP VIEW IF EXISTS stg.demographics CASCADE;
CREATE VIEW stg.demographics AS
SELECT
    d.seqn::bigint                       AS seqn,
    d.cycle,
    d.ridageyr::int                      AS age_years,
    CASE d.riagendr WHEN 1 THEN 'male' WHEN 2 THEN 'female' END AS sex,
    CASE d.ridreth3
        WHEN 1 THEN 'Mexican American'
        WHEN 2 THEN 'Other Hispanic'
        WHEN 3 THEN 'Non-Hispanic White'
        WHEN 4 THEN 'Non-Hispanic Black'
        WHEN 6 THEN 'Non-Hispanic Asian'
        WHEN 7 THEN 'Other or multiracial'
    END                                  AS race_ethnicity,
    (d.ridstatr = 2)                     AS was_examined,
    d.wtmec2yr                           AS exam_weight,  -- 2-year MEC exam weight
    d.sdmvpsu::int                       AS psu,
    d.sdmvstra::int                      AS strata
FROM raw.demo d;
