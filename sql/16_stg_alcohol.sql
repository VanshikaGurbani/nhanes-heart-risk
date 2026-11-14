-- Only the two alcohol questions that exist in all four cycles are used. Cycle J (2017-2018)
-- redesigned the alcohol section, so ALQ101 / ALQ110 are not comparable across cycles.
-- ALQ130: average drinks on days you drank in the past 12 months. 777 = refused, 999 = don't know.
--   It is skipped for people who did not drink in the past 12 months. If the alcohol section was
--   answered (ALQ151 present) and ALQ130 was skipped, the person did not drink, so drinks per day = 0.
-- ALQ151: ever had 4 or 5 drinks nearly every day (1 yes, 2 no, 7 refused, 9 don't know).
DROP VIEW IF EXISTS stg.alcohol CASCADE;
CREATE VIEW stg.alcohol AS
SELECT
    a.seqn::bigint AS seqn,
    a.cycle,
    CASE
        WHEN a.alq130 IN (777, 999) THEN NULL
        WHEN a.alq130 IS NOT NULL THEN a.alq130
        WHEN a.alq151 IS NOT NULL THEN 0
    END AS drinks_per_day,
    CASE NULLIF(NULLIF(a.alq151, 7), 9) WHEN 1 THEN 'yes' WHEN 2 THEN 'no' END AS ever_heavy_drinker
FROM raw.alq a;
