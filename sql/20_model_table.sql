-- One row per adult (age 20 or older) with a known heart disease label, one table for modeling.
-- Left joins keep people who were interviewed but not examined; their lab columns are NULL and
-- the missing-value strategy is decided in Python, not here.
DROP TABLE IF EXISTS model.heart_risk CASCADE;
CREATE TABLE model.heart_risk AS
SELECT
    d.seqn,
    d.cycle,
    r.years                                  AS survey_years,
    r.split                                  AS split,       -- temporal: train < valid < test
    d.age_years,
    CASE WHEN d.age_years < 40 THEN '20-39'
         WHEN d.age_years < 60 THEN '40-59'
         ELSE '60+' END                      AS age_band,
    d.sex,
    d.race_ethnicity,                         -- kept for description only, never a feature
    d.was_examined,
    b.bmi,
    b.waist_cm,
    p.sbp_mean,
    p.dbp_mean,
    l.total_chol,
    l.hdl,
    l.hba1c,
    l.fasting_glucose,
    s.smoking_status,
    q.diabetes,
    a.drinks_per_day,
    a.ever_heavy_drinker,
    c.heart_disease,
    c.chd, c.angina, c.heart_attack,          -- label components, for exploration only
    c.chf, c.stroke,                          -- NOT part of the label, never features
    d.exam_weight,
    d.psu,
    d.strata
FROM stg.demographics d
JOIN stg.conditions   c ON c.seqn = d.seqn AND c.cycle = d.cycle
JOIN ref.cycle        r ON r.letter = d.cycle
LEFT JOIN stg.body           b ON b.seqn = d.seqn AND b.cycle = d.cycle
LEFT JOIN stg.blood_pressure p ON p.seqn = d.seqn AND p.cycle = d.cycle
LEFT JOIN stg.labs           l ON l.seqn = d.seqn AND l.cycle = d.cycle
LEFT JOIN stg.smoking        s ON s.seqn = d.seqn AND s.cycle = d.cycle
LEFT JOIN stg.diabetes       q ON q.seqn = d.seqn AND q.cycle = d.cycle
LEFT JOIN stg.alcohol        a ON a.seqn = d.seqn AND a.cycle = d.cycle
WHERE d.age_years >= 20
  AND c.heart_disease IS NOT NULL;

ALTER TABLE model.heart_risk ADD PRIMARY KEY (seqn);
CREATE INDEX heart_risk_split_idx ON model.heart_risk (split);

-- Accounting for everyone who was dropped, so the funnel is auditable (see validate.py).
DROP TABLE IF EXISTS model.cohort_funnel CASCADE;
CREATE TABLE model.cohort_funnel AS
SELECT
    d.cycle,
    COUNT(*)                                                                  AS all_participants,
    COUNT(*) FILTER (WHERE d.age_years >= 20)                                 AS adults_20_plus,
    COUNT(*) FILTER (WHERE d.age_years >= 20 AND c.seqn IS NOT NULL)          AS adults_with_mcq,
    COUNT(*) FILTER (WHERE d.age_years >= 20 AND c.heart_disease IS NOT NULL) AS adults_with_label
FROM stg.demographics d
LEFT JOIN stg.conditions c ON c.seqn = d.seqn AND c.cycle = d.cycle
GROUP BY d.cycle
ORDER BY d.cycle;
