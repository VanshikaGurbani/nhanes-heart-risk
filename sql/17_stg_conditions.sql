-- Medical conditions questionnaire (self-reported "has a doctor ever told you that you had ...").
-- Asked of adults 20 and older. Codes: 1 yes, 2 no, 7 refused, 9 don't know, NULL not asked.
--
-- LABEL DEFINITION (documented in docs/DESIGN.md):
--   heart_disease = 1  if the person answered YES to coronary heart disease (MCQ160C),
--                      angina (MCQ160D) or heart attack (MCQ160E).
--   heart_disease = 0  only if all three were answered NO.
--   heart_disease = NULL (person dropped from the model table) if there is no YES and at least one
--                      of the three is refused, don't know or missing: the label is unknown.
-- Congestive heart failure (MCQ160B) and stroke (MCQ160F) are different conditions and are not
-- part of the label. They are kept as columns for exploration only and are never features.
DROP VIEW IF EXISTS stg.conditions CASCADE;
CREATE VIEW stg.conditions AS
WITH c AS (
    SELECT
        m.seqn::bigint AS seqn,
        m.cycle,
        NULLIF(NULLIF(m.mcq160c, 7), 9) AS chd,
        NULLIF(NULLIF(m.mcq160d, 7), 9) AS angina,
        NULLIF(NULLIF(m.mcq160e, 7), 9) AS heart_attack,
        NULLIF(NULLIF(m.mcq160b, 7), 9) AS chf,
        NULLIF(NULLIF(m.mcq160f, 7), 9) AS stroke
    FROM raw.mcq m
)
SELECT
    seqn,
    cycle,
    (chd = 1)          AS chd,
    (angina = 1)       AS angina,
    (heart_attack = 1) AS heart_attack,
    (chf = 1)          AS chf,
    (stroke = 1)       AS stroke,
    CASE
        WHEN chd = 1 OR angina = 1 OR heart_attack = 1 THEN 1
        WHEN chd = 2 AND angina = 2 AND heart_attack = 2 THEN 0
    END AS heart_disease
FROM c;
