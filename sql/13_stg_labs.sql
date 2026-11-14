-- Laboratory results, full-outer-joined so that someone with only some labs keeps a row.
-- Lab values are measured quantities with no refused / don't know codes.
-- Fasting glucose exists only for the morning fasting subsample, so it is missing for most people.
DROP VIEW IF EXISTS stg.labs CASCADE;
CREATE VIEW stg.labs AS
SELECT
    COALESCE(t.seqn, h.seqn, g.seqn, a.seqn)::bigint AS seqn,
    COALESCE(t.cycle, h.cycle, g.cycle, a.cycle)     AS cycle,
    t.lbxtc  AS total_chol,       -- mg/dL
    h.lbdhdd AS hdl,              -- mg/dL
    a.lbxgh  AS hba1c,            -- percent
    g.lbxglu AS fasting_glucose   -- mg/dL
FROM raw.tchol t
FULL JOIN raw.hdl h ON h.seqn = t.seqn AND h.cycle = t.cycle
FULL JOIN raw.ghb a ON a.seqn = COALESCE(t.seqn, h.seqn) AND a.cycle = COALESCE(t.cycle, h.cycle)
FULL JOIN raw.glu g ON g.seqn = COALESCE(t.seqn, h.seqn, a.seqn)
                   AND g.cycle = COALESCE(t.cycle, h.cycle, a.cycle);
