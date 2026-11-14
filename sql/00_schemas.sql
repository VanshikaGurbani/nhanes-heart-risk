-- Four schemas. raw: files exactly as the CDC published them. stg: cleaned, one view per
-- NHANES table. model: the one-row-per-person modeling table. ref: small lookup tables.
CREATE SCHEMA IF NOT EXISTS raw;
CREATE SCHEMA IF NOT EXISTS ref;
CREATE SCHEMA IF NOT EXISTS stg;
CREATE SCHEMA IF NOT EXISTS model;
