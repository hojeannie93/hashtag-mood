-- ───────────────────────────────────────────────────────────────────────────
-- #Mood — Phase 6 retention & reaction metrics
-- ───────────────────────────────────────────────────────────────────────────
-- These are owned by humans, run by hand against the prod Postgres. Plug each
-- into Railway's Data → Query tab (or psql $DATABASE_URL).
--
-- The two numbers we care about this phase:
--   1. Day-7 return rate  → are we earning the native app conversation?
--   2. Reaction rate      → are we actually helping people, or just playing
--                            music they shrug at?
--
-- Don't read these for the first ~10 days after Ship 3 — the cohort is too
-- small and the 7-day window hasn't filled. Three days of data is noise.
-- ───────────────────────────────────────────────────────────────────────────


-- ============================================================================
-- 1. Day-7 return rate
-- ============================================================================
-- "Of users who signed up at least 7 full days ago, what fraction have logged
--  an entry (recommendation) within the 7 days following their signup?"
-- This is the only retention number that matters in Phase 6. Target: ≥ 25%.
--
-- Definitions:
--   • "user"  = a row in `users` (every signup creates one).
--   • "entry" = a row in `recommendations` with safety = false.
--   • "Day-7" = at least one entry whose created_at is within
--               (signup_at, signup_at + 7 days].

WITH cohort AS (
  SELECT
    id,
    signup_at,
    primary_provider
  FROM users
  WHERE signup_at < NOW() - INTERVAL '7 days'
),
returned AS (
  SELECT
    c.id,
    EXISTS (
      SELECT 1
      FROM recommendations r
      WHERE r.user_id = c.id
        AND r.safety = FALSE
        AND r.created_at >  c.signup_at
        AND r.created_at <= c.signup_at + INTERVAL '7 days'
    ) AS came_back
  FROM cohort c
)
SELECT
  COUNT(*) FILTER (WHERE came_back) AS returners,
  COUNT(*)                          AS eligible_users,
  ROUND(
    100.0 * COUNT(*) FILTER (WHERE came_back) / NULLIF(COUNT(*), 0),
    1
  ) AS d7_return_rate_pct
FROM returned;


-- ── Day-7 return rate split by signup provider (Google vs Apple) ────────────
-- Useful for figuring out whether one provider's users stick around better.

WITH cohort AS (
  SELECT id, signup_at, primary_provider
  FROM users
  WHERE signup_at < NOW() - INTERVAL '7 days'
),
returned AS (
  SELECT
    c.id,
    c.primary_provider,
    EXISTS (
      SELECT 1
      FROM recommendations r
      WHERE r.user_id = c.id
        AND r.safety = FALSE
        AND r.created_at >  c.signup_at
        AND r.created_at <= c.signup_at + INTERVAL '7 days'
    ) AS came_back
  FROM cohort c
)
SELECT
  COALESCE(primary_provider, '(unknown)') AS provider,
  COUNT(*) FILTER (WHERE came_back) AS returners,
  COUNT(*)                          AS eligible_users,
  ROUND(
    100.0 * COUNT(*) FILTER (WHERE came_back) / NULLIF(COUNT(*), 0),
    1
  ) AS d7_return_rate_pct
FROM returned
GROUP BY 1
ORDER BY 2 DESC;


-- ============================================================================
-- 2. Reaction rate
-- ============================================================================
-- "Of every non-safety song we served, what fraction got a 👍 or 👎?"
-- Only ground-truth signal of whether we're actually helping. Tracking it
-- daily makes prompt-engine changes legible — if reaction rate drops the day
-- after a prompt tweak, the tweak hurt.

SELECT
  created_at::date AS day,
  COUNT(*)                                  AS served,
  COUNT(*) FILTER (WHERE helped = 1)        AS thumbs_up,
  COUNT(*) FILTER (WHERE helped = -1)       AS thumbs_down,
  COUNT(*) FILTER (WHERE helped IS NOT NULL) AS reacted,
  ROUND(
    100.0 * COUNT(*) FILTER (WHERE helped IS NOT NULL) / NULLIF(COUNT(*), 0),
    1
  ) AS reaction_pct,
  ROUND(
    100.0 * COUNT(*) FILTER (WHERE helped = 1)
    / NULLIF(COUNT(*) FILTER (WHERE helped IS NOT NULL), 0),
    1
  ) AS positive_pct_of_reacted
FROM recommendations
WHERE safety = FALSE
GROUP BY day
ORDER BY day DESC
LIMIT 60;


-- ============================================================================
-- 3. Funnel — anon → signed-in → returned
-- ============================================================================
-- For each day, how many anon sessions saw at least one recommendation, how
-- many of those went on to sign up, and how many of those signups have come
-- back in their first 7 days. Lets us see if signup conversion is the
-- bottleneck vs day-7 stickiness.

WITH anon_sessions AS (
  -- Distinct anon sessions per day (taking the day of FIRST recommendation
  -- in that session so each session counts on one day only).
  SELECT
    session_id,
    MIN(created_at)::date AS day
  FROM recommendations
  WHERE safety = FALSE
  GROUP BY session_id
),
signed_up AS (
  SELECT
    u.signup_anon_id AS session_id,
    u.id             AS user_id,
    u.signup_at::date AS signup_day
  FROM users u
  WHERE u.signup_anon_id IS NOT NULL
)
SELECT
  a.day,
  COUNT(DISTINCT a.session_id)                            AS anon_sessions,
  COUNT(DISTINCT s.user_id)                               AS signups,
  ROUND(
    100.0 * COUNT(DISTINCT s.user_id)
    / NULLIF(COUNT(DISTINCT a.session_id), 0),
    1
  ) AS signup_conversion_pct
FROM anon_sessions a
LEFT JOIN signed_up s ON s.session_id = a.session_id
GROUP BY a.day
ORDER BY a.day DESC
LIMIT 30;
