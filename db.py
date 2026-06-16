"""
Postgres persistence for #Mood — recommendations + events.

Activates only when DATABASE_URL is set (Railway injects it automatically when a
Postgres service is attached). With no DATABASE_URL, every helper is a silent
no-op so local dev doesn't require a database.

Helpers:
  init_db()                — idempotent CREATE TABLE IF NOT EXISTS
  save_recommendation(...) — one row per /api/recommend call
  save_event(...)          — one row per /api/event call

All inserts are fire-and-forget at the request level: failures are logged but
never raised. We don't want a logging error to break user-facing flows.
"""
import os
import json
import psycopg
from psycopg_pool import ConnectionPool
from contextlib import contextmanager


def _normalize_dsn(url: str) -> str:
    # Railway sometimes hands out "postgres://..." while libpq expects "postgresql://"
    if url.startswith("postgres://"):
        return "postgresql://" + url[len("postgres://"):]
    return url


_DATABASE_URL = os.environ.get("DATABASE_URL")
_pool: ConnectionPool | None = None

if _DATABASE_URL:
    try:
        _pool = ConnectionPool(
            conninfo=_normalize_dsn(_DATABASE_URL),
            min_size=1,
            max_size=4,
            open=True,
            timeout=10,
        )
        print("db: Postgres pool initialized")
    except Exception as e:
        print(f"db: failed to init pool ({type(e).__name__}: {e}) — persistence disabled")
        _pool = None
else:
    print("db: no DATABASE_URL — persistence disabled (local dev mode)")


@contextmanager
def _conn():
    if _pool is None:
        yield None
        return
    with _pool.connection() as conn:
        yield conn


def init_db() -> None:
    """Idempotent schema setup. Safe to call on every boot."""
    if _pool is None:
        return
    with _conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS recommendations (
                    id              UUID PRIMARY KEY,
                    session_id      TEXT,
                    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),

                    -- input
                    prompt          TEXT NOT NULL,
                    category        TEXT,

                    -- output
                    safety          BOOLEAN NOT NULL DEFAULT FALSE,
                    song_title      TEXT,
                    artist          TEXT,
                    one_liner       TEXT,
                    key_lyric       TEXT,
                    iso_reasoning   TEXT,
                    album_art_url   TEXT,
                    preview_url     TEXT,
                    track_view_url  TEXT,
                    streaming_links JSONB,

                    -- meta
                    user_agent      TEXT,
                    ip_hash         TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_rec_session ON recommendations (session_id);
                CREATE INDEX IF NOT EXISTS idx_rec_created ON recommendations (created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_rec_category ON recommendations (category);

                -- Phase 6 additions: reaction signal + analytics-only language tag.
                -- Idempotent so multiple boots are safe; nullable so historical rows stay valid.
                ALTER TABLE recommendations ADD COLUMN IF NOT EXISTS helped SMALLINT;
                ALTER TABLE recommendations ADD COLUMN IF NOT EXISTS detected_language TEXT;

                CREATE TABLE IF NOT EXISTS events (
                    id                BIGSERIAL PRIMARY KEY,
                    recommendation_id UUID,
                    session_id        TEXT,
                    event_type        TEXT NOT NULL,
                    created_at        TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
                CREATE INDEX IF NOT EXISTS idx_event_rec  ON events (recommendation_id);
                CREATE INDEX IF NOT EXISTS idx_event_type ON events (event_type);
                CREATE INDEX IF NOT EXISTS idx_event_created ON events (created_at DESC);

                -- Phase 6 ship 3.5: user-written reflection + soft delete.
                -- notes is the user's own text per entry. deleted_at is a
                -- soft-delete sentinel so analytics queries keep working.
                ALTER TABLE recommendations ADD COLUMN IF NOT EXISTS notes TEXT;
                ALTER TABLE recommendations ADD COLUMN IF NOT EXISTS deleted_at TIMESTAMPTZ;

                -- Phase 6 ship 2: authenticated users. PK is the Clerk user_id
                -- string (e.g. "user_2abc..."), used directly so we don't carry
                -- a separate internal id.
                CREATE TABLE IF NOT EXISTS users (
                    id              TEXT PRIMARY KEY,
                    email           TEXT,
                    primary_provider TEXT,
                    signup_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    last_active_at  TIMESTAMPTZ,
                    signup_anon_id  TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_users_signup    ON users (signup_at DESC);
                CREATE INDEX IF NOT EXISTS idx_users_anon      ON users (signup_anon_id);

                -- user_id column on recommendations + events so post-signup
                -- queries can join cleanly. Indexes support the journal view.
                ALTER TABLE recommendations ADD COLUMN IF NOT EXISTS user_id TEXT;
                CREATE INDEX IF NOT EXISTS idx_rec_user ON recommendations (user_id, created_at DESC);

                ALTER TABLE events ADD COLUMN IF NOT EXISTS user_id TEXT;
                CREATE INDEX IF NOT EXISTS idx_event_user ON events (user_id, created_at DESC);
            """)
        conn.commit()
    print("db: schema ready")


def save_recommendation(
    rec_id: str,
    session_id: str | None,
    prompt: str,
    category: str | None,
    result: dict,
    user_agent: str | None = None,
    ip_hash: str | None = None,
    user_id: str | None = None,
) -> None:
    """Persist one recommend call. Result may be a safety response or a song payload."""
    if _pool is None:
        return
    try:
        with _conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO recommendations (
                        id, session_id, user_id, prompt, category,
                        safety, song_title, artist, one_liner, key_lyric,
                        iso_reasoning, album_art_url, preview_url, track_view_url,
                        streaming_links, user_agent, ip_hash
                    )
                    VALUES (%s,%s,%s,%s,%s, %s,%s,%s,%s,%s, %s,%s,%s,%s, %s,%s,%s)
                    ON CONFLICT (id) DO NOTHING
                    """,
                    (
                        rec_id,
                        session_id,
                        user_id,
                        prompt,
                        category,
                        bool(result.get("safety", False)),
                        result.get("song_title"),
                        result.get("artist"),
                        result.get("one_liner"),
                        result.get("key_lyric"),
                        result.get("iso_reasoning"),
                        result.get("album_art_url"),
                        result.get("preview_url"),
                        result.get("track_view_url"),
                        json.dumps(result.get("streaming_links")) if result.get("streaming_links") else None,
                        user_agent,
                        ip_hash,
                    ),
                )
            conn.commit()
    except Exception as e:
        print(f"db: save_recommendation failed: {type(e).__name__}: {e}")


def save_event(
    recommendation_id: str | None,
    session_id: str | None,
    event_type: str,
    user_id: str | None = None,
) -> None:
    """Persist one event (view, share, save, open_spotify, etc)."""
    if _pool is None:
        return
    try:
        with _conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO events (recommendation_id, session_id, user_id, event_type)
                    VALUES (%s, %s, %s, %s)
                    """,
                    (recommendation_id, session_id, user_id, event_type),
                )
            conn.commit()
    except Exception as e:
        print(f"db: save_event failed: {type(e).__name__}: {e}")


def set_reaction(rec_id: str, value: int | None) -> None:
    """Set the user's 👍/👎 reaction on a recommendation. value: 1, -1, or None to clear."""
    if _pool is None:
        return
    try:
        with _conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE recommendations SET helped = %s WHERE id = %s",
                    (value, rec_id),
                )
            conn.commit()
    except Exception as e:
        print(f"db: set_reaction failed: {type(e).__name__}: {e}")


def set_language(rec_id: str, lang: str | None) -> None:
    """Persist the langdetect result against a recommendation. Analytics only."""
    if _pool is None or not lang:
        return
    try:
        with _conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE recommendations SET detected_language = %s WHERE id = %s",
                    (lang, rec_id),
                )
            conn.commit()
    except Exception as e:
        print(f"db: set_language failed: {type(e).__name__}: {e}")


def upsert_user(
    user_id: str,
    email: str | None,
    primary_provider: str | None,
    signup_anon_id: str | None,
) -> None:
    """Insert a user on first signup; bump last_active_at on every subsequent call.
    primary_provider and signup_anon_id are locked in at first insert and never
    overwritten (a returning user's session_id is different from their original)."""
    if _pool is None:
        return
    try:
        with _conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO users (id, email, primary_provider, signup_anon_id, last_active_at)
                    VALUES (%s, %s, %s, %s, NOW())
                    ON CONFLICT (id) DO UPDATE
                        SET email = COALESCE(users.email, EXCLUDED.email),
                            last_active_at = NOW()
                    """,
                    (user_id, email, primary_provider, signup_anon_id),
                )
            conn.commit()
    except Exception as e:
        print(f"db: upsert_user failed: {type(e).__name__}: {e}")


def fetch_journal(user_id: str, limit: int = 20, before_iso: str | None = None) -> list[dict]:
    """Reverse-chron journal entries for a signed-in user. Cursor pagination via
    `before_iso` — pass the previous page's last entry's `created_at` to fetch
    the next page. Returns dicts shaped for the SPA's renderPick() so tapping a
    row can replay the song without an extra round-trip."""
    if _pool is None or not user_id:
        return []
    try:
        sql = """
            SELECT id, created_at, prompt,
                   song_title, artist, one_liner, key_lyric, iso_reasoning,
                   album_art_url, preview_url, track_view_url,
                   streaming_links, helped, detected_language, notes
              FROM recommendations
             WHERE user_id = %s
               AND safety = FALSE
               AND song_title IS NOT NULL
               AND deleted_at IS NULL
        """
        params: list = [user_id]
        if before_iso:
            sql += " AND created_at < %s"
            params.append(before_iso)
        sql += " ORDER BY created_at DESC LIMIT %s"
        params.append(limit)

        with _conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                cols = [d.name for d in cur.description]
                rows = [dict(zip(cols, r)) for r in cur.fetchall()]
        # Normalize created_at to ISO string so JSON serialization is straightforward.
        for r in rows:
            ca = r.get("created_at")
            if ca is not None:
                r["created_at"] = ca.isoformat()
        return rows
    except Exception as e:
        print(f"db: fetch_journal failed: {type(e).__name__}: {e}")
        return []


def update_entry_note(rec_id: str, user_id: str, notes: str | None) -> bool:
    """Set the user-written notes on a journal entry. Ownership-checked at the
    SQL level — a hostile rec_id from another account silently no-ops."""
    if _pool is None or not rec_id or not user_id:
        return False
    try:
        with _conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE recommendations
                       SET notes = %s
                     WHERE id = %s AND user_id = %s AND deleted_at IS NULL
                    """,
                    (notes, rec_id, user_id),
                )
                matched = cur.rowcount > 0
            conn.commit()
        return matched
    except Exception as e:
        print(f"db: update_entry_note failed: {type(e).__name__}: {e}")
        return False


def claim_recommendation(rec_id: str, user_id: str) -> bool:
    """Explicitly assign user_id to a single recommendation row, BUT only if
    it's still unclaimed (user_id IS NULL). Used by /api/auth/migrate for the
    deterministic post-signup handoff — the frontend passes the just-served
    recommendation_id so the song that triggered signup is reliably preserved
    even when the session_id heuristic finds nothing."""
    if _pool is None or not rec_id or not user_id:
        return False
    try:
        with _conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE recommendations
                       SET user_id = %s
                     WHERE id = %s AND user_id IS NULL
                    """,
                    (user_id, rec_id),
                )
                matched = cur.rowcount > 0
            conn.commit()
        return matched
    except Exception as e:
        print(f"db: claim_recommendation failed: {type(e).__name__}: {e}")
        return False


def soft_delete_entry(rec_id: str, user_id: str) -> bool:
    """Soft-delete a journal entry by stamping deleted_at. Analytics queries
    keep working; the journal view filters these out. Ownership-checked."""
    if _pool is None or not rec_id or not user_id:
        return False
    try:
        with _conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE recommendations
                       SET deleted_at = NOW()
                     WHERE id = %s AND user_id = %s AND deleted_at IS NULL
                    """,
                    (rec_id, user_id),
                )
                matched = cur.rowcount > 0
            conn.commit()
        return matched
    except Exception as e:
        print(f"db: soft_delete_entry failed: {type(e).__name__}: {e}")
        return False


def migrate_anon_to_user(user_id: str, anon_id: str) -> dict:
    """Migrate this anon session's events to the user — NOT recommendations.

    Recommendations are claimed explicitly via the save-to-journal action
    (`claim_recommendation`, invoked through /api/auth/migrate's
    pending_rec_id parameter OR /api/journal/<rec_id>/save). Bulk-claiming
    every served song would dump the alternates the user never even tapped
    on into their journal — exactly the bug this design fixes.

    Idempotent — safe to call on every signin from the same anon device.
    Returns {recommendations: 0, events: n} for observability; recommendations
    is always 0 here so the caller can layer the explicit-claim count on top.
    """
    if _pool is None or not anon_id or not user_id:
        return {"recommendations": 0, "events": 0}
    try:
        with _conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE events
                       SET user_id = %s
                     WHERE session_id = %s AND user_id IS NULL
                    """,
                    (user_id, anon_id),
                )
                evt_n = cur.rowcount
            conn.commit()
        return {"recommendations": 0, "events": evt_n}
    except Exception as e:
        print(f"db: migrate_anon_to_user failed: {type(e).__name__}: {e}")
        return {"recommendations": 0, "events": 0}
