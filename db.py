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
                        id, session_id, prompt, category,
                        safety, song_title, artist, one_liner, key_lyric,
                        iso_reasoning, album_art_url, preview_url, track_view_url,
                        streaming_links, user_agent, ip_hash
                    )
                    VALUES (%s,%s,%s,%s, %s,%s,%s,%s,%s, %s,%s,%s,%s, %s,%s,%s)
                    ON CONFLICT (id) DO NOTHING
                    """,
                    (
                        rec_id,
                        session_id,
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
) -> None:
    """Persist one event (view, share, save, open_spotify, etc)."""
    if _pool is None:
        return
    try:
        with _conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO events (recommendation_id, session_id, event_type)
                    VALUES (%s, %s, %s)
                    """,
                    (recommendation_id, session_id, event_type),
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
