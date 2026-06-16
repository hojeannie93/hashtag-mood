#!/usr/bin/env python3
"""
#Mood backend — Flask wrapper around engine.recommend()
"""
import io
import os
import re
import uuid
from collections import OrderedDict
from pathlib import Path
from urllib.parse import quote_plus
import requests
from flask import Flask, request, jsonify, send_from_directory, Response
from flask_cors import CORS
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

from engine import recommend  # noqa: E402
import db  # noqa: E402

# Clerk authentication. Optional at boot — if CLERK_SECRET_KEY is missing,
# _current_user() returns None for every request and every endpoint that
# branches on auth state falls back to its anon path. Lets the app run
# locally without Clerk credentials configured.
CLERK_SECRET_KEY = os.environ.get("CLERK_SECRET_KEY")
CLERK_AUTHORIZED_PARTIES = [
    p.strip() for p in os.environ.get("CLERK_AUTHORIZED_PARTIES", "").split(",")
    if p.strip()
]
_clerk_sdk = None
if CLERK_SECRET_KEY:
    try:
        from clerk_backend_api import (  # noqa: E402
            Clerk as _Clerk,
            authenticate_request as _clerk_authenticate_request,
            AuthenticateRequestOptions as _ClerkAuthOpts,
        )
        _clerk_sdk = _Clerk(bearer_auth=CLERK_SECRET_KEY)
        print("clerk: SDK ready")
    except ImportError:
        print("clerk: clerk-backend-api not installed — auth disabled")
        _clerk_sdk = None

def _current_user(req) -> dict | None:
    """Return {id, email} for an authenticated request, or None for anon.
    Tolerant — endpoints that work for both anon and authed users branch on the result.
    Looks up primary_provider lazily via _get_primary_provider when needed."""
    if not _clerk_sdk:
        return None
    try:
        state = _clerk_authenticate_request(
            req,
            _ClerkAuthOpts(
                secret_key=CLERK_SECRET_KEY,
                authorized_parties=CLERK_AUTHORIZED_PARTIES or None,
            ),
        )
        if not state.is_signed_in or not state.payload:
            return None
        return {
            "id": state.payload.get("sub"),
            "email": state.payload.get("email"),
        }
    except Exception as e:
        print(f"clerk auth failed: {type(e).__name__}: {e}")
        return None

def _get_primary_provider(user_id: str) -> str | None:
    """Resolve the OAuth provider the user signed up with. One Clerk Backend API
    call. Returns 'google' / 'apple' / None. Called from /api/auth/migrate only."""
    if not _clerk_sdk or not user_id:
        return None
    try:
        user = _clerk_sdk.users.get(user_id=user_id)
        accounts = getattr(user, "external_accounts", None) or []
        for ea in accounts:
            provider = getattr(ea, "provider", None) or ""
            if provider.startswith("oauth_"):
                return provider[len("oauth_"):]
        return None
    except Exception as e:
        print(f"clerk user lookup failed: {type(e).__name__}: {e}")
        return None

# Language detection (analytics-only — never used to gate user-facing behavior).
# DetectorFactory.seed makes detect() deterministic across runs for reproducible
# analytics; a non-deterministic result on a 2-word prompt is acceptable but a
# stable one is nicer to query against.
try:
    from langdetect import detect as _langdetect, DetectorFactory as _LDF
    _LDF.seed = 0
    def _detect_language(text: str) -> str | None:
        try:
            return _langdetect(text)
        except Exception:
            return None
except ImportError:  # pragma: no cover — production should always have it
    def _detect_language(text: str) -> str | None:
        return None

app = Flask(__name__, static_folder="static")
CORS(app)

db.init_db()

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# Album art now served from a pre-generated pool in static/album_art/. The
# classifier maps each recommendation's emotional texture to a cover. To
# regenerate the pool, run: python generate_art_pool.py
ART_POOL_DIR = Path(__file__).parent / "static" / "album_art"
ART_TAGS = [
    "elegiac", "peaceful", "numb", "reflective", "stuck",
    "belonging", "anxious", "acute", "liminal", "hopeful",
]
ART_FALLBACK = "peaceful"  # used if a tag is missing or classifier returns junk

# In-memory state (process-local; resets on restart)
sessions: dict[str, list[str]] = {}
recommendations: dict[str, dict] = {}
events: list[dict] = []


CAT_STYLE_ANCHOR = (
    "Clean modern anime illustration in the slice-of-life tradition of Studio Ghibli "
    "and Kyoto Animation cozy interior scenes. Confident line art with subtle weight "
    "variation, soft watercolor color washes inside the lines, gentle cel-shading "
    "for warmth and depth. Maximalist cozy aesthetic — every corner of the frame "
    "thoughtfully filled with detail. Each composition should stand alone as a "
    "portrait worth keeping as a wallpaper. "
    "CHARACTER: a small chibi cat with creamy off-white fur, simple rounded head, "
    "two triangular ears, tiny pink triangular nose, soft pink rosy blush circles "
    "on each cheek, no whiskers, no clothing. EXPRESSION VARIES BY SCENE — see "
    "scene-specific direction. Eyes can be closed sleepy arches (content), softly "
    "open with small black pupils (alert/curious), looking down quietly (sad), "
    "looking up softly (hopeful), half-lidded thoughtful, or any state the mood "
    "calls for. The expression must match the emotional moment. "
    "ENVIRONMENT: maximalist cozy — books, lamps, plants, teacups, curtains, rugs, "
    "knit throws, fairy lights, framed botanical prints, ceramic vessels, fountain "
    "pens, leather journals, vinyl records, hanging plants, ferns, candles, vintage "
    "radios. Every scene should feel like someone actually lives there. "
    "LIGHTING: cinematic and intentional. Use light to express mood — warm amber "
    "lamp glow, golden hour rays, moonlit indigo, twilight gradient, midnight "
    "starlight, soft window backlight, color temperature contrast between warm "
    "interior and cool exterior. Light is half the storytelling. "
    "PALETTE: warm pastels with richer accent saturation. Cream, sage, dusty rose, "
    "amber, sky blue, butter yellow, soft terra-cotta, muted plum, deep night blue "
    "and indigo for night scenes, soft silver moonlight. Colors feel alive. "
    "COMPOSITION: square 1:1 with foreground / middle / background depth. Could be "
    "a still frame from a quiet slice-of-life anime. "
    "STRICT: NO text, NO words, NO letters, NO logos, NO signage anywhere — book "
    "spines, mugs, notebooks, calendars must be wordless. NO border or frame "
    "around the image."
)


CLASSIFIER_SYSTEM = """
You classify the emotional texture of a music recommendation so we can pick the right
pre-painted album cover from a pool of 10.

CRITICAL: Classify the user's CURRENT emotional state — described in the one-liner.
Do NOT classify the song's forward push (the iso_reasoning describes where the song
takes them, NOT where they are). The cover must meet the user where they are, not
where the song is going.

Tags (output exactly one, lowercase, nothing else):

- elegiac     — quiet weight that doesn't lift: slow loss, faded friendship, soft grief
                that's sitting. The one-liner names absence, silence, what's gone.
- peaceful    — comfortable solitude, calm contentment, morning ease. NO hedge, no
                awareness of absence. The one-liner is settled.
- numb        — flat, disconnected, can't access feeling, going through motions.
- reflective  — introspective question, looking back or inward, "who am I becoming",
                growth-vs-change wondering. NOT used for decision pressure.
- stuck       — paused while life moves around you, standing still while others advance,
                frustrated waiting, time-themed. The one-liner names the gap between
                you and motion happening elsewhere.
- belonging   — wanting to be near or part of something, friend-group longing, the ache
                of being on the outside looking in.
- anxious     — restless decision pressure, "what if I choose wrong", risk weighing,
                racing thoughts. The one-liner names a choice or a stake.
- acute       — fresh wound: just-cheated, just-broken-up, just-laid-off, just-betrayed,
                raw heartbreak. The wound is still bleeding.
- liminal     — in-between two places or identities, hometown return, "everything looks
                the same but nothing feels the same", neither here nor there.
- hopeful     — gentle forward motion already underway, first light after a hard night.
                Use SPARINGLY — most one-liners that *sound* hopeful are actually
                elegiac/stuck/reflective and just have a slight forward arc.

ANTI-EXAMPLES (these are common misclassifications — get them right):
- "For when the silence between old friends says more than the words"
    → elegiac (NOT hopeful — the user is in the silence, not past it)
- "For when you're happy for them but standing still"
    → stuck (NOT liminal — the user names the gap, not a transition)
- "For when the next move could change everything but staying feels incomplete"
    → anxious (NOT reflective — there is a choice at stake)
- "For when the familiar world has shifted and you're not sure where you fit"
    → liminal (NOT stuck — identity/place is in transition)
- "For when you can't tell if it's growth or just a different kind of lost"
    → reflective (NOT liminal — the question is inward)

Output ONLY the tag. No explanation, no punctuation, no quotes.
"""


def classify_emotion(one_liner: str, iso_reasoning: str) -> str:
    """Map a recommendation to an album-cover tag via gpt-4o-mini."""
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": CLASSIFIER_SYSTEM},
                {"role": "user", "content": (
                    f"one-liner: {one_liner}\n"
                    f"iso_reasoning: {iso_reasoning}"
                )},
            ],
            temperature=0.0,
        )
        tag = resp.choices[0].message.content.strip().lower()
        # Trim any stray punctuation/quotes the model adds
        tag = tag.strip(".,'\" \n")
        if tag in ART_TAGS:
            return tag
        print(f"  classifier returned unknown tag {tag!r}, falling back")
    except Exception as e:
        print(f"  classifier failed: {type(e).__name__}: {e}")
    return ART_FALLBACK


def pick_album_assets(one_liner: str, iso_reasoning: str,
                      song_title: str, artist: str,
                      itunes_track: dict | None = None) -> dict:
    """Return {album_art_url, preview_url, track_view_url}. Falls back to cat pool.

    The iTunes lookup happens inside engine.recommend() in parallel with the
    one-liner generation, so the caller passes the pre-fetched track here.
    If not provided (e.g. external caller), we lazily fetch as a fallback.
    """
    if itunes_track is None:
        from engine import lookup_itunes_track
        itunes_track = lookup_itunes_track(song_title, artist)
    if itunes_track.get("art_url"):
        print(f"  iTunes hit: {song_title} (preview: {bool(itunes_track.get('preview_url'))})")
        return {
            "album_art_url": itunes_track["art_url"],
            "preview_url": itunes_track.get("preview_url"),
            "track_view_url": itunes_track.get("track_view_url"),
        }

    # Fallback: classify and pick a cat cover, no preview
    tag = classify_emotion(one_liner, iso_reasoning)
    path = ART_POOL_DIR / f"{tag}.png"
    if not path.exists():
        if (ART_POOL_DIR / f"{ART_FALLBACK}.png").exists():
            tag = ART_FALLBACK
        else:
            return {"album_art_url": None, "preview_url": None, "track_view_url": None}
    print(f"  album art: fallback cat cover ({tag})")
    return {
        "album_art_url": f"/static/album_art/{tag}.png",
        "preview_url": None,
        "track_view_url": None,
    }


@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/sso-callback")
def sso_callback():
    """Dedicated SPA route Clerk redirects back to after the OAuth provider
    returns. The SPA detects this path on load and calls handleRedirectCallback
    to finish the auth handshake."""
    return send_from_directory("static", "index.html")


@app.route("/favicon.ico")
def favicon():
    # Browsers auto-request this before parsing the <link rel="icon"> in <head>.
    # Serve the SVG to avoid a console 404 on first paint.
    return send_from_directory("static", "favicon.svg", mimetype="image/svg+xml")


VALID_CATEGORIES = {"rhythmic", "melodic", "rock", "instrumental", "atmospheric"}


@app.route("/api/recommend", methods=["POST"])
def api_recommend():
    data = request.get_json(silent=True) or {}
    prompt = (data.get("prompt") or "").strip()
    session_id = data.get("session_id") or str(uuid.uuid4())
    used_songs = data.get("used_songs") or sessions.get(session_id, [])
    raw_category = (data.get("category") or "").strip().lower()
    category = raw_category if raw_category in VALID_CATEGORIES else None
    auth_user = _current_user(request)
    auth_user_id = auth_user["id"] if auth_user else None

    if not prompt:
        return jsonify({"error": "empty prompt"}), 400

    try:
        result = recommend(prompt, used_songs=used_songs, category=category)
    except Exception as e:
        return jsonify({"error": f"engine failed: {e}"}), 500

    if result.get("safety"):
        safety_id = str(uuid.uuid4())
        db.save_recommendation(
            rec_id=safety_id,
            session_id=session_id,
            prompt=prompt,
            category=category,
            result=result,
            user_agent=request.headers.get("User-Agent"),
            user_id=auth_user_id,
        )
        return jsonify({"safety": True, "message": result["message"]})

    def _hydrate_pick(pick: dict) -> dict:
        """Turn an engine pick into a fully-shaped client payload — assets,
        streaming links, fresh rec_id — and persist to DB + memory."""
        pick_rec_id = str(uuid.uuid4())
        assets = pick_album_assets(
            pick["one_liner"],
            pick.get("iso_reasoning", ""),
            pick["song_title"],
            pick["artist"],
            itunes_track=pick.get("itunes_track"),
        )
        q = quote_plus(f"{pick['artist']} {pick['song_title']}")
        streaming_links = {
            "apple_music": assets.get("track_view_url"),
            "spotify":     f"https://open.spotify.com/search/{q}",
            "youtube":     f"https://music.youtube.com/search?q={q}",
        }
        full_pick = {
            "one_liner": pick["one_liner"],
            "song_title": pick["song_title"],
            "artist": pick["artist"],
            "key_lyric": pick["key_lyric"],
            "iso_reasoning": pick.get("iso_reasoning", ""),
            "album_art_url": assets["album_art_url"],
            "preview_url": assets.get("preview_url"),
            "track_view_url": assets.get("track_view_url"),
            "streaming_links": streaming_links,
            "plain_lyrics": pick.get("plain_lyrics"),
            "recommendation_id": pick_rec_id,
            "notes": None,  # fresh entries start without a user-written note
            "safety": False,
        }
        recommendations[pick_rec_id] = full_pick
        sessions.setdefault(session_id, []).append(
            f"{pick['song_title']} by {pick['artist']}"
        )
        # Always save with user_id=NULL — even for already-signed-in users.
        # The journal is now an explicit-save list: a song doesn't land in
        # there until the user taps "save to journal". This avoids the
        # alternates the user never even saw silently appearing.
        db.save_recommendation(
            rec_id=pick_rec_id,
            session_id=session_id,
            prompt=prompt,
            category=category,
            result=full_pick,
            user_agent=request.headers.get("User-Agent"),
            user_id=None,
        )
        db.set_language(pick_rec_id, detected_lang)
        return full_pick

    detected_lang = _detect_language(prompt)
    primary_full = _hydrate_pick(result["primary_pick"])
    alternate_fulls = [_hydrate_pick(p) for p in result.get("alternate_picks", [])]

    # Response keeps the primary pick's fields at top level (so existing client
    # paths — sync-lyrics, story-card, share — keep working unchanged) and adds
    # alternate_picks for the skip-to-next-song flow.
    response = {**primary_full, "alternate_picks": alternate_fulls}
    return jsonify(response)


_ITUNES_HOST_RE = re.compile(r"^https://is\d+(?:-ssl)?\.mzstatic\.com/")


@app.route("/api/art-proxy")
def api_art_proxy():
    url = request.args.get("url", "")
    allowed_prefixes = (
        "https://oaidalleapiprodscus.blob.core.windows.net/",
        "https://cdn.openai.com/",
        "https://images.openai.com/",
    )
    if not (url.startswith(allowed_prefixes) or _ITUNES_HOST_RE.match(url)):
        return "invalid url", 400
    try:
        upstream = requests.get(url, stream=True, timeout=20)
    except Exception as e:
        return f"fetch failed: {e}", 502
    return Response(
        upstream.iter_content(chunk_size=8192),
        content_type=upstream.headers.get("Content-Type", "image/png"),
        headers={
            "Access-Control-Allow-Origin": "*",
            "Cache-Control": "public, max-age=3600",
        },
    )


@app.route("/api/event", methods=["POST"])
def api_event():
    data = request.get_json(silent=True) or {}
    rec_id = data.get("recommendation_id")
    event_type = data.get("event_type")
    session_id = data.get("session_id")
    auth_user = _current_user(request)
    user_id = auth_user["id"] if auth_user else None
    if not event_type:
        return jsonify({"ok": False, "error": "missing event_type"}), 400
    events.append({"recommendation_id": rec_id, "event_type": event_type})
    db.save_event(
        recommendation_id=rec_id,
        session_id=session_id,
        event_type=event_type,
        user_id=user_id,
    )
    return jsonify({"ok": True})


@app.route("/api/reaction", methods=["POST"])
def api_reaction():
    """Record 👍 / 👎 / cleared reaction on a recommendation.
    value: 1 = helped, -1 = didn't help, 0 = clear. Auth-optional."""
    data = request.get_json(silent=True) or {}
    rec_id = data.get("recommendation_id")
    raw_value = data.get("value")
    session_id = data.get("session_id")
    auth_user = _current_user(request)
    user_id = auth_user["id"] if auth_user else None
    if not rec_id:
        return jsonify({"ok": False, "error": "missing recommendation_id"}), 400
    if raw_value not in (1, -1, 0):
        return jsonify({"ok": False, "error": "value must be 1, -1, or 0"}), 400
    stored_value = raw_value if raw_value in (1, -1) else None
    event_type = {1: "reaction_up", -1: "reaction_down", 0: "reaction_clear"}[raw_value]
    db.set_reaction(rec_id, stored_value)
    db.save_event(
        recommendation_id=rec_id,
        session_id=session_id,
        event_type=event_type,
        user_id=user_id,
    )
    return jsonify({"ok": True})


@app.route("/api/config")
def api_config():
    """Surface non-secret config to the SPA. The publishable key is meant to be
    public (Clerk's docs call it 'pk_test_...' / 'pk_live_...') — it identifies
    the Clerk instance but doesn't authorize backend access.
    Accepts NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY as a fallback because Clerk's
    quickstart docs use that name and people often copy it verbatim."""
    pk = (
        os.environ.get("CLERK_PUBLISHABLE_KEY")
        or os.environ.get("NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY")
        or ""
    )
    return jsonify({"clerk_publishable_key": pk})


@app.route("/api/auth/migrate", methods=["POST"])
def api_auth_migrate():
    """Claim anonymous entries for the now-signed-in user.

    Two claim strategies (we use BOTH when both are available):
    1. session_id (`anon_id`) — bulk-claims every row tagged with this anon
       browser. Best-effort; fails silently if the session_id was wiped or
       never persisted (Safari ITP, cleared storage, server restarted, etc).
    2. `pending_rec_id` — explicit claim of one specific row. The frontend
       stashes the just-served recommendation_id to localStorage BEFORE the
       OAuth redirect, so even when the session_id approach finds nothing,
       the song the user signed up to save is reliably preserved.
    """
    auth_user = _current_user(request)
    if not auth_user or not auth_user.get("id"):
        return jsonify({"ok": False, "error": "not authenticated"}), 401
    data = request.get_json(silent=True) or {}
    anon_id = (data.get("anon_id") or "").strip() or None
    pending_rec_id = (data.get("pending_rec_id") or "").strip() or None
    user_id = auth_user["id"]
    provider = _get_primary_provider(user_id)
    db.upsert_user(
        user_id=user_id,
        email=auth_user.get("email"),
        primary_provider=provider,
        signup_anon_id=anon_id,
    )
    moved = db.migrate_anon_to_user(user_id, anon_id) if anon_id else {"recommendations": 0, "events": 0}
    pending_claimed = False
    if pending_rec_id:
        pending_claimed = db.claim_recommendation(pending_rec_id, user_id)
        if pending_claimed:
            # Count it toward the migration total so the frontend's "saved to
            # your journal ✓" toast fires for the just-preserved song.
            moved["recommendations"] = max(moved.get("recommendations", 0), 1)
    return jsonify({
        "ok": True,
        "migrated": moved,
        "pending_claimed": pending_claimed,
        "primary_provider": provider,
    })


@app.route("/api/journal")
def api_journal():
    """Reverse-chron timeline of the signed-in user's entries.

    Each entry is shaped to plug straight into the SPA's renderPick() — same
    field names as /api/recommend's response — so tapping a row replays the
    song without an extra round-trip.

    Cursor pagination: pass ?before=<iso_ts> with the prior page's last
    created_at to fetch the next page. ?limit caps at 50.
    """
    auth_user = _current_user(request)
    if not auth_user or not auth_user.get("id"):
        return jsonify({"ok": False, "error": "not authenticated"}), 401
    user_id = auth_user["id"]
    before_iso = request.args.get("before") or None
    try:
        limit = min(int(request.args.get("limit", 20)), 50)
    except ValueError:
        limit = 20
    rows = db.fetch_journal(user_id=user_id, limit=limit, before_iso=before_iso)
    # Rename DB column id → recommendation_id so the journal payload matches
    # exactly what /api/recommend returns. Same for safety flag (always False
    # in journal results because fetch_journal filters them out).
    entries = []
    for r in rows:
        entries.append({
            "recommendation_id": r.get("id"),
            "created_at": r.get("created_at"),
            "prompt": r.get("prompt"),
            "song_title": r.get("song_title"),
            "artist": r.get("artist"),
            "one_liner": r.get("one_liner"),
            "key_lyric": r.get("key_lyric"),
            "iso_reasoning": r.get("iso_reasoning"),
            "album_art_url": r.get("album_art_url"),
            "preview_url": r.get("preview_url"),
            "track_view_url": r.get("track_view_url"),
            "streaming_links": r.get("streaming_links") or {},
            "plain_lyrics": None,  # lyrics aren't cached in recommendations row
            "helped": r.get("helped"),
            "detected_language": r.get("detected_language"),
            "notes": r.get("notes"),
            "safety": False,
        })
    next_cursor = entries[-1]["created_at"] if len(entries) == limit else None
    return jsonify({"entries": entries, "next_cursor": next_cursor})


@app.route("/api/journal/<rec_id>", methods=["PATCH"])
def api_journal_patch(rec_id):
    """Update the user-written notes on a single entry. Body: {notes: str|null}.
    Returns 401 if unauth, 404 if no row matched the (rec_id, user_id) pair —
    same response so we don't leak whether the row exists for another user."""
    auth_user = _current_user(request)
    if not auth_user or not auth_user.get("id"):
        return jsonify({"ok": False, "error": "not authenticated"}), 401
    data = request.get_json(silent=True) or {}
    notes_raw = data.get("notes")
    if notes_raw is not None and not isinstance(notes_raw, str):
        return jsonify({"ok": False, "error": "notes must be a string or null"}), 400
    # Cap at a sensible length so a hostile client can't dump megabytes.
    notes = notes_raw.strip() if isinstance(notes_raw, str) else None
    if notes is not None and len(notes) > 4000:
        notes = notes[:4000]
    if notes == "":
        notes = None  # treat empty string as cleared note
    ok = db.update_entry_note(rec_id, auth_user["id"], notes)
    if not ok:
        return jsonify({"ok": False, "error": "not found"}), 404
    return jsonify({"ok": True, "notes": notes})


@app.route("/api/journal/<rec_id>/save", methods=["POST"])
def api_journal_save(rec_id):
    """Explicit save-to-journal action. Claims this specific recommendation
    row for the signed-in user. Idempotent — re-claiming a row the user
    already owns is a no-op success.

    The same DB primitive (claim_recommendation) is used by /api/auth/migrate
    for the post-signup handoff, so behavior stays consistent."""
    auth_user = _current_user(request)
    if not auth_user or not auth_user.get("id"):
        return jsonify({"ok": False, "error": "not authenticated"}), 401
    user_id = auth_user["id"]
    claimed = db.claim_recommendation(rec_id, user_id)
    # claim_recommendation only matches user_id IS NULL rows. If the user
    # already owns it (re-tap), the row stays untouched and we still return ok.
    return jsonify({"ok": True, "claimed": claimed})


@app.route("/api/journal/<rec_id>", methods=["DELETE"])
def api_journal_delete(rec_id):
    """Soft-delete a journal entry. The row stays in Postgres with deleted_at
    set; journal fetch + result-screen replay both filter it out."""
    auth_user = _current_user(request)
    if not auth_user or not auth_user.get("id"):
        return jsonify({"ok": False, "error": "not authenticated"}), 401
    ok = db.soft_delete_entry(rec_id, auth_user["id"])
    if not ok:
        return jsonify({"ok": False, "error": "not found"}), 404
    return jsonify({"ok": True})


# ── Synced lyrics via Whisper ─────────────────────────────────────────────────
# Lazy: only runs when the user actually taps play. Result cached per-rec-id so
# replays are instant. LRU-bounded so memory stays predictable across many recs.
sync_cache: "OrderedDict[str, dict]" = OrderedDict()
SYNC_CACHE_MAX = 100


def _match_transcript_to_lyrics(whisper_words: list, plain_lyrics: str) -> list[dict]:
    """Best-anchor sequential matcher.

    iTunes previews can start anywhere in the song — sometimes at the chorus,
    sometimes at verse 2, sometimes at the song's beginning. A naive
    sequential matcher starting at line 0 fails when the preview actually
    starts later, because its cursor never advances to the chorus.

    This matcher tries every plausible starting position in the lyrics
    (anywhere one of the first several transcribed words appears) and picks
    the position that produces the most matches. Then it runs sequentially
    from there to build the final mapping.

    Returns [{"t": float, "line": int}, ...] sorted by t.
    """
    LOOK_AHEAD = 25
    SUBSTR_MIN = 4

    # Tokenize lyrics into (line_idx, normalized_word) tuples
    lines = plain_lyrics.split("\n")
    lyric_tokens: list[tuple[int, str]] = []
    for line_idx, line in enumerate(lines):
        for word in re.findall(r"[A-Za-z0-9']+", line):
            normalized = re.sub(r"[^a-z0-9]", "", word.lower())
            if normalized:
                lyric_tokens.append((line_idx, normalized))

    # Normalize Whisper words into (t, normalized_word) tuples
    whisper_tokens: list[tuple[float, str]] = []
    for w in whisper_words:
        t = w.get("start")
        if t is None:
            continue
        word = re.sub(r"[^a-z0-9]", "", str(w.get("word", "")).lower())
        if word:
            whisper_tokens.append((float(t), word))

    if not lyric_tokens or not whisper_tokens:
        return []

    def matches_at(a: str, b: str) -> bool:
        if a == b:
            return True
        if len(a) >= SUBSTR_MIN and a in b:
            return True
        if len(b) >= SUBSTR_MIN and b in a:
            return True
        return False

    def run_match(start_idx: int, record: bool = False):
        """Walk whisper_tokens against lyric_tokens starting at start_idx.
        Returns (match_count, line_first_t) — line_first_t only populated
        when record=True (counting-only path is faster on the hot loop)."""
        cursor = start_idx
        count = 0
        line_first_t: dict[int, float] = {}
        for t, word in whisper_tokens:
            end = min(cursor + LOOK_AHEAD, len(lyric_tokens))
            for i in range(cursor, end):
                lyric_line, lyric_word = lyric_tokens[i]
                if matches_at(word, lyric_word):
                    count += 1
                    if record:
                        if lyric_line not in line_first_t or t < line_first_t[lyric_line]:
                            line_first_t[lyric_line] = t
                    cursor = i + 1
                    break
        return count, line_first_t

    # Build a word index: lyric token index by normalized word
    word_to_indexes: dict[str, list[int]] = {}
    for i, (_, w) in enumerate(lyric_tokens):
        word_to_indexes.setdefault(w, []).append(i)

    # Candidate starting positions: any lyric index whose word matches one
    # of the first ~8 transcribed words (or matches via substring rule).
    candidate_starts: set[int] = {0}  # always try the song's beginning
    LEAD = 8
    for _, word in whisper_tokens[:LEAD]:
        # Exact-word hits via index
        if word in word_to_indexes:
            candidate_starts.update(word_to_indexes[word])
        # Fuzzy: only if substring rule could fire (saves O(N) per word otherwise)
        if len(word) >= SUBSTR_MIN:
            for lw, idxs in word_to_indexes.items():
                if word in lw or lw in word:
                    candidate_starts.update(idxs)

    # Pick the starting position with the most matches. Tie-break to earliest
    # position so we prefer the song's natural ordering when scores are equal.
    best_start = min(candidate_starts, key=lambda i: (-run_match(i)[0], i))

    # Re-run with best start to build the mapping
    _, line_first_t = run_match(best_start, record=True)

    sparse = sorted(
        [{"t": t, "line": line_idx} for line_idx, t in line_first_t.items()],
        key=lambda x: x["t"],
    )

    # Smooth the sparse Whisper mapping into a dense line-by-line walk-forward.
    # The matcher's sparse output causes the frontend highlight to JUMP between
    # matched lines (skipping unmatched intermediates). Instead: use the first
    # few matches to infer a per-line cadence, then advance one line at a time
    # at that cadence until the preview ends. This trades Whisper's per-line
    # precision (which drifts on its own anyway) for smooth, jump-free UX.
    num_lines = len(lines)
    return _smooth_to_line_by_line(sparse, num_lines, preview_duration=30.0)


def _smooth_to_line_by_line(
    sparse: list[dict],
    num_lines: int,
    preview_duration: float = 30.0,
) -> list[dict]:
    """Turn a sparse Whisper-derived mapping into a dense one-line-per-step walk.

    Use the median delta between consecutive matched lines to infer the song's
    per-line cadence, then walk forward from the first match through every line
    at that cadence until time runs past the preview's end.

    If we don't have at least 2 matches to infer cadence from, or the inferred
    cadence is implausible (very short / very long sung lines), fall back to the
    sparse mapping.
    """
    if len(sparse) < 2:
        return sparse

    deltas: list[float] = []
    for i in range(len(sparse) - 1):
        dline = sparse[i + 1]["line"] - sparse[i]["line"]
        dt = sparse[i + 1]["t"] - sparse[i]["t"]
        if dline > 0 and dt > 0:
            deltas.append(dt / dline)

    if not deltas:
        return sparse

    deltas.sort()
    cadence = deltas[len(deltas) // 2]  # median is robust to one bad measurement

    # Sanity-clamp: typical sung lines are between half a second and ~8 seconds.
    # Outside that, the cadence inference is suspect — keep the sparse mapping.
    if cadence < 0.5 or cadence > 8.0:
        return sparse

    anchor = sparse[0]
    dense: list[dict] = []
    for line_idx in range(anchor["line"], num_lines):
        t = anchor["t"] + (line_idx - anchor["line"]) * cadence
        if t > preview_duration:
            break
        dense.append({"t": round(t, 2), "line": line_idx})
    return dense


def _transcribe_preview(preview_url: str) -> list[dict]:
    """Download the iTunes preview and transcribe with Whisper.
    Returns the list of word-level dicts from Whisper's verbose_json response."""
    audio = requests.get(preview_url, timeout=10)
    audio.raise_for_status()
    buf = io.BytesIO(audio.content)
    buf.name = "preview.m4a"  # Whisper API uses the filename for format detection
    result = client.audio.transcriptions.create(
        model="whisper-1",
        file=buf,
        response_format="verbose_json",
        timestamp_granularities=["word"],
    )
    # The SDK returns an object with .words; convert to plain dicts for JSON safety
    raw_words = getattr(result, "words", None) or []
    return [
        {"word": w.word if hasattr(w, "word") else w.get("word", ""),
         "start": w.start if hasattr(w, "start") else w.get("start"),
         "end": w.end if hasattr(w, "end") else w.get("end")}
        for w in raw_words
    ]


def _cache_sync_result(rec_id: str, payload: dict) -> None:
    sync_cache[rec_id] = payload
    sync_cache.move_to_end(rec_id)
    while len(sync_cache) > SYNC_CACHE_MAX:
        sync_cache.popitem(last=False)


@app.route("/api/sync-lyrics", methods=["POST"])
def api_sync_lyrics():
    data = request.get_json(silent=True) or {}
    rec_id = (data.get("recommendation_id") or "").strip()
    if not rec_id:
        return jsonify({"status": "error", "error": "missing recommendation_id"}), 400

    # Return cached result if we already transcribed this recommendation
    if rec_id in sync_cache:
        sync_cache.move_to_end(rec_id)
        return jsonify(sync_cache[rec_id])

    rec = recommendations.get(rec_id)
    if not rec:
        return jsonify({"status": "error", "error": "recommendation not found"}), 404

    preview_url = rec.get("preview_url")
    plain_lyrics = rec.get("plain_lyrics")

    # No lyrics at all → instrumental or unsourced. Frontend should already be
    # hiding the panel, but be defensive and return a clear status.
    if not plain_lyrics:
        payload = {"status": "instrumental", "mapping": []}
        _cache_sync_result(rec_id, payload)
        return jsonify(payload)

    if not preview_url:
        payload = {"status": "no_preview", "mapping": []}
        _cache_sync_result(rec_id, payload)
        return jsonify(payload)

    # Transcribe and match
    try:
        words = _transcribe_preview(preview_url)
    except Exception as e:
        print(f"  whisper failed for {rec_id}: {type(e).__name__}: {e}")
        payload = {"status": "transcription_failed", "mapping": []}
        _cache_sync_result(rec_id, payload)
        return jsonify(payload)

    mapping = _match_transcript_to_lyrics(words, plain_lyrics)
    status = "matched" if mapping else "no_match"
    payload = {"status": status, "mapping": mapping}
    _cache_sync_result(rec_id, payload)
    print(f"  sync: {rec_id} → {status} ({len(mapping)} lines mapped, {len(words)} words)")
    return jsonify(payload)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5001"))
    app.run(host="0.0.0.0", port=port, debug=True)
