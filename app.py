#!/usr/bin/env python3
"""
#Mood backend — Flask wrapper around engine.recommend()
"""
import os
import re
import uuid
from pathlib import Path
from urllib.parse import quote_plus
import requests
from flask import Flask, request, jsonify, send_from_directory, Response
from flask_cors import CORS
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

from engine import recommend  # noqa: E402

app = Flask(__name__, static_folder="static")
CORS(app)

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


def lookup_itunes_track(song_title: str, artist: str) -> dict:
    """Look up a track on iTunes, return {art_url, preview_url, track_view_url}."""
    try:
        resp = requests.get(
            "https://itunes.apple.com/search",
            params={
                "term": f"{song_title} {artist}",
                "entity": "song",
                "limit": 1,
            },
            timeout=6,
        )
        results = resp.json().get("results", [])
        if not results:
            return {}
        r = results[0]
        art = r.get("artworkUrl100", "")
        return {
            "art_url": art.replace("100x100bb.jpg", "1000x1000bb.jpg") or None,
            "preview_url": r.get("previewUrl"),
            "track_view_url": r.get("trackViewUrl"),
        }
    except Exception as e:
        print(f"  iTunes lookup failed: {type(e).__name__}: {e}")
        return {}


def pick_album_assets(one_liner: str, iso_reasoning: str,
                      song_title: str, artist: str) -> dict:
    """Return {album_art_url, preview_url, track_view_url}. Falls back to cat pool."""
    track = lookup_itunes_track(song_title, artist)
    if track.get("art_url"):
        print(f"  iTunes hit: {song_title} (preview: {bool(track.get('preview_url'))})")
        return {
            "album_art_url": track["art_url"],
            "preview_url": track.get("preview_url"),
            "track_view_url": track.get("track_view_url"),
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

    if not prompt:
        return jsonify({"error": "empty prompt"}), 400

    try:
        result = recommend(prompt, used_songs=used_songs, category=category)
    except Exception as e:
        return jsonify({"error": f"engine failed: {e}"}), 500

    if result.get("safety"):
        return jsonify({"safety": True, "message": result["message"]})

    rec_id = str(uuid.uuid4())
    assets = pick_album_assets(
        result["one_liner"],
        result.get("iso_reasoning", ""),
        result["song_title"],
        result["artist"],
    )

    # Streaming deep links. Apple Music gets the direct track URL when iTunes
    # returns one; Spotify and YouTube Music use search URLs that land on the
    # right track without needing OAuth.
    q = quote_plus(f"{result['artist']} {result['song_title']}")
    streaming_links = {
        "apple_music": assets.get("track_view_url"),
        "spotify":     f"https://open.spotify.com/search/{q}",
        "youtube":     f"https://music.youtube.com/search?q={q}",
    }

    full = {
        "one_liner": result["one_liner"],
        "song_title": result["song_title"],
        "artist": result["artist"],
        "key_lyric": result["key_lyric"],
        "iso_reasoning": result.get("iso_reasoning", ""),
        "album_art_url": assets["album_art_url"],
        "preview_url": assets.get("preview_url"),
        "track_view_url": assets.get("track_view_url"),
        "streaming_links": streaming_links,
        "recommendation_id": rec_id,
        "safety": False,
    }
    recommendations[rec_id] = full
    sessions.setdefault(session_id, []).append(
        f"{result['song_title']} by {result['artist']}"
    )
    return jsonify(full)


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
    events.append({
        "recommendation_id": data.get("recommendation_id"),
        "event_type": data.get("event_type"),
    })
    return jsonify({"ok": True})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5001"))
    app.run(host="0.0.0.0", port=port, debug=True)
