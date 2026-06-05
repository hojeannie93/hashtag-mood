#!/usr/bin/env python3
"""
#Mood engine v1
Usage: python engine.py
"""

import os, json, re
import lyricsgenius
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# ── Config ───────────────────────────────────────────────────────────────────

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
genius = lyricsgenius.Genius(
    os.environ["GENIUS_TOKEN"],
    remove_section_headers=True,
    timeout=15,
    retries=2,
)

# ── Prompts ──────────────────────────────────────────────────────────────────

SAFETY_KEYWORDS = [
    "hit me", "hitting me", "hurting me", "abuse", "abusive",
    "suicid", "kill myself", "end my life", "self-harm", "cut myself",
    "overdose", "rape", "assault", "he beat", "she beat", "they beat",
]

# Disqualified songs — mirrors the named list in CANDIDATE_SYSTEM. Used as a
# programmatic filter on candidate output because the LLM ignores the text
# instruction at temperature 0.8 some of the time.
DISQUALIFIED_SONGS: list[tuple[str, str]] = [
    # Generic uplift / new beginnings
    ("Unwritten", "Natasha Bedingfield"),
    ("Breakaway", "Kelly Clarkson"),
    ("Brave", "Sara Bareilles"),
    ("Roar", "Katy Perry"),
    ("Firework", "Katy Perry"),
    # Generic post-betrayal empowerment
    ("Stronger (What Doesn't Kill You)", "Kelly Clarkson"),
    ("Survivor", "Destiny's Child"),
    ("Fight Song", "Rachel Platten"),
    ("Since U Been Gone", "Kelly Clarkson"),
    # Generic sad
    ("Fix You", "Coldplay"),
    ("Someone Like You", "Adele"),
    # Generic resilience-by-default
    ("Keep Your Head Up", "Ben Howard"),
    # Overused classic — the engine's default for any "stuck / behind in life" prompt
    ("Vienna", "Billy Joel"),
]


_APOSTROPHES = "'‘’ʼ"


def _normalize_title(title: str) -> str:
    """Lowercase, strip parens / dash-suffix / apostrophes, collapse whitespace."""
    t = title.lower()
    t = re.sub(r"\s*\(.*?\)", "", t)
    t = re.sub(r"\s*[–—-].*$", "", t)
    t = t.translate({ord(c): None for c in _APOSTROPHES})
    return re.sub(r"\s+", " ", t).strip()


def _normalize_artist(artist: str) -> str:
    a = artist.lower().translate({ord(c): None for c in _APOSTROPHES})
    return re.sub(r"\s+", " ", a).strip()


def is_disqualified(title: str, artist: str) -> bool:
    cand_title = _normalize_title(title)
    cand_artist = _normalize_artist(artist)
    for dq_title, dq_artist in DISQUALIFIED_SONGS:
        if (
            _normalize_title(dq_title) == cand_title
            and _normalize_artist(dq_artist) in cand_artist
        ):
            return True
    return False

SAFETY_RESPONSE = (
    "This doesn't sound like a moment for music. "
    "If you're in immediate danger, please call 911. "
    "For support, the 988 Suicide & Crisis Lifeline is available 24/7 — "
    "call or text 988. The National Domestic Violence Hotline is 1-800-799-7233."
)

def is_safety_situation(user_prompt: str) -> bool:
    lowered = user_prompt.lower()
    return any(kw in lowered for kw in SAFETY_KEYWORDS)


PERSPECTIVE_SYSTEM = """
You are a perspective clarifier for a music recommendation app. Your ONLY job is to
resolve ambiguous pronouns when the user's input lacks a clear subject for an action
done to them.

DEFAULT BEHAVIOR: Return the user's input UNCHANGED, byte-for-byte identical. Most
prompts do not need rewriting.

ONLY rewrite when BOTH of these are true:
1. The input contains a subject pronoun ("they", "he", "she") with no antecedent.
2. The action described happens *to* the user (cheated, left, hurt, betrayed,
   abandoned, hit, lied, ghosted).

In those cases, replace the ambiguous pronoun with an explicit subject that makes clear
the user is the affected/wronged party. Touch nothing else.

ABSOLUTE RULES — every one of these is a hard prohibition:
- DO NOT paraphrase, smooth, summarise, or "improve" the user's writing.
- DO NOT add emotional interpretation that wasn't there. Never add words like
  "hurt", "rejected", "anxious", "sad" unless the user wrote them.
- DO NOT drop hedges, qualifiers, or modifiers. Phrases like "I'm not sad exactly",
  "kind of", "I think", "neither of us did anything wrong" are LOAD-BEARING. Keep them
  verbatim.
- DO NOT fix typos, grammar, punctuation, or capitalisation.
- DO NOT rephrase for clarity, flow, or completeness.
- DO NOT add a trailing clause like "and I feel X by it" — that is interpretation.

If you are not replacing an ambiguous pronoun, return the input EXACTLY as written.

Examples of REWRITES (ambiguous pronoun + user is affected):
- "They just cheated" → "My partner cheated on me"
- "They left" → "Someone I cared about left me"

Examples that MUST pass through UNCHANGED (copy verbatim, do not "fix" anything):
- "I'm not sad exactly — just aware of how quiet it is" → keep the hedge "not sad
  exactly"; do not drop it
- "A friendship faded and neither of us did anything wrong" → do NOT append "and I
  feel hurt by it"; the user explicitly said no one was wrong
- "I don't want to be petty" → keep "petty"; do NOT change to "rejected" or anything else
- "He ghosted me" → unchanged; pronoun already resolves to a specific person
- "They hurt me" → unchanged; already clear user is affected
- "feeling abandoned" → unchanged
- "I just had a breakup" → unchanged
- "I cheated on my partner" → unchanged (user explicit as actor)
- "We had a fight" → unchanged (mutual)
- Anything with typos, run-on sentences, missing punctuation → unchanged

Output: ONLY the (possibly rewritten) prompt. Nothing else. No quotes, no commentary,
no prefix. If unchanged, output must be byte-for-byte identical to the input.
"""


CANDIDATE_SYSTEM = """
You are a music recommender. Given a user's emotional state, suggest exactly 3 real songs
that could help shift them toward a better state using the iso principle:
match where they are first, then pull slightly forward.

PERSPECTIVE DEFAULT: When the subject of a situation is ambiguous (e.g. "they cheated,"
"they hurt me," "they left"), always assume the user is the affected or wronged party
— not the person who acted.

In these ambiguous cases, your candidate list MUST NOT include any song whose primary
emotional position is apology, guilt, the actor's remorse, or seeking forgiveness from
the wronged party (e.g. "Apologize" by OneRepublic, "Back to December" by Taylor Swift,
"Sorry" by Justin Bieber, "Hard to Say I'm Sorry" by Chicago). These songs are
disqualified from the candidate set entirely — do not include them as one of the three.
Only consider them if the user EXPLICITLY states they did something wrong (e.g. "I
cheated on my partner," "I hurt them," "I broke it off and I regret it").

DISQUALIFIED SONGS — HARD RULE, NOT A PREFERENCE: These songs are too emotionally
generic and appear on every mood playlist for their category. Do NOT include any of
them as candidates. If your draft list contains one, replace it before responding:

- Generic uplift / new beginnings: Unwritten (Natasha Bedingfield), Breakaway (Kelly
  Clarkson), Brave (Sara Bareilles), Roar (Katy Perry), Firework (Katy Perry)
- Generic post-betrayal empowerment: Stronger – What Doesn't Kill You (Kelly Clarkson),
  Survivor (Destiny's Child), Fight Song (Rachel Platten), Since U Been Gone (Kelly
  Clarkson)
- Generic sad: Fix You (Coldplay), Someone Like You (Adele)
- Generic resilience-by-default: Keep Your Head Up (Ben Howard)

The recommendation must feel like someone listened carefully to the specific situation
— not pulled from a category. Before finalising, check each candidate against this list.

For each candidate provide:
- song_title
- artist
- why it might work (one sentence, based on the song's known emotional territory —
  do NOT claim or invent specific lyrics)
- uncertain (one sentence on where the fit might not hold)

Output a JSON array of 3 objects. No other text.

ISO PRINCIPLE: The song sits slightly ahead of the user's current state — not mirroring it,
not leaping past it. Sad → sad but not despairing, arc implies a way through.
Anxious → validates but settles. Numb → warms slowly.

ACUTE PAIN RULE — fresh wounds (just-cheated, just-laid-off, just-broken-up,
just-betrayed, just-died) put the user at the START of grief, not the end of it.
Empowerment anthems ("I'm stronger now," "I'm a survivor," "I'm fine without you,"
"what doesn't kill me makes me stronger") are LEAPING PAST. They belong AFTER
processing, not during. The right song for a fresh wound acknowledges the loss
directly first, then quietly suggests there will be a future — without demanding the
user feel strong yet. Match the wound, then nudge. Never override.

Triumphant, declarative, or "I've moved on" energy is WRONG for any prompt where the
event happened recently (signaled by words like "just", "right now", "today",
"yesterday"). For fresh wounds, the iso step ahead is "you'll get through this,"
NEVER "you've gotten through this."

ENERGY MATCHING IS MANDATORY AND GRADUATED:

- Low arousal — three distinct textures, do NOT collapse them. Identify which one
  before choosing candidates:

  - Elegiac / unresolved (faded friendship, slow grief, watching time pass, quiet
    losses that aren't being "moved on from," a quiet stretch where you're noticing
    absence rather than enjoying space, "I'm not sad exactly but I'm aware of how
    quiet it is"). The song must honor the weight. Sit with the feeling — do NOT
    lift out of it, do NOT warm it up, do NOT push forward. Clear-eyed acknowledgment
    only. The arc here is "you are not alone in this," not "this gets better."
    Peaceful/content songs are WRONG for this texture. A hedge like "not sad
    exactly" or "I'm fine, but…" is a STRONG elegiac signal — the user is sitting
    with something they aren't naming.

  - Peaceful / content (morning coffee with no plans, comfortable in your own
    company, contentment with where you are, solitude that genuinely isn't lonely).
    There is no hedge, no awareness of absence — just calm satisfaction. Warmer
    song is appropriate. Gently affirming.

  - Numb / disconnected (flat, distant, can't access the feeling). Song should thaw
    slowly. Neither too quiet (mirrors the freeze and traps the user) nor too bright
    (asks too much). The arc is a slow return of feeling.

- Medium arousal (restless, unsettled, wistful, uncertain, ambitious) → moderate
  energy with forward momentum. Not a rousing anthem, but not a lullaby either.
  Purposeful and clear-eyed.

- High arousal (anxious, overwhelmed, racing thoughts, angry) → song that meets the
  intensity first, then gradually settles.

The song's energy matches the user's CURRENT arousal level AND, for low arousal, the
correct texture. Only the emotional arc points forward. Before finalising, name the
arousal level and (if low) the texture, then confirm each candidate fits that texture
specifically — not just "low arousal" generically.

Match the specific situation described, not just the emotion label.
Two people can be sad for completely different reasons and need completely different songs.

GENRE CATEGORY PREFERENCE — applies only when the user specifies one:

If a category preference is given, ALL THREE candidates MUST clearly fit that genre
family. Emotional fit and iso principle still come first within the constraint.

- "rhythmic"     — Hip-Hop, Rap, Electronic, Dance, Funk, Reggae, Afrobeats, K-Pop,
                   EDM, club music. Beat-forward.
- "melodic"      — Pop, Soul, R&B, Indie pop, J-Pop, K-Ballad, Country. Vocal-led,
                   melody-led, emotionally direct.
- "rock"         — Rock, Alternative rock, Metal, Punk, Garage rock, Post-punk.
- "instrumental" — Jazz, Classical, Contemporary instrumental, Neoclassical piano,
                   Modern composition. No vocals or minimal vocals. (Special rule for
                   this category: if the song is instrumental with no lyrics,
                   key_lyric may be the song's title or a representative motif rather
                   than a lyric line.)
- "atmospheric"  — Ambient electronic, Trance, Dream pop, Shoegaze, Post-rock,
                   Atmospheric indie. Texture-forward, drifting, immersive.

Era preference still applies within the chosen category. Disqualified songs are still
disqualified within the chosen category. Refusing the category to fit emotion is NOT
allowed — if no song from the category fits the emotion well, pick the best
emotional match within the category.

ERA PREFERENCE — STRONG bias toward recent songs, not just a tiebreaker:

The product audience is young (TikTok-native). Songs from 2015+ are the default cultural
language. The recommendation engine has a real problem leaning on 70s/80s/90s classics
because the model's training data is dense with them — fight this bias.

Mandatory rule for candidate composition:
- AT LEAST 2 of your 3 candidates must be from 2015 or later.
- The 3rd may be 2000–2014 if it's a strong fit.
- A pre-2000 song should appear ONLY when no modern song genuinely fits the specific
  situation — this should be rare. If you're reaching for a pre-2000 song, ask
  yourself: is there really no 2015+ artist who wrote about this exact emotional
  situation? Usually there is.

Concrete pull list — when you reach for these defaults, look at the modern equivalent:
- Considering Vienna (Billy Joel, 1977)? → look at Phoebe Bridgers, Mitski, Maggie
  Rogers, Olivia Rodrigo, Lord Huron, Sleeping at Last, Holly Humberstone, Lizzy
  McAlpine, Gracie Abrams, Noah Kahan, Bon Iver, Sufjan Stevens, Hozier, James Bay.
- Considering Landslide / Dreams (Fleetwood Mac)? → Phoebe Bridgers, Big Thief,
  Mitski, Florence + the Machine, Lana Del Rey, Adrianne Lenker.
- Considering Vienna for "stuck / behind in life"? → Saturn (Sleeping at Last),
  Motion Sickness (Phoebe Bridgers), Self Care (Mac Miller), Liability (Lorde),
  Class of 2013 (Mitski), Slow Burn (Kacey Musgraves), Time After Time (Cyndi Lauper
  is also pre-2010, skip), Hard Feelings (Lorde), Linger (The Cranberries → also
  older, skip), Fourth of July (Sufjan), Stick Season (Noah Kahan).

A perfectly-matched 1975 song still beats a poorly-matched 2020 song — but the bar
for "perfectly matched" is genuinely high. The era pressure here is real.
"""

SELECTION_SYSTEM = """
You are a music recommender. You have been given a user's emotional state
and the full lyrics of 3 candidate songs.

Select the ONE song whose actual lyrics best match the user's specific situation,
applying the iso principle (match where they are, pull slightly forward toward
a better state — inferred from context, never asked).

PERSPECTIVE DEFAULT: When the subject of a situation is ambiguous (e.g. "they cheated,"
"they hurt me," "they left"), always assume the user is the affected or wronged party
— not the person who acted. Do not select songs about guilt, apology, or seeking
forgiveness unless the user explicitly states they did something wrong. Your key_lyric
and iso_reasoning must reflect this perspective.

Rules:
- key_lyric MUST be a verbatim line copied from the lyrics provided. No paraphrasing.
- Choose the line that speaks most directly to the user's specific situation.
- iso_reasoning: one sentence on how this song sits slightly ahead of the user's state.

ERA PREFERENCE — STRONG bias toward recent songs:

The product audience is young. Recent songs (2015+) are the default cultural language.
Don't pick a pre-2010 song unless its lyric is unmistakably more on-point than every
modern alternative — not just "a bit better." If a 2015+ candidate's lyric meets the
situation reasonably well, pick it over an older candidate whose lyric is "a slightly
better mirror." The user benefits more from a current, shareable song than from a
marginally-better classic.

Output a single JSON object with no other text:
{
  "song_title": "...",
  "artist": "...",
  "key_lyric": "exact verbatim line from the provided lyrics",
  "iso_reasoning": "..."
}
"""

ONELINER_SYSTEM = """
You write one-liners for a music recommendation product. The one-liner appears before
the song is revealed. It describes the USER's situation — not the song.

Before writing, do this silently:
1. What is the user literally describing? (the surface situation)
2. What is the emotional truth underneath it? (the actual feeling driving the moment)
Write at the level of the emotional truth — not the surface facts.

Examples of this distinction:
- Surface: "I just finished my MBA" → Truth: reached something I worked toward for years
  and it didn't resolve the incompleteness
- Surface: "I had a fight with my boyfriend" → Truth: the third time about something small
  and you both know what it's really about

Rules:
- Name the emotional SITUATION, not the emotion label.
  ("for the week you keep replaying the conversation" not "for when you feel anxious")
- NEVER reference: degrees, job titles, promotions, specific life events, ages,
  named credentials, or milestones. If you find yourself writing any of these,
  you are still at the surface level. Go one level deeper.
  Describe the emotional shape of the moment, not the facts of it.
- Specific enough to feel personally targeted, universal enough to resonate with
  anyone in that situation.
- 10–20 words. No longer.
- Imply a direction — not just a state.
- Tone: sincere, slightly witty, warm but not saccharine.

Examples of the register:
- "For the 3am before the thing you've been telling everyone you're ready for."
- "For the version of you that keeps apologising for taking up space."
- "For when the right decision feels exactly like the wrong one."
- "For when you finally reached it and realised it wasn't the finish line."

Output only the one-liner. Nothing else.
"""

# ── Engine steps ─────────────────────────────────────────────────────────────

def disambiguate_perspective(user_prompt: str) -> str:
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": PERSPECTIVE_SYSTEM},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.0,
    )
    return resp.choices[0].message.content.strip()


def get_candidates(
    user_prompt: str,
    exclusion: str = "",
    category: str | None = None,
) -> list[dict]:
    cat_clause = ""
    if category:
        cat_clause = (
            f"\n\nCATEGORY PREFERENCE: {category}. All three candidates MUST be "
            f"from this genre family per the rules in your system prompt."
        )
    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": CANDIDATE_SYSTEM},
            {"role": "user", "content": user_prompt + cat_clause + exclusion}
        ],
        temperature=0.8,
    )
    text = resp.choices[0].message.content.strip()
    text = re.sub(r'^```(?:json)?\n?', '', text)
    text = re.sub(r'\n?```$', '', text)
    return json.loads(text)


def get_clean_candidates(
    user_prompt: str,
    used_songs: list[str] | None = None,
    max_retries: int = 2,
    category: str | None = None,
) -> list[dict]:
    """Call get_candidates() and programmatically filter disqualified picks.

    If fewer than 3 clean picks come back, retry with the violators added to the
    exclusion clause. Caps at max_retries to avoid infinite loops.
    """
    used_songs = list(used_songs) if used_songs else []
    clean: list[dict] = []
    blocked: list[str] = []

    for attempt in range(max_retries + 1):
        excl_parts = used_songs + blocked + [
            f"{c['song_title']} by {c['artist']}" for c in clean
        ]
        exclusion = ""
        if excl_parts:
            exclusion = (
                f"\n\nDo NOT suggest any of these songs — they have already been used "
                f"or are disqualified: {', '.join(excl_parts)}. Pick something different."
            )

        if attempt == 0:
            print("\nFinding candidates...")
        else:
            print(f"  retrying — {len(blocked)} disqualified so far")

        candidates = get_candidates(user_prompt, exclusion, category=category)

        for c in candidates:
            if is_disqualified(c["song_title"], c["artist"]):
                key = f"{c['song_title']} by {c['artist']}"
                blocked.append(key)
                print(f"    × filtered (disqualified): {c['song_title']} by {c['artist']}")
            else:
                clean.append(c)

        if len(clean) >= 3:
            break

    return clean[:3]


def fetch_lyrics(song_title: str, artist: str) -> str | None:
    try:
        song = genius.search_song(song_title, artist)
        if song and song.lyrics:
            return song.lyrics[:4000]  # truncate to manage tokens
        return None
    except Exception as e:
        print(f"  Lyrics fetch failed for {song_title}: {e}")
        return None


def select_song(user_prompt: str, candidates_with_lyrics: list[dict]) -> dict:
    lyrics_block = ""
    for i, c in enumerate(candidates_with_lyrics, 1):
        lyrics_block += f"\n\n--- Candidate {i}: {c['song_title']} by {c['artist']} ---\n"
        lyrics_block += c.get("lyrics", "[unavailable]")

    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": SELECTION_SYSTEM},
            {"role": "user", "content": f"User's situation: {user_prompt}\n\nLyrics:{lyrics_block}"}
        ],
        temperature=0.3,
    )
    text = resp.choices[0].message.content.strip()
    text = re.sub(r'^```(?:json)?\n?', '', text)
    text = re.sub(r'\n?```$', '', text)
    return json.loads(text)


def generate_oneliner(user_prompt: str, song: dict) -> str:
    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": ONELINER_SYSTEM},
            {"role": "user", "content": (
                f"User's situation: {user_prompt}\n\n"
                f"Selected song: {song['song_title']} by {song['artist']}\n"
                f"Key lyric: {song['key_lyric']}\n"
                f"Iso reasoning: {song['iso_reasoning']}"
            )}
        ],
        temperature=0.9,
    )
    return resp.choices[0].message.content.strip()


# ── Main ──────────────────────────────────────────────────────────────────────

def recommend(
    user_prompt: str,
    used_songs: list[str] | None = None,
    category: str | None = None,
) -> dict:
    # Safety check — before anything else
    if is_safety_situation(user_prompt):
        return {"safety": True, "message": SAFETY_RESPONSE}

    # Perspective pre-step — clarify ambiguous subject before downstream calls
    print("\nReading the situation...")
    clarified = disambiguate_perspective(user_prompt)
    if clarified != user_prompt:
        print(f"  (reading as: {clarified})")
    user_prompt = clarified

    candidates = get_clean_candidates(user_prompt, used_songs=used_songs, category=category)

    print("Fetching lyrics...")
    candidates_with_lyrics = []
    for c in candidates:
        print(f"  - {c['song_title']} by {c['artist']}")
        lyrics = fetch_lyrics(c["song_title"], c["artist"])
        if lyrics:
            candidates_with_lyrics.append({**c, "lyrics": lyrics})
        else:
            print(f"    (no lyrics found, skipping)")

    if not candidates_with_lyrics:
        raise ValueError("Could not fetch lyrics for any candidates.")

    print("Selecting best match...")
    song = select_song(user_prompt, candidates_with_lyrics)

    print("Writing one-liner...")
    oneliner = generate_oneliner(user_prompt, song)

    return {
        "one_liner": oneliner,
        "song_title": song["song_title"],
        "artist": song["artist"],
        "key_lyric": song["key_lyric"],
        "iso_reasoning": song["iso_reasoning"],
        "safety": False,
    }


if __name__ == "__main__":
    print("=== #Mood engine v1 ===")
    used_songs: list[str] = []

    while True:
        user_prompt = input("\nWhat's going on? (or 'quit')\n> ").strip()
        if user_prompt.lower() in ("quit", "exit", "q"):
            break

        result = recommend(user_prompt, used_songs=used_songs)

        print("\n" + "─" * 50)
        if result.get("safety"):
            print(f"\n{result['message']}")
        else:
            print(f"\n{result['one_liner']}\n")
            print(f"{result['song_title']} — {result['artist']}")
            print(f"\n\"{result['key_lyric']}\"")
            print(f"\n[{result['iso_reasoning']}]")
            used_songs.append(f"{result['song_title']} by {result['artist']}")
        print("\n" + "─" * 50)
