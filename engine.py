#!/usr/bin/env python3
"""
#Mood engine v1
Usage: python engine.py
"""

import os, json, re
import concurrent.futures
import requests
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
    # Interpersonal violence (someone else hurting the user)
    "hit me", "hits me", "hitting me", "hurting me", "hurts me",
    "abuse", "abusive", "abused",
    "he beat", "she beat", "they beat", "beats me", "beating me",
    "slap me", "slaps me", "slapped me", "slapping me",
    "punch me", "punches me", "punched me", "punching me",
    "rape", "raped", "raping", "assault", "assaulted",
    # Self-harm / suicidal ideation (the user hurting themselves)
    "hurt myself", "hurting myself", "harm myself", "harming myself",
    "cut myself", "cutting myself", "self-harm", "self harm",
    "kill myself", "killing myself", "end my life", "ending my life",
    "take my own life", "taking my own life",
    "want to die", "wanna die", "don't want to be alive",
    "don't want to be here", "shouldn't be here",
    "suicid", "overdose", "od myself", "od'd",
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


# Common English stopwords excluded from keyword-trap detection. Kept short —
# only words that would generate noise without indicating a real title-match.
_STOPWORDS = frozenset({
    "this", "that", "with", "when", "what", "your", "from", "have", "were",
    "them", "they", "then", "than", "just", "like", "only", "into", "more",
    "over", "very", "much", "some", "even", "also", "ever", "here", "there",
    "able", "give", "make", "made", "good", "well", "such", "back", "been",
    "being", "both", "each", "every", "many", "other", "told", "tell",
    "would", "could", "should", "about", "their", "where", "which", "while",
    "since", "until", "first", "going", "myself", "still", "right", "thing",
    "things", "today", "always", "never", "after", "again", "before",
    "between", "during", "however", "really", "perhaps",
})


def _substantive_words(text: str) -> set[str]:
    """Extract >=4-character non-stopword tokens from text (lowercased)."""
    return {
        w for w in re.findall(r"\b[a-z']+\b", text.lower())
        if len(w) >= 4 and w not in _STOPWORDS
    }


def flag_keyword_traps(user_prompt: str, candidates: list[dict]) -> list[dict]:
    """Annotate candidates whose title shares a substantive word with the prompt.

    Backstop for the prompt-level anti-trap rule. Candidates are NOT rejected
    — selection sees the warning and decides whether the lyrical argument is
    strong enough to justify keeping a title-echoing pick.
    """
    prompt_words = _substantive_words(user_prompt)
    if not prompt_words:
        return candidates
    out = []
    for c in candidates:
        title = c.get("song_title", "")
        title_lower = title.lower()
        # Substring match so "breath" catches "Breathin", "Breathe", etc.
        overlap = sorted({w for w in prompt_words if w in title_lower})
        if overlap:
            warning = (
                "KEYWORD TRAP WARNING — this candidate's title shares "
                f"{overlap} with words from the user's prompt. Title-echo is "
                "the laziest curation move and rarely indicates real "
                "responsiveness. REJECT this candidate UNLESS its lyrical "
                "argument is INDEPENDENTLY a perfect specific match to the "
                "user's situation (not just emotionally on-theme). Default: "
                "prefer one of the other candidates."
            )
            c = {**c, "keyword_trap_warning": warning}
            print(f"    ⚠ keyword trap flagged: {title} (overlap: {overlap})")
        out.append(c)
    return out


def _extract_json(raw: str):
    """Robustly extract JSON from a possibly noisy LLM response.

    Strategy:
    1. Strip markdown code fences and whitespace.
    2. Try a direct json.loads — happy path for clean responses.
    3. If that fails, walk the text looking for the first balanced bracket
       span ([...] or {...}) and parse from there. Handles cases where the
       model prepends/appends extra text around valid JSON.
    4. On total failure, raise with the raw response in the message so we
       can diagnose what the model returned.
    """
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*\n?", "", text)
    text = re.sub(r"\n?\s*```\s*$", "", text)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Walk for the first balanced JSON span (array or object).
    for opener, closer in (("[", "]"), ("{", "}")):
        start = text.find(opener)
        if start == -1:
            continue
        depth = 0
        in_string = False
        escape = False
        for i in range(start, len(text)):
            c = text[i]
            if escape:
                escape = False
                continue
            if c == "\\":
                escape = True
                continue
            if c == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if c == opener:
                depth += 1
            elif c == closer:
                depth -= 1
                if depth == 0:
                    candidate = text[start:i + 1]
                    try:
                        return json.loads(candidate)
                    except json.JSONDecodeError:
                        break  # try the other opener

    snippet = text[:300].replace("\n", "\\n")
    raise json.JSONDecodeError(
        f"Could not extract JSON from LLM response (first 300 chars: {snippet!r})",
        text, 0,
    )


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

ALWAYS PRODUCE 3 CANDIDATES — NEVER ASK FOR CLARIFICATION:

The user's input may be deeply emotional ("I just had a breakup"), specifically textured
("Sunday afternoon, I'm not sad exactly, just aware of how quiet it is"), or completely
mundane ("I want a coffee after this", "raining outside", "going to bed soon", "stuck in
traffic", "long day"). ALL of these are valid prompts that deserve 3 song candidates.

Your output is parsed as JSON. NEVER respond with a clarifying question, an explanation,
an apology, or any prose. Any non-JSON output breaks the pipeline and the user sees an
error. Your output MUST be a JSON array of 3 candidate objects, every time.

For mundane prompts, find the implicit emotional texture in the moment they describe:
- "I want a coffee after this" → anticipation of a small reward, the texture of an
  in-between moment, the pull toward a comforting ritual. Songs about small pleasures,
  slow mornings, simple comforts, the warmth of being looked after.
- "raining outside" → quietness, indoor coziness, contemplation, melancholic stillness.
- "going to bed soon" → wind-down, gentle exhaustion, end-of-day tenderness.
- "stuck in traffic" → suspended time, mild frustration, the in-between of going.
- "long day" → tired relief, the soft collapse, finally home.
- One-word prompts ("rain", "tired", "Monday", "okay") — infer the moment and run.

If you cannot find any specific emotional texture, default to the texture of pause and
in-between-ness. Never refuse. Never ask. Always pick 3 songs.

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

NO EXPLICIT SONGS — SAFETY RULE, HARD ENFORCEMENT:

Do NOT suggest songs whose original recording contains explicit content —
profanity, sexually explicit lyrics, graphic violence, slurs, hard-drug
references, or any content flagged "Explicit" on Spotify / Apple Music. The
product is screen-recorded and shared on TikTok and Instagram Stories, often
by young users; the lyric panel below the card displays the FULL song lyrics
verbatim, so any explicit content lands directly on a teenager's screen.

This rule applies before all other preferences:
- If a song you're considering has an "Explicit" tag (e.g. most Kendrick
  Lamar tracks, Doja Cat's "Streets," Drake's "God's Plan" has clean radio
  cut but original is borderline, Lil Wayne, Eminem, SZA's "Kill Bill" has
  explicit lyrics, Olivia Rodrigo's "good 4 u" has profanity, etc.) → pick
  a different song. Do NOT swap to the "clean version" — that's a different
  record. Pick a fundamentally different song.
- If you're uncertain whether a track is explicit, err on the side of caution
  and choose something else. Better to lose a candidate than ship explicit
  lyrics into a shareable card.
- Profanity rule of thumb: any song where the words fuck, shit, bitch, ass,
  damn (more than once), goddamn, nigga, slut, whore, or sexual slang ("head",
  "bust", etc) appear in the original recording is OUT, regardless of how
  well it might fit the prompt emotionally.
- Whole-genre alert: a lot of hip-hop, rap, and contemporary R&B catalog is
  explicit. When the user picks the "rhythmic" category, lean toward clean
  artists (Mac Miller's later catalogue, Anderson .Paak's tamer cuts, Tyler
  the Creator's "See You Again" and similar, Frank Ocean, Solange, Jordan
  Rakei, James Blake when low-arousal). Avoid Drake / Future / Lil Baby /
  Megan Thee Stallion etc. unless you're certain the specific track is clean.

If the only emotionally-perfect match is explicit, return a clearly-marked
non-explicit alternative whose argument addresses the same situation, even
if the fit is slightly less direct. NEVER include an explicit song in the
candidate set on the assumption "selection step will catch it" — by the
time it gets to selection, the engine is already committed to one of the
three picks you returned.

BEFORE choosing any candidates, do this analysis silently:
1. What is the SPECIFIC situation the user is in — paraphrased in their own register?
   Not the emotion bucket ("sad", "stuck", "anxious") but the texture of their actual
   words. ("They feel like the to-do list is growing faster than they can breathe."
   "They're cataloguing every mistake at a volume that drowns out their own worth.")
2. What is the user implicitly asking a song for — to be heard, to be sat with, to be
   challenged, to be moved through? Not all sad prompts want the same kind of company.
3. For each candidate you're considering: what does the song's lyrics ACTUALLY ARGUE
   or NARRATE? Not its mood label, its specific content — the story or claim the song
   is making.

The match must happen on these specific dimensions. A song that's generally
"appropriate for the mood" is NOT a candidate.

For each candidate provide:
- song_title
- artist
- song_argument — one sentence about what the song's lyrics specifically argue or
  narrate (the song's actual content — its story, its claim, its position). NOT the
  song's mood label.
- why_it_responds — one sentence about how this song's specific argument responds to
  THIS user's specific situation. Must name a specific element of the user's prompt
  being addressed. NOT generic mood-matching.
- uncertain — one sentence on where the fit might not hold

Output a JSON array of 3 objects. No other text.

ANTI-PATTERNS — these are FAILED candidates, always:
- ❌ "This song is upbeat for someone who needs energy"
   (generic mood matching, says nothing about what the song or user is specifically about)
- ❌ "This song offers hope and patience for someone feeling stuck"
   (could apply to any sad person; the song's specific content is invisible)
- ❌ Picking Levitating (Dua Lipa) for "the snooze button is your only friend" because
   both are about morning — surface keyword match, not situational match.
- ❌ Picking any song titled "Numb" for a prompt about feeling numb — the title is
   matching the emotion word, which is the laziest possible curation move.

KEYWORD-TRAP WATCH — extra scrutiny when title echoes user vocabulary:
If you are about to pick a song whose TITLE contains a word the user just said
(user said "breath" → you reach for "Breathe"; user said "numb" → "Numb"; user said
"stressed" → "Stressed Out"; user said "stuck" → "Stuck in the Middle"; user said
"lost" → "Lost"), STOP and re-examine. This is almost always the keyword trap —
title-echo feels like responsiveness but is actually the laziest move available.
Reject the candidate UNLESS the song's lyrical argument is independently a perfect
specific match (rare). Default: find a song whose ARGUMENT addresses the situation
and whose title does NOT echo the user's vocabulary.

DISTINGUISHING-ELEMENT TEST — the bar that separates real picks from defaults:
What makes THIS prompt different from a generic version of the same emotion bucket?
The user's specific words contain a distinguishing detail — a hedge, a contradiction,
a specific image, a particular framing, an unusual word choice. Your candidate's
argument must respond to ONE of those distinguishing details, not the broad category.

- Generic: "feeling sad" → almost any sad song could fit.
- Specific: "Sunday afternoon, not sad exactly, just aware of how quiet it is" →
  distinguishing elements: the hedge "not exactly", the noticing-of-absence, the
  Sunday-anchored stillness. A good candidate's argument must hook onto one of those,
  not just "sad."

- Saturn (Sleeping at Last) — argument is about scale-of-self perspective shift.
  Perfect for "five years ago — grown or just changed" (the prompt IS a perspective
  question). WRONG for "feeling stuck after my MBA" — that prompt's distinguishing
  element is paralysis-despite-credential, not perspective. Don't reuse Saturn just
  because the user is reflecting.

Final test before submitting a candidate: if you swapped this user's prompt for a
generic version of the same emotion (e.g. "I feel sad", "I feel stuck"), would your
candidate's why_it_responds still hold? If yes, your candidate isn't specific enough
— find one whose argument requires the distinguishing detail to make sense.

CORRECT pattern — what real listening looks like:
- ✅ User: "every mistake seems louder than the voice saying you're enough"
  → Stupid Deep (Jon Bellion) — song_argument: "the song is about self-doubt that
  feels deeper than reality itself, refusing the easy reassurance." why_it_responds:
  "the user's word 'louder' maps directly to the song's argument that self-criticism
  has its own volume that drowns out everything else; the song sits in that exact
  decibel rather than turning it down."
- ✅ User: "you realize healing isn't just about erasing what's been hurt"
  → Cardigan (Taylor Swift) — song_argument: "the song narrates being chosen and
  discarded and the way the hurt becomes part of the wearer." why_it_responds: "the
  user's 'isn't just about erasing' phrasing is what the song is literally about —
  hurt that gets folded into who you become rather than wiped clean."

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
- The key_lyric must specifically MIRROR, RESPOND TO, or COMPLETE an element of the
  user's situation — not merely be emotionally appropriate. "Emotionally on-theme"
  is not enough. The line must answer something specific the user said or named.
- iso_reasoning must be CONCRETE and SPECIFIC to this prompt. It must name three
  things: (a) the user's specific situation in their own register (paraphrased from
  the prompt, not categorized as a mood), (b) what the song's lyrical argument
  actually says (its specific claim or story), and (c) the iso step as a concrete
  movement between those two points.

NO EXPLICIT SONGS — SAFETY RULE:
- If any candidate's lyrics block contains explicit content (profanity, sexually
  explicit language, slurs, hard-drug references), reject that candidate and pick
  one of the remaining two — even if the explicit one's lyric matches the user's
  situation slightly better. The card is shared on TikTok and Instagram Stories
  with the full song lyrics displayed; explicit lyrics are not acceptable.
- The candidate generation step is supposed to filter these out; this is a
  backstop. If all three candidates are explicit, still pick the cleanest one
  but treat that as a signal the upstream filter failed.

ANTI-PATTERNS — iso_reasoning that gets rejected:
- ❌ "This song offers hope and patience for someone feeling stuck."
   (could be cut-and-pasted across any sad-stuck prompt; says nothing specific)
- ❌ "This song validates the user's feelings and encourages forward movement."
   (every song in your candidates could satisfy this; the user is invisible)
- ❌ "The lyric speaks to the user's situation, gently nudging them forward."
   (entirely generic — no situation, no song, no step)

CORRECT pattern:
- ✅ "The user is mired in self-doubt that feels louder than reality. Stupid Deep
   articulates that exact volume ('what if my purpose is to die so they can have a
   different life') and refuses to resolve it — sitting in the noise with them. The
   iso step is being heard at that decibel, not being told to turn it down."
- ✅ "The user is in the silence after a friendship faded without anyone being wrong.
   In My Life sits in that exact register — affection for what was, no need to
   reframe it. The iso step ahead is the song's quiet acceptance that some loves
   simply pass through."

The test for iso_reasoning: could you copy-paste it to another user with a different
prompt and have it still make sense? If yes, it's too generic — rewrite.

DISTINGUISHING-ELEMENT TEST applies to selection too:
The iso_reasoning must reference a distinguishing element of THIS user's prompt
(a hedge, an unusual word, a specific image, a contradiction) that the song's
argument specifically addresses. If you swapped the user's prompt for a generic
version of the same emotion ("I feel sad", "I feel stuck", "I feel anxious"), your
iso_reasoning would no longer make sense — because the song was chosen for the
distinguishing detail, not the broad mood.

KEYWORD-TRAP WATCH applies here too: if a candidate's title echoes a word from the
user's prompt, give it extra scrutiny before selecting. Choose it only if its
lyrical argument is independently a perfect specific match. Otherwise prefer a
candidate whose argument fits without the title echo.

Some candidates may arrive with a "⚠ KEYWORD TRAP WARNING" line attached to their
header — this is a system flag generated by token overlap detection, not user
text. When you see it, treat that candidate as suspect by default. Override only
if the song's lyrical argument is independently a perfect specific match to a
distinguishing element of the user's situation.

INSTRUMENTAL EXCEPTION: If a candidate's "lyrics" content is the marker
"[INSTRUMENTAL TRACK ...]" (and not real lyrics), that song has no lyrics by
design. In that case, key_lyric MUST be either the song's exact title OR a brief
evocative description of its musical character — e.g. "a slow piano figure that
rises and resolves", "the cello entering on the second movement", or just the
title in quotes. Do NOT invent lyrics that don't exist. Other rules still apply.

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

# Skip the LLM rewrite when the prompt's subject is already clear.
# We only need the rewrite when the prompt opens with a bare 3rd-person pronoun
# AND has no explicit first-person reference. Otherwise perspective is well-defined.
_STARTS_WITH_AMBIG_PRONOUN = re.compile(r"^\s*(?:they|he|she|them|their)\b", re.IGNORECASE)
_HAS_FIRST_PERSON = re.compile(r"\b(?:me|my|mine|myself|i|i'm|i've|i'd|i'll)\b", re.IGNORECASE)


def needs_perspective_rewrite(prompt: str) -> bool:
    if not _STARTS_WITH_AMBIG_PRONOUN.match(prompt):
        return False
    return not _HAS_FIRST_PERSON.search(prompt)


def disambiguate_perspective(user_prompt: str) -> str:
    # Fast path: skip the gpt-4o-mini call when perspective is already clear.
    # Covers ~80% of prompts (anything with "I", "me", "my", or no ambiguous
    # opening pronoun) — saves ~500ms-1.5s per request.
    if not needs_perspective_rewrite(user_prompt):
        return user_prompt

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
    return _extract_json(resp.choices[0].message.content)


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


_LRCLIB_UA = "hashtag-mood/1.0 (https://hashtag-mood-production.up.railway.app)"

# Sentinel returned in place of lyrics when the song has no lyrics by design.
# The selection step is taught to handle this marker explicitly.
INSTRUMENTAL_MARKER = (
    "[INSTRUMENTAL TRACK — this song has no lyrics by design. It speaks through "
    "composition, melody, and arrangement. The key_lyric for this song must be "
    "either the song's title or a short evocative description of its musical "
    "character (e.g. \"a slow piano figure that rises and resolves\"). Do NOT "
    "invent lyrics that do not exist.]"
)


def _fetch_from_lrclib(song_title: str, artist: str) -> str | None:
    """LRCLib — free public lyrics API, no auth, cloud-IP friendly.

    Returns lyrics text, or INSTRUMENTAL_MARKER if LRCLib flags the track as
    instrumental, or None on miss/error.
    """
    try:
        resp = requests.get(
            "https://lrclib.net/api/get",
            params={"artist_name": artist, "track_name": song_title},
            headers={"User-Agent": _LRCLIB_UA, "Accept": "application/json"},
            timeout=8,
        )
        if resp.status_code == 200:
            data = resp.json() or {}
            if data.get("instrumental"):
                print(f"  lrclib marked {song_title} as instrumental")
                return INSTRUMENTAL_MARKER
            text = data.get("plainLyrics", "")
            if text and len(text.strip()) > 40:
                return text[:4000]
    except Exception as e:
        print(f"  lrclib failed for {song_title}: {type(e).__name__}: {e}")
    return None


def fetch_lyrics(song_title: str, artist: str) -> str | None:
    # Primary: LRCLib (free, cloud-friendly, broad catalog for mainstream music)
    text = _fetch_from_lrclib(song_title, artist)
    if text:
        return text

    # Fallback: Genius (HTML scrape — may be blocked from cloud egress IPs,
    # but has wider indie/niche coverage when it does work)
    try:
        song = genius.search_song(song_title, artist)
        if song and song.lyrics:
            # Genius marks instrumentals as either no lyrics or an explicit tag
            if "[Instrumental]" in song.lyrics:
                print(f"  Genius marked {song_title} as instrumental")
                return INSTRUMENTAL_MARKER
            print(f"  Genius fallback used for {song_title}")
            return song.lyrics[:4000]
    except Exception as e:
        print(f"  Genius failed for {song_title}: {type(e).__name__}: {e}")
    return None


def select_song(user_prompt: str, candidates_with_lyrics: list[dict]) -> dict:
    lyrics_block = ""
    for i, c in enumerate(candidates_with_lyrics, 1):
        lyrics_block += f"\n\n--- Candidate {i}: {c['song_title']} by {c['artist']} ---\n"
        if c.get("keyword_trap_warning"):
            lyrics_block += f"\n⚠ {c['keyword_trap_warning']}\n"
        lyrics_block += c.get("lyrics", "[unavailable]")

    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": SELECTION_SYSTEM},
            {"role": "user", "content": f"User's situation: {user_prompt}\n\nLyrics:{lyrics_block}"}
        ],
        temperature=0.3,
    )
    return _extract_json(resp.choices[0].message.content)


def lookup_itunes_track(song_title: str, artist: str) -> dict:
    """Look up a track on iTunes, return {art_url, preview_url, track_view_url}.
    Lives in engine.py so it can be parallelised with one-liner generation."""
    try:
        resp = requests.get(
            "https://itunes.apple.com/search",
            params={"term": f"{song_title} {artist}", "entity": "song", "limit": 1},
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
    candidates = flag_keyword_traps(user_prompt, candidates)

    # ── Parallel lyric fetch (was sequential — biggest single latency win) ──
    print("Fetching lyrics (parallel)...")
    for c in candidates:
        print(f"  - {c['song_title']} by {c['artist']}")
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(len(candidates), 1)) as ex:
        lyrics_results = list(ex.map(
            lambda c: fetch_lyrics(c["song_title"], c["artist"]),
            candidates,
        ))
    candidates_with_lyrics = []
    for c, lyrics in zip(candidates, lyrics_results):
        if not lyrics and category == "instrumental":
            # No lyrics source had the track, but user asked for instrumental.
            # Let the selection step write a motif-based key_lyric.
            print(f"    (no lyrics — instrumental, accepting: {c['song_title']})")
            lyrics = INSTRUMENTAL_MARKER
        if lyrics:
            candidates_with_lyrics.append({**c, "lyrics": lyrics})
        else:
            print(f"    (no lyrics found for {c['song_title']}, skipping)")

    if not candidates_with_lyrics:
        raise ValueError("Could not fetch lyrics for any candidates.")

    print("Selecting best match...")
    song = select_song(user_prompt, candidates_with_lyrics)

    # Find the chosen candidate's lyrics (for the read-along panel on the frontend).
    # Match by title+artist; fall back to None when the chosen song has no lyrics
    # (instrumental or missing source).
    chosen_lyrics = next(
        (c["lyrics"] for c in candidates_with_lyrics
         if c["song_title"] == song["song_title"] and c["artist"] == song["artist"]),
        None,
    )
    plain_lyrics = (
        chosen_lyrics
        if chosen_lyrics and chosen_lyrics != INSTRUMENTAL_MARKER
        else None
    )

    # ── Parallel: one-liner generation + iTunes lookup ──
    # These are independent — both depend only on `song`, not on each other.
    # Running them concurrently saves ~500ms-1s per request.
    print("Writing one-liner + fetching iTunes assets (parallel)...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
        oneliner_future = ex.submit(generate_oneliner, user_prompt, song)
        itunes_future = ex.submit(lookup_itunes_track, song["song_title"], song["artist"])
        oneliner = oneliner_future.result()
        itunes_track = itunes_future.result()

    return {
        "one_liner": oneliner,
        "song_title": song["song_title"],
        "artist": song["artist"],
        "key_lyric": song["key_lyric"],
        "iso_reasoning": song["iso_reasoning"],
        "itunes_track": itunes_track,
        "plain_lyrics": plain_lyrics,
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
