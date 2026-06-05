#!/usr/bin/env python3
"""
Generate the 10-cover album art pool. Run once. Outputs to static/album_art/.

Each cover is anchored to the same cat character + watercolor lofi style as the
loading animation, varying only the environmental scene to express a distinct
emotional texture.

Usage: python generate_art_pool.py [--force]
  --force   re-generate covers even if the PNG already exists
"""
import os
import base64
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# Reuse the same style anchor the per-request generator used
from app import CAT_STYLE_ANCHOR  # noqa: E402

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

OUT_DIR = Path(__file__).parent / "static" / "album_art"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Tag → scene prompt. Each is a full anime portrait, maximalist cozy, with the cat's
# expression explicitly directed to match the mood texture.
COVERS: dict[str, str] = {
    "elegiac": (
        "EXPRESSION: the cat's eyes are softly open and downcast (small black "
        "pupils visible, looking gently downward), a quiet small frown, ears "
        "slightly tilted back, a soft melancholy. NOT smiling. "
        "SCENE: late afternoon in a cozy study, soft melancholy. The cat sits "
        "upright at a wooden writing desk looking quietly down at an open photo "
        "album beside a single dried pressed flower. A worn closed locket. A "
        "half-burnt candle with a thin curl of smoke. A leather journal lies "
        "closed. A cooling cup of tea. An amber-shaded brass desk lamp casts warm "
        "light against a tall window beyond, where rain streaks slowly down "
        "soft-blue afternoon glass. A maidenhair fern on the sill. A framed "
        "botanical print on the wall. Warm interior amber against cool grey-blue "
        "rain. Sad but tender, the room holds the weight kindly."
    ),
    "peaceful": (
        "EXPRESSION: the cat's eyes are closed in soft happy arches, a tiny "
        "upturned smile, ears relaxed, completely at ease. "
        "SCENE: early morning, golden sunbeam content. The cat lies sprawled "
        "comfortably on a thick handwoven rug in cream and sage, in front of a "
        "low oak bookshelf packed with soft-spined books in dusty rose, amber, "
        "sage, plum, and cream. A small wooden tray with a ceramic teapot and "
        "two small cups. A flowering monstera in a terra-cotta pot. A "
        "half-eaten butter croissant on a small plate. A sketchbook open with a "
        "pencil sketch of a plant. A small ceramic cat figurine on a shelf. A "
        "vintage radio. Tall windows with sheer linen curtains, fat sunbeams "
        "streaming across the floor in golden butter light. Cream and butter "
        "yellow palette, complete calm."
    ),
    "numb": (
        "EXPRESSION: the cat's eyes are wide open but blank and unfocused, "
        "looking straight ahead through everything, mouth a small neutral line, "
        "a distant expression. Slightly small in posture. "
        "SCENE: pale foggy morning, quiet kitchen. The cat sits very still at a "
        "small wooden kitchen table, beside an untouched cup of cold coffee with "
        "the thinnest curl of cold steam, an untouched slice of toast on a "
        "plate, a forgotten teaspoon. A pale grey-blue window shows soft mist "
        "outside, vague silhouettes of distant trees. A single warm amber "
        "pendant light has just turned on overhead, beginning to cast warmth. A "
        "jacket left on the back of a chair. A wilting fern in the corner. The "
        "smallest sliver of golden light entering at the edge of the frame "
        "suggests the world is still there, just barely beginning to come back."
    ),
    "reflective": (
        "EXPRESSION: the cat's eyes are half-lidded and thoughtful, looking off "
        "into the middle distance, a small calm closed mouth, a paw resting "
        "gently on an open page. Contemplative, present. "
        "SCENE: deep dusk reading nook, indigo and amber. The cat sits upright "
        "at a worn wooden writing desk, paw resting on an open leather journal. "
        "A brass desk lamp with a green glass shade pools warm amber light. A "
        "fountain pen and a glass ink bottle. A tall packed bookshelf behind. A "
        "trailing philodendron in a hanging ceramic pot. A small framed "
        "botanical illustration on the wall. A vintage record player turning. "
        "Beyond the window: deep indigo dusk sky with the first single bright "
        "star, distant windows of other homes glowing warm. Cinematic indigo "
        "and amber contrast, quiet inner question."
    ),
    "stuck": (
        "EXPRESSION: the cat's eyes are softly open, looking up and to the side "
        "with a small puzzled or slightly impatient frown, one paw resting "
        "against the cheek in a thinking pose. Mildly frustrated patience. "
        "SCENE: late golden afternoon, time-themed pause. The cat sits upright "
        "on a worn wooden desk beside a vintage round wall clock with brass "
        "hands. Long late-day sunlight slants through window blinds in stripes "
        "of warm amber across the desk. A stack of closed hardback books, a "
        "half-finished wooden puzzle scattered, an unopened envelope, a teacup "
        "with the bag still steeping, a small notebook open to a blank page "
        "with a pen across it. A small potted cactus and a succulent. A potted "
        "fern by the window. Warm sepia and amber tones. The moment is held."
    ),
    "belonging": (
        "EXPRESSION: the cat's eyes are softly open looking out a window with a "
        "small wistful expression, ears slightly back, a tender yearning. "
        "Looking out, not at the viewer. "
        "SCENE: evening from a cozy interior, longing to belong. The cat sits "
        "on a wide cushioned windowsill seat draped in a knit throw. Inside: a "
        "warm amber lamp, fairy lights strung along the window frame, a "
        "half-read paperback face-down, a steaming teacup, a small potted basil "
        "in a ceramic pot. A potted fern. Outside: a quiet evening street with "
        "warm cafe and apartment windows glowing across the way, distant tiny "
        "silhouettes of people gathered around tables, twilight sky in dusty "
        "rose, lavender, and deep blue. Yearning, warm, full of tender ache."
    ),
    "anxious": (
        "EXPRESSION: the cat's eyes are wide open and alert with small black "
        "pupils, ears slightly forward and attentive, mouth a small uncertain "
        "line, one paw on a railing. A restful but watchful tension. "
        "SCENE: twilight balcony, beautiful restless evening. The cat sits on a "
        "small ironwork balcony at dusk. A small bistro table behind with a "
        "cooling teacup, a hardback book left open face-down, a knit shawl half "
        "draped across the chair. A small terra-cotta pot of basil. Below: a "
        "softly glowing city of thousands of warm windows, distant traffic "
        "lights, ribbons of street lamps. Sky in deep dusty rose fading into "
        "indigo with the first sharp stars. Warm amber light spilling out from "
        "the open door behind, contrasting the cool twilight. Beautiful "
        "nervous beauty."
    ),
    "acute": (
        "EXPRESSION: the cat is curled small, eyes gently closed but with a "
        "single tiny silver tear sparkle on one cheek, mouth a small soft line, "
        "ears slightly back. Vulnerable and held. "
        "SCENE: deep night, tender protection. The cat is curled tightly under "
        "a thick patchwork quilt of cream, sage, dusty rose, and amber squares "
        "in a softly lit reading nook. An old plush bunny is tucked under the "
        "cat's chin. A small amber-shaded lamp on a side table glows warmly. A "
        "worn paperback has fallen open beside. A cup of tea gone cold. A small "
        "ceramic vessel of dried lavender. Beyond a round window: deep "
        "night-indigo sky scattered with stars and a thin silver crescent moon. "
        "Warm tender lamplight against soft starry night. Held safely through "
        "a hard moment."
    ),
    "liminal": (
        "EXPRESSION: the cat's eyes are softly open looking outward toward the "
        "horizon with a quiet thoughtful expression, neither sad nor smiling, a "
        "paw resting beside an open notebook. Calm in-between. "
        "SCENE: golden sunset by a window, between worlds. The cat sits on a "
        "wooden writing desk beside a wide-open window with the sheerest "
        "curtain catching breeze. A small open notebook with a fountain pen, a "
        "half-packed wooden box of letters tied with twine, a fading mug of "
        "tea, a brass lamp just turning on inside casting first amber light. An "
        "open journal with pressed flowers beneath glass. Outside: rolling "
        "hills fading into a golden sunset of amber, dusty rose, and indigo, a "
        "winding country path leading off into the distance, a single tree "
        "silhouetted. A small terra-cotta pot of lavender on the sill. "
        "Cinematic transition palette."
    ),
    "hopeful": (
        "EXPRESSION: the cat's eyes are softly open with a small upturned "
        "smile, ears slightly forward in gentle attention, looking out toward "
        "the morning light, paws together neatly. Gently optimistic. "
        "SCENE: early morning, new chapter. The cat sits at a sunny wooden desk "
        "by a wide-open window. Fresh-cut wildflowers — daisies, lavender, "
        "small yellow blooms — in a small glass vase. An open notebook with a "
        "single fresh blank page. A steaming cup of coffee with curling steam. "
        "A potted fern catching golden light. A small framed pressed-flower art "
        "on the wall. A hanging plant in front of the window. A small stack of "
        "soft-spined books in cream and amber. Sheer linen curtains catching a "
        "morning breeze. Butter-yellow morning light pouring in across "
        "everything. Outside: a sunrise sky in butter yellow, soft cream, and "
        "first pale blue. Full of gentle promise and quiet joy."
    ),
}


def generate_cover(tag: str, scene: str) -> bytes:
    prompt = f"{CAT_STYLE_ANCHOR} {scene}"
    resp = client.images.generate(
        model="gpt-image-1",
        prompt=prompt,
        size="1024x1024",
        quality="high",
        output_format="png",
        n=1,
    )
    return base64.b64decode(resp.data[0].b64_json)


def main():
    force = "--force" in sys.argv
    skipped, generated, failed = [], [], []

    for tag, scene in COVERS.items():
        path = OUT_DIR / f"{tag}.png"
        if path.exists() and not force:
            skipped.append(tag)
            print(f"  ✓ skip  {tag} (exists)")
            continue

        print(f"  · gen   {tag} …", end=" ", flush=True)
        t0 = time.time()
        try:
            png = generate_cover(tag, scene)
            path.write_bytes(png)
            elapsed = time.time() - t0
            generated.append(tag)
            print(f"done ({elapsed:.1f}s, {len(png):,}B)")
        except Exception as e:
            failed.append((tag, str(e)))
            print(f"FAILED: {type(e).__name__}: {e}")

    print(f"\nDone. generated={len(generated)} skipped={len(skipped)} failed={len(failed)}")
    if failed:
        for tag, err in failed:
            print(f"  · {tag}: {err}")
        sys.exit(1)


if __name__ == "__main__":
    main()
