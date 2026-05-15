"""One-shot hero-image generator for the AutoAD README.

Uses Google Gemini's image-generation API per the gemini-imagegen skill.
Saves to ``assets/hero.png``. Idempotent: overwrites if rerun.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from google import genai
from google.genai import types

CONFIG_PATH = Path.home() / ".gemini-imagegen.json"
OUTPUT_PATH = Path("E:/Projects/AutoAD/assets/hero.png")

PROMPT = """Clean modern academic teaser figure for a machine-learning research paper,
in the style of a NeurIPS or ICLR Figure 1. Wide banner composition (21:9). Soft off-white
background with a faint, subtle gradient.

LEFT 60 PERCENT OF THE CANVAS - the "normal data manifold":
A gently curved 2D surface, very pale blue-to-white gradient. Sitting on this surface are
six tight elliptical clusters of small filled dots, each cluster a different muted color:
teal, slate blue, sage green, soft purple, amber, and coral. Each cluster has roughly 30
to 50 dots, packed densely with a clear boundary. The clusters are arranged across the
surface so that they look like distinct modes of a distribution.

ONE specific cluster (the coral / warm red-orange one) is the HELD-OUT cluster. It is
surrounded by a dashed dark-red boundary, displaced slightly away from where it would
naturally sit, with a small curved arrow showing the displacement. Near it, a small
monospace tag in dark grey reads "pseudo-anomaly".

RIGHT 40 PERCENT OF THE CANVAS - the ranked detector list:
A vertical stack of five or six rounded-rectangle cards on a slightly raised plane.
Each card has a very subtle drop shadow, a small abstract icon on the left (a tiny
schematic isolation tree, a tiny cluster of three dots with lines for KNN, a tiny
diagonal hyperplane for SVM), an unlabeled name area, and a horizontal score bar in
muted teal. The top card has a small green checkmark in the upper right corner,
indicating it has been SELECTED. The other cards are in descending score order with
shorter bars.

CONNECTING ELEMENT:
Three or four thin, elegant curved lines flow from the held-out coral cluster on the
left toward the cards on the right, with a soft fade-in. They represent the holdout
being scored by each detector. The lines are pale teal, semi-transparent.

LABELS:
- Top label centered, small caps, very light grey: "LEAVE-CLUSTER-OUT VALIDATION"
- Bottom label centered, small monospace, medium grey: "rank detectors without anomaly labels"

STYLE:
- Vector-clean illustration look. NO photorealism, no 3D rendering, no glossy effects.
- Restrained color palette: muted blues and teals as primary, a single coral accent,
  neutral light greys, a soft sage green for the checkmark.
- Subtle gradients and faint shadows for depth, but overall flat and crisp.
- Aesthetic: technical, intelligent, calm, modern. Think Mike Bostock + Edward Tufte
  meets a NeurIPS teaser figure.
- High visual hierarchy: the held-out cluster and the selected top card should be the
  two strongest focal points.
- The whole image should READ as a scientific figure, not a marketing poster."""


def main() -> int:
    if not CONFIG_PATH.exists():
        print(f"ERROR: config not found at {CONFIG_PATH}", file=sys.stderr)
        return 1
    config = json.loads(CONFIG_PATH.read_text())
    client = genai.Client(api_key=config["api_key"])
    # generate_content requires a Gemini *image-generation* model. The
    # config's `default_model` may be an Imagen ID intended for
    # `generate_images`; pick a Gemini one explicitly.
    model = "gemini-3.1-flash-image-preview"
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"Generating hero image with model={model}, aspect=21:9 ...")

    max_retries = 3
    delay = 5
    last_err: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            response = client.models.generate_content(
                model=model,
                contents=PROMPT,
                config=types.GenerateContentConfig(
                    response_modalities=["IMAGE"],
                    image_config=types.ImageConfig(aspect_ratio="21:9", image_size="2K"),
                ),
            )
            for part in response.parts:
                if part.inline_data:
                    img = part.as_image()
                    img.save(OUTPUT_PATH)
                    size_kb = OUTPUT_PATH.stat().st_size / 1024
                    # Reload via PIL to get dimensions for the report.
                    try:
                        from PIL import Image as PILImage
                        with PILImage.open(OUTPUT_PATH) as p:
                            w, h = p.size
                        print(f"Saved: {OUTPUT_PATH} ({w}x{h}, {size_kb:.1f} KB)")
                    except Exception:
                        print(f"Saved: {OUTPUT_PATH} ({size_kb:.1f} KB)")
                    return 0
            print(f"  No image part in response on attempt {attempt}; retrying ...")
        except Exception as e:
            last_err = e
            print(f"  Attempt {attempt} failed: {e}")
            if attempt < max_retries:
                time.sleep(delay)
                delay *= 2

    print(f"ERROR: failed after {max_retries} attempts; last error: {last_err}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
