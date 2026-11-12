"""End-to-end check without the web UI: loads the models and runs one try-on per garment type.

Run:  .venv\\Scripts\\python -m scripts.smoke_test [--steps 30]
Writes results to outputs/_smoke/.
"""
import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from PIL import Image  # noqa: E402

from app.vton_engine import TryOnEngine  # noqa: E402

CASES = [
    ("people/women_049713_0.jpg", "garments/upper/ootd_00126_00.jpg", "upper"),
    ("people/women_049713_0.jpg", "garments/lower/ootd_051946_1.jpg", "lower"),
    ("people/women_049713_0.jpg", "garments/overall/ootd_053786_1.jpg", "overall"),
    ("people/men_model_5.png", "garments/upper/catvton_2.jpg", "upper"),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--only", type=int, default=None, help="run only case N")
    args = ap.parse_args()
    out = ROOT / "outputs" / "_smoke"
    out.mkdir(parents=True, exist_ok=True)
    eng = TryOnEngine()
    t = time.time()
    eng.load()
    print(f"models loaded in {time.time() - t:.1f}s  {eng.info()}", flush=True)
    cases = CASES if args.only is None else [CASES[args.only]]
    for i, (p, g, kind) in enumerate(cases):
        person = eng.prepare_upload(Image.open(ROOT / "gallery" / p))
        garment = Image.open(ROOT / "gallery" / g)
        r = eng.try_on(person, garment, cloth_type=kind, steps=args.steps, seed=42)
        name = f"{i}_{kind}"
        r.image.save(out / f"{name}.jpg", quality=92)
        r.mask.convert("L").save(out / f"{name}_mask.png")
        # side-by-side sheet: person | garment | result
        h = 512
        tiles = [im.convert("RGB") for im in (person, garment, r.image)]
        tiles = [im.resize((int(im.width * h / im.height), h)) for im in tiles]
        sheet = Image.new("RGB", (sum(t.width for t in tiles), h), "white")
        x = 0
        for tile in tiles:
            sheet.paste(tile, (x, 0))
            x += tile.width
        sheet.save(out / f"{name}_sheet.jpg", quality=90)
        print(f"[{name}] {r.seconds:.1f}s  size={r.image.size}", flush=True)
    print("OK ->", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
