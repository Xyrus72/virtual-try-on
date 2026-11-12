"""Populate ./gallery with sample people and garments.

Sources (research / non-commercial use):
  * CatVTON demo images (bundled in third_party/CatVTON/resource/demo/example)
  * OOTDiffusion example garments from VITON-HD / DressCode (downloaded from GitHub)
"""
import shutil
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "third_party" / "CatVTON" / "resource" / "demo" / "example"
G = ROOT / "gallery"
OOTD = "https://raw.githubusercontent.com/levihsu/OOTDiffusion/main/run/examples/garment/"

# Categories verified by eye (VITON-HD *_00 and DressCode 048-050xxx are tops).
OOTD_GARMENTS = {
    "upper": ["00055_00", "00126_00", "00151_00", "00470_00", "02015_00", "02305_00", "03032_00",
              "03244_00", "04825_00", "06123_00", "07382_00", "07764_00", "10297_00", "12562_00",
              "048554_1", "048769_1", "049805_1", "049920_1", "049940_1", "049947_1", "049949_1",
              "049965_1", "050002_1", "050105_1", "050181_1", "050191_1"],
    "lower": ["051412_1", "051473_1", "051515_1", "051517_1", "051827_1", "051946_1", "051988_1",
              "051991_1"],
    "overall": ["051998_1", "052234_1", "053290_1", "053319_1", "053742_1", "053744_1", "053786_1",
                "053790_1"],
}


def copy_demo() -> None:
    (G / "people").mkdir(parents=True, exist_ok=True)
    for sub in ("women", "men"):
        for f in sorted((DEMO / "person" / sub).glob("*")):
            if f.suffix.lower() in {".jpg", ".png"}:
                shutil.copy(f, G / "people" / f"{sub}_{f.name}")
    for t in ("upper", "overall"):
        (G / "garments" / t).mkdir(parents=True, exist_ok=True)
        for i, f in enumerate(sorted((DEMO / "condition" / t).glob("*.jpg"))):
            shutil.copy(f, G / "garments" / t / f"catvton_{i + 1}.jpg")


def fetch_ootd() -> None:
    for t, ids in OOTD_GARMENTS.items():
        d = G / "garments" / t
        d.mkdir(parents=True, exist_ok=True)
        for gid in ids:
            dst = d / f"ootd_{gid}.jpg"
            if dst.exists():
                continue
            try:
                urllib.request.urlretrieve(OOTD + gid + ".jpg", dst)
                print("  +", dst.relative_to(ROOT))
            except Exception as e:  # noqa: BLE001
                print("  ! skip", gid, e)


if __name__ == "__main__":
    copy_demo()
    fetch_ootd()
    print("Gallery ready:", G)
    sys.exit(0)
