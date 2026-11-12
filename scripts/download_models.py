"""Download every model file the try-on app needs into ./models (Hugging Face cache layout).

Run:  .venv\\Scripts\\python scripts\\download_models.py
Safe to re-run; already-downloaded files are skipped.
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(ROOT / "models" / "hf"))
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")  # Xet stalls/corrupts on slow links; plain HTTP resumes

from huggingface_hub import snapshot_download  # noqa: E402

DOWNLOADS = [
    # CatVTON try-on attention weights + DensePose + SCHP human parsers (auto-masking)
    ("zhengchong/CatVTON", ["DensePose/*", "SCHP/*", "mix-48k-1024/*", "config.json"]),
    # Stable Diffusion 1.5 inpainting base (scheduler, safety checker; UNet comes below as fp16)
    ("booksforcharlie/stable-diffusion-inpainting",
     ["model_index.json", "scheduler/*", "unet/config.json", "safety_checker/*", "feature_extractor/*"]),
]
# (repo, files, local folder) - plain files the engine loads directly from models/<folder>
LOCAL = [
    ("stable-diffusion-v1-5/stable-diffusion-inpainting",
     ["unet/config.json", "unet/diffusion_pytorch_model.fp16.safetensors"], "sd-inpaint-unet"),
    ("stabilityai/sd-vae-ft-mse", ["config.json", "diffusion_pytorch_model.safetensors"], "sd-vae-ft-mse"),
]


def main() -> int:
    from huggingface_hub import hf_hub_download
    import shutil

    for repo, patterns in DOWNLOADS:
        print(f"==> {repo}", flush=True)
        path = snapshot_download(repo_id=repo, allow_patterns=patterns, max_workers=8)
        print(f"    ok: {path}", flush=True)
    for repo, files, folder in LOCAL:
        dst = ROOT / "models" / folder
        dst.mkdir(parents=True, exist_ok=True)
        for f in files:
            target = dst / Path(f).name
            if target.exists():
                continue
            print(f"==> {repo}/{f}", flush=True)
            shutil.copy(hf_hub_download(repo, f), target)
    print("All models downloaded.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
