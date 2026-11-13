"""Virtual try-on engine built on CatVTON (ICLR 2025, Zheng-Chong et al.).

Pipeline for one request:
  1. Frame the person photo to 3:4 without cutting the person off (crop wide photos, pad tall ones).
  2. Auto-mask the clothing region with DensePose + two SCHP human parsers (or use a user-painted mask).
  3. Run the CatVTON diffusion model at 768x1024.
  4. Paste the generated region back into the ORIGINAL full-resolution photo with a feathered mask,
     so face, hair, hands and background stay pixel-perfect.
"""
from __future__ import annotations

import os
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

ROOT = Path(__file__).resolve().parents[1]
CATVTON_DIR = ROOT / "third_party" / "CatVTON"
DETECTRON2_DIR = ROOT / "third_party" / "detectron2_src"
os.environ.setdefault("HF_HOME", str(ROOT / "models" / "hf"))
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
for p in (str(CATVTON_DIR), str(DETECTRON2_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image, ImageOps  # noqa: E402

# detectron2 is used as pure Python (no compiled ops needed for DensePose R50-FPN inference).
torch.jit.script = lambda f, *a, **k: f  # same patch the official CatVTON app applies

from diffusers.image_processor import VaeImageProcessor  # noqa: E402
from huggingface_hub import snapshot_download  # noqa: E402

from model.cloth_masker import AutoMasker  # noqa: E402  (CatVTON)
from model.pipeline import CatVTONPipeline  # noqa: E402  (CatVTON)

WIDTH, HEIGHT = 768, 1024
CLOTH_TYPES = ("upper", "lower", "overall")


@dataclass
class TryOnResult:
    image: Image.Image          # full-resolution composite
    raw: Image.Image            # raw 768x1024 model output
    mask: Image.Image           # mask used, at original resolution
    seed: int
    seconds: float
    warning: Optional[str] = None


def _offline_snapshot(repo_id: str, **kw) -> str:
    """Use the local cache; only hit the network if a file is missing."""
    try:
        return snapshot_download(repo_id=repo_id, local_files_only=True, **kw)
    except Exception:
        return snapshot_download(repo_id=repo_id, **kw)


class Framing:
    """Maps an arbitrary photo onto the model's 3:4 canvas and back."""

    def __init__(self, image: Image.Image):
        self.orig = image
        w, h = image.size
        target = WIDTH / HEIGHT
        if w / h > target:            # too wide -> crop width, keep full height (whole body stays)
            new_w = int(round(h * target))
            x0 = (w - new_w) // 2
            self.box = (x0, 0, x0 + new_w, h)       # region of original used
            self.pad = (0, 0)
            self.canvas_size = (new_w, h)
        else:                         # too tall -> pad width with blurred edge extension
            new_w = int(round(h * target))
            padl = (new_w - w) // 2
            self.box = (0, 0, w, h)
            self.pad = (padl, new_w - w - padl)
            self.canvas_size = (new_w, h)

    def to_model(self, image: Image.Image, is_mask: bool = False) -> Image.Image:
        img = image.crop(self.box)
        if self.pad != (0, 0):
            arr = np.array(img)
            if is_mask:
                arr = cv2.copyMakeBorder(arr, 0, 0, self.pad[0], self.pad[1], cv2.BORDER_CONSTANT, value=0)
            else:
                ext = cv2.copyMakeBorder(arr, 0, 0, self.pad[0], self.pad[1], cv2.BORDER_REPLICATE)
                blurred = cv2.GaussianBlur(ext, (0, 0), sigmaX=25)
                ext[:, : self.pad[0]] = blurred[:, : self.pad[0]]
                if self.pad[1]:
                    ext[:, -self.pad[1]:] = blurred[:, -self.pad[1]:]
                arr = ext
            img = Image.fromarray(arr)
        resample = Image.NEAREST if is_mask else Image.LANCZOS
        return img.resize((WIDTH, HEIGHT), resample)

    def from_model(self, image: Image.Image, is_mask: bool = False) -> Image.Image:
        """Model-space image -> original-resolution image (only the region that maps back)."""
        resample = Image.BILINEAR if is_mask else Image.LANCZOS
        big = image.resize(self.canvas_size, resample)
        if self.pad != (0, 0):
            cw, ch = self.canvas_size
            big = big.crop((self.pad[0], 0, cw - self.pad[1], ch))
        full = Image.new(big.mode, self.orig.size, 0 if is_mask else (0, 0, 0))
        full.paste(big, (self.box[0], self.box[1]))
        return full


class TryOnEngine:
    def __init__(self, device: Optional[str] = None, dtype: str = "auto", safety_check: Optional[bool] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if dtype == "auto":
            if self.device == "cuda":
                self.dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            else:
                self.dtype = torch.float32
        else:
            self.dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[dtype]
        if safety_check is None:
            safety_check = os.environ.get("VTON_SAFETY_CHECK", "1") != "0"
        self.safety_check = safety_check
        self.lock = threading.Lock()
        self.pipeline: Optional[CatVTONPipeline] = None
        self.automasker: Optional[AutoMasker] = None
        self.mask_processor = VaeImageProcessor(
            vae_scale_factor=8, do_normalize=False, do_binarize=True, do_convert_grayscale=True
        )
        self.load_seconds = 0.0
        self.last_warning: Optional[str] = None

    # ------------------------------------------------------------------ loading
    def load(self) -> None:
        if self.pipeline is not None:
            return
        t0 = time.time()
        # DensePose writes temp files relative to CWD; keep them inside the project.
        work = ROOT / "runtime"
        work.mkdir(exist_ok=True)
        os.chdir(work)
        catvton = _offline_snapshot("zhengchong/CatVTON",
                                    allow_patterns=["DensePose/*", "SCHP/*", "mix-48k-1024/*", "config.json"])
        base = _offline_snapshot("booksforcharlie/stable-diffusion-inpainting",
                                 allow_patterns=["model_index.json", "scheduler/*", "unet/config.json",
                                                 "safety_checker/*", "feature_extractor/*"])
        local_vae = ROOT / "models" / "sd-vae-ft-mse"
        if (local_vae / "diffusion_pytorch_model.safetensors").exists():
            vae = str(local_vae)
        else:
            vae = _offline_snapshot("stabilityai/sd-vae-ft-mse",
                                    allow_patterns=["config.json", "diffusion_pytorch_model.safetensors"])
        # CatVTONPipeline hardcodes the VAE repo id; point it at the local snapshot.
        import model.pipeline as mp
        _orig = mp.AutoencoderKL.from_pretrained
        mp.AutoencoderKL.from_pretrained = lambda *_a, **_k: _orig(vae)
        # Prefer the verified local fp16 UNet (same SD-1.5 inpainting weights, half the download).
        local_unet = ROOT / "models" / "sd-inpaint-unet"
        _orig_unet = mp.UNet2DConditionModel.from_pretrained
        if (local_unet / "diffusion_pytorch_model.fp16.safetensors").exists():
            mp.UNet2DConditionModel.from_pretrained = lambda *_a, **_k: _orig_unet(
                str(local_unet), variant="fp16", torch_dtype=self.dtype)
        try:
            self.pipeline = CatVTONPipeline(
                base_ckpt=base,
                attn_ckpt=catvton,
                attn_ckpt_version="mix",
                weight_dtype=self.dtype,
                device=self.device,
                skip_safety_check=not self.safety_check,
                use_tf32=True,
            )
        finally:
            mp.AutoencoderKL.from_pretrained = _orig
            mp.UNet2DConditionModel.from_pretrained = _orig_unet
        self.automasker = AutoMasker(
            densepose_ckpt=os.path.join(catvton, "DensePose"),
            schp_ckpt=os.path.join(catvton, "SCHP"),
            device=self.device,
        )
        self.load_seconds = time.time() - t0

    @property
    def ready(self) -> bool:
        return self.pipeline is not None

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def prepare_upload(image: Image.Image, max_side: int = 2048) -> Image.Image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        if max(image.size) > max_side:
            image.thumbnail((max_side, max_side), Image.LANCZOS)
        return image

    def _auto_mask_model_space(self, person_model: Image.Image, cloth_type: str) -> Image.Image:
        out = self.automasker(person_model, cloth_type)
        dense = np.array(out["densepose"])
        if dense.max() == 0:
            raise ValueError("No person detected in the photo. Use a clear, front-facing photo "
                             "with the body (at least the upper half) visible.")
        self.last_warning = self._coverage_check(dense, cloth_type)
        return out["mask"]

    @staticmethod
    def _coverage_check(dense: np.ndarray, cloth_type: str) -> Optional[str]:
        """Make sure the body part the garment goes on is actually in the photo.

        Percentages are of the 768x1024 frame, from DensePose body-part labels.
        Raises ValueError when the garment has nowhere to go; returns a warning when it will be cut short.
        """
        pct = lambda ids: float(np.isin(dense, ids).mean()) * 100  # noqa: E731
        torso, thighs, shins = pct([1, 2]), pct([7, 8, 9, 10]), pct([11, 12, 13, 14])
        if cloth_type == "upper" and torso < 2:
            raise ValueError("Your upper body isn't clearly visible. Use a photo showing your chest and waist.")
        if cloth_type in ("lower", "overall") and thighs < 1.5:
            what = "Bottoms" if cloth_type == "lower" else "Dresses"
            alt = " Choose 'Top' instead, or" if torso >= 2 else ""
            raise ValueError(f"Your legs aren't in this photo, so there's nowhere to put the "
                             f"{what.lower()}.{alt} take a full-body photo (head to feet).")
        if cloth_type in ("lower", "overall") and shins < 0.5:
            return ("Your photo ends around the knees, so long pants or dresses will come out "
                    "shorter. A head-to-feet photo works best.")
        return None

    def auto_mask(self, person: Image.Image, cloth_type: str) -> Image.Image:
        """Returns the clothing mask at the person photo's original resolution."""
        self._check_type(cloth_type)
        self.load()
        frame = Framing(person)
        with self.lock, torch.inference_mode():
            m = self._auto_mask_model_space(frame.to_model(person), cloth_type)
        return frame.from_model(m.convert("L"), is_mask=True).point(lambda v: 255 if v > 127 else 0)

    @staticmethod
    def _check_type(cloth_type: str) -> None:
        if cloth_type not in CLOTH_TYPES:
            raise ValueError(f"cloth_type must be one of {CLOTH_TYPES}")

    # ------------------------------------------------------------------ try-on
    def try_on(
        self,
        person: Image.Image,
        garment: Image.Image,
        cloth_type: str = "upper",
        steps: int = 50,
        guidance: float = 2.5,
        seed: int = -1,
        mask: Optional[Image.Image] = None,
        keep_original: bool = True,
        progress: Optional[Callable[[float, str], None]] = None,
    ) -> TryOnResult:
        self._check_type(cloth_type)
        steps = int(max(10, min(100, steps)))
        guidance = float(max(1.0, min(7.5, guidance)))
        if seed is None or seed < 0:
            seed = int(np.random.randint(0, 2**31 - 1))
        report = progress or (lambda *_: None)

        report(0.02, "Loading models")
        self.load()
        t0 = time.time()
        frame = Framing(person)
        person_m = frame.to_model(person)
        garment_m = self._prepare_garment(garment)

        with self.lock, torch.inference_mode():
            report(0.05, "Detecting body & clothing")
            warning = None
            if mask is None:
                mask_m = self._auto_mask_model_space(person_m, cloth_type)
                warning = self.last_warning
            else:
                mask_m = frame.to_model(mask.convert("L").resize(person.size, Image.NEAREST), is_mask=True)
                mask_m = mask_m.point(lambda v: 255 if v > 127 else 0)
                if np.array(mask_m).max() == 0:
                    raise ValueError("The painted mask is empty.")
            mask_m_blur = self.mask_processor.blur(mask_m, blur_factor=9)

            report(0.12, "Generating outfit")
            sched = self.pipeline.noise_scheduler
            orig_step = sched.step
            counter = {"i": 0}

            def step_hook(*a, **k):
                counter["i"] += 1
                report(0.12 + 0.83 * counter["i"] / steps, f"Generating outfit ({counter['i']}/{steps})")
                return orig_step(*a, **k)

            sched.step = step_hook
            try:
                gen = torch.Generator(device=self.device).manual_seed(seed)
                raw = self.pipeline(
                    image=person_m,
                    condition_image=garment_m,
                    mask=mask_m_blur,
                    num_inference_steps=steps,
                    guidance_scale=guidance,
                    height=HEIGHT,
                    width=WIDTH,
                    generator=gen,
                )[0]
            finally:
                sched.step = orig_step
            if self.device == "cuda":
                torch.cuda.empty_cache()
        if self.safety_check and self._is_nsfw_placeholder(raw):
            raise ValueError("The result was blocked by the safety filter. Try a different photo or garment.")

        report(0.97, "Blending into your photo")
        if keep_original:
            final = self._composite(person, frame, raw, mask_m)
            mask_full = frame.from_model(mask_m.convert("L"), is_mask=True)
        else:
            final = raw
            mask_full = mask_m
        report(1.0, "Done")
        return TryOnResult(image=final, raw=raw, mask=mask_full, seed=seed, seconds=time.time() - t0,
                           warning=warning)

    @staticmethod
    def _is_nsfw_placeholder(img: Image.Image) -> bool:
        ph = Image.open(CATVTON_DIR / "resource" / "img" / "NSFW.jpg").convert("RGB").resize(img.size)
        diff = np.abs(np.asarray(ph, np.int16) - np.asarray(img.convert("RGB"), np.int16)).mean()
        return diff < 2.0

    @staticmethod
    def _prepare_garment(garment: Image.Image) -> Image.Image:
        g = garment
        if g.mode in ("RGBA", "LA") or (g.mode == "P" and "transparency" in g.info):
            g = g.convert("RGBA")
            bg = Image.new("RGBA", g.size, (255, 255, 255, 255))
            g = Image.alpha_composite(bg, g)
        g = g.convert("RGB")
        # resize & pad onto white 3:4 canvas (same as CatVTON's resize_and_padding)
        w, h = g.size
        if w / h < WIDTH / HEIGHT:
            nh, nw = HEIGHT, w * HEIGHT // h
        else:
            nw, nh = WIDTH, h * WIDTH // w
        g = g.resize((nw, nh), Image.LANCZOS)
        canvas = Image.new("RGB", (WIDTH, HEIGHT), (255, 255, 255))
        canvas.paste(g, ((WIDTH - nw) // 2, (HEIGHT - nh) // 2))
        return canvas

    @staticmethod
    def _composite(person: Image.Image, frame: Framing, raw: Image.Image, mask_m: Image.Image) -> Image.Image:
        """Blend generated pixels into the original photo inside a feathered, slightly grown mask."""
        m = np.array(mask_m.convert("L")).astype(np.uint8)
        k = max(3, (max(m.shape) // 64) | 1)
        m = cv2.dilate(m, np.ones((k, k), np.uint8), iterations=1)
        m = cv2.GaussianBlur(m, (0, 0), sigmaX=k / 2)
        mask_full = frame.from_model(Image.fromarray(m), is_mask=True)
        gen_full = frame.from_model(raw)
        alpha = (np.array(mask_full).astype(np.float32) / 255.0)[..., None]
        base = np.array(person).astype(np.float32)
        gen = np.array(gen_full).astype(np.float32)
        out = gen * alpha + base * (1 - alpha)
        return Image.fromarray(out.clip(0, 255).astype(np.uint8))

    # ------------------------------------------------------------------ info
    def info(self) -> dict:
        d = {
            "ready": self.ready,
            "device": self.device,
            "dtype": str(self.dtype).replace("torch.", ""),
            "safety_check": self.safety_check,
            "load_seconds": round(self.load_seconds, 1),
            "model": "CatVTON (mix-48k-1024) on SD-1.5 inpainting",
        }
        if self.device == "cuda":
            free, total = torch.cuda.mem_get_info()
            d.update(gpu=torch.cuda.get_device_name(0), vram_free_gb=round(free / 2**30, 1),
                     vram_total_gb=round(total / 2**30, 1))
        return d
