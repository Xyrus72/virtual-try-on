# FitRoom: local virtual dress try-on

Upload a photo of yourself (or take one with the camera), pick a top, bottom or dress, and FitRoom generates a photo of you wearing it. Everything runs locally on your GPU. It's free and nothing gets uploaded anywhere.

## Start

Double-click **`run.bat`**. The browser opens at http://127.0.0.1:7860.
Loading the models takes about 30–60 s after start (the status pill at the top turns green).

First time on a new machine, `run.bat` runs `setup.ps1` automatically. That step installs Python 3.11 + CUDA PyTorch and downloads about 9 GB.

## Use your phone as the camera (no webcam needed)

1. Start the app with `run.bat` and connect the phone to the **same Wi-Fi** as the PC.
2. Click **Phone camera** on the PC and scan the QR code with your phone.
3. On the phone, tap **Back camera** (someone holds the phone) or **Selfie camera** and take the photo.
   It appears on the PC within a second or two. You can also send garment photos the same way.

The phone uses its own camera app, so you get full photo quality and it works over plain Wi-Fi (no app install, no HTTPS).
On the first start Windows asks whether Python may use the network: tick **Private networks** and allow it,
otherwise the phone can't reach the PC. `run.bat` makes the app visible on your local network only
(anyone on the same Wi-Fi could open it). Set `VTON_HOST=127.0.0.1` to switch phone access off.

## How to get the best results

| Do | Avoid |
|---|---|
| Front-facing photo, full or half body, good light | Side poses, crossed arms over the outfit |
| Garment as a product/flat-lay shot on plain background | Garment photos worn by another person in a busy scene |
| Pick the right type: **Top / Bottom / Dress** | Using "Top" for a dress |
| 50 steps for final, 30 for quick previews | Guidance above ~3.5 (adds artifacts) |

Use **Show auto mask** to see which area will be replaced (red). Paint or erase to fine-tune, e.g. to make a top longer or to protect a hand.
**New variation** re-runs with a new random seed. **Keep original photo quality** pastes the generated outfit back into your full-resolution photo, so your face and background stay untouched.

## How it works

1. **Framing**: the photo is fit to the model's 3:4 canvas without cropping the person (tall photos are padded, wide ones cropped at the sides).
2. **Auto-masking**: DensePose (body parts) and two SCHP human parsers (LIP + ATR clothing labels) work out which region to replace for the chosen garment type, while keeping the face, hair, hands and other clothes.
3. **Try-on**: [CatVTON](https://arxiv.org/abs/2407.15886) (ICLR 2025) runs a Stable-Diffusion-1.5 inpainting UNet in bf16. Its fine-tuned self-attention looks at the person and the garment side by side. It was trained on the VITON-HD + DressCode datasets (`mix-48k-1024` weights).
4. **Compositing**: the 768×1024 result is upscaled and blended back into the original photo with a feathered mask.

About 6–8 GB VRAM. Measured on an RTX 5060 Ti: ~28 s for 30 steps, ~45 s for 50 steps (models load in ~15 s).

## Project layout

```
app/vton_engine.py   inference engine (framing, masking, CatVTON, compositing)
app/server.py        FastAPI server + job queue + history
app/static/          web UI (vanilla HTML/CSS/JS)
gallery/             sample people + garments (VITON-HD / DressCode examples)
outputs/             every result you generate (+ mask, inputs, settings JSON)
scripts/             download_models.py, build_gallery.py, smoke_test.py
third_party/         CatVTON code (HF Space) + detectron2 (pure-python, no compile)
models/hf/           downloaded weights (HF cache)
```

Command-line test: `.venv\Scripts\python -m scripts.smoke_test --steps 30` (writes to `outputs/_smoke/`).

Settings via environment variables: `VTON_PORT` (default 7860), `VTON_HOST` (set `0.0.0.0` to use from your phone on the same Wi-Fi), `VTON_SAFETY_CHECK=0` to disable the NSFW filter.

## Troubleshooting

- **"GPU ran out of memory"**: other apps are holding VRAM (e.g. an Ollama model). Stop them with `ollama stop <model>` and retry.
- **"No person detected"**: the photo needs a clearly visible person, at least from head to waist.
- Results look wrong for a dress: make sure **Dress / Full** is selected.

## License

CatVTON weights are **CC BY-NC-SA 4.0**, and the sample images come from research datasets. For personal and non-commercial use only.
