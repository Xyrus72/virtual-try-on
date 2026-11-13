"""FitRoom — local virtual dress try-on web app (FastAPI + CatVTON).

Run:  .venv\\Scripts\\python -m app.server      (then open http://127.0.0.1:7860)
"""
from __future__ import annotations

import io
import json
import os
import queue
import threading
import time
import traceback
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError

from app.vton_engine import CLOTH_TYPES, ROOT, TryOnEngine

STATIC = ROOT / "app" / "static"
GALLERY = ROOT / "gallery"
OUTPUTS = ROOT / "outputs"
OUTPUTS.mkdir(exist_ok=True)
IMG_EXT = {".jpg", ".jpeg", ".png", ".webp"}
MAX_UPLOAD = 25 * 1024 * 1024

engine = TryOnEngine()
load_state = {"status": "loading", "error": None}
jobs: dict[str, dict] = {}
job_queue: "queue.Queue[str]" = queue.Queue()

app = FastAPI(title="FitRoom Virtual Try-On")


@app.middleware("http")
async def _no_cache_ui(request, call_next):
    resp = await call_next(request)
    if request.url.path == "/" or request.url.path.startswith("/static/"):
        resp.headers["Cache-Control"] = "no-cache"
    return resp


# ---------------------------------------------------------------- background workers
def _load_models() -> None:
    try:
        engine.load()
        load_state["status"] = "ready"
    except Exception as e:  # pragma: no cover - surfaced in UI
        traceback.print_exc()
        load_state.update(status="error", error=f"{type(e).__name__}: {e}")


def _worker() -> None:
    while True:
        job_id = job_queue.get()
        job = jobs.get(job_id)
        if not job:
            continue
        job.update(status="running", progress=0.01, message="Starting")

        def progress(p: float, msg: str) -> None:
            job.update(progress=round(p, 3), message=msg)

        try:
            p = job.pop("_params")
            res = engine.try_on(progress=progress, **p)
            day = datetime.now().strftime("%Y%m%d")
            out_dir = OUTPUTS / day
            out_dir.mkdir(parents=True, exist_ok=True)
            stem = f"{datetime.now().strftime('%H%M%S')}_{job_id[:6]}"
            res.image.save(out_dir / f"{stem}.jpg", quality=95)
            res.raw.save(out_dir / f"{stem}_raw.png")
            res.mask.convert("L").save(out_dir / f"{stem}_mask.png")
            p["person"].save(out_dir / f"{stem}_person.jpg", quality=90)
            p["garment"].convert("RGB").save(out_dir / f"{stem}_garment.jpg", quality=90)
            meta = {
                "id": stem, "day": day, "created": time.time(), "cloth_type": p["cloth_type"],
                "steps": p["steps"], "guidance": p["guidance"], "seed": res.seed,
                "seconds": round(res.seconds, 1), "warning": res.warning,
            }
            (out_dir / f"{stem}.json").write_text(json.dumps(meta))
            job.update(status="done", progress=1.0, message="Done", result=_urls(day, stem) | meta)
        except ValueError as e:
            job.update(status="error", error=str(e))
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            msg = str(e)
            if "out of memory" in msg.lower():
                msg = ("GPU ran out of memory. Close other GPU apps (e.g. Ollama models) "
                       "or lower the steps, then try again.")
            job.update(status="error", error=msg)
        finally:
            for k in list(jobs)[:-200]:  # bound memory
                jobs.pop(k, None)


def _urls(day: str, stem: str) -> dict:
    base = f"/outputs/{day}/{stem}"
    return {"image_url": f"{base}.jpg", "raw_url": f"{base}_raw.png", "mask_url": f"{base}_mask.png",
            "person_url": f"{base}_person.jpg", "garment_url": f"{base}_garment.jpg"}


@app.on_event("startup")
def _startup() -> None:
    threading.Thread(target=_load_models, daemon=True).start()
    threading.Thread(target=_worker, daemon=True).start()


# ---------------------------------------------------------------- input helpers
async def _read_image(upload: Optional[UploadFile], ref: Optional[str], what: str,
                      keep_alpha: bool = False) -> Image.Image:
    if upload is not None and upload.filename:
        data = await upload.read()
        if len(data) > MAX_UPLOAD:
            raise HTTPException(413, f"{what} image is larger than 25 MB.")
        try:
            img = Image.open(io.BytesIO(data))
            img.load()
        except (UnidentifiedImageError, OSError):
            raise HTTPException(400, f"Could not read the {what} image. Use JPG, PNG or WEBP.")
    elif ref:
        path = (ROOT / ref.lstrip("/")).resolve()
        allowed = (GALLERY.resolve(), OUTPUTS.resolve())
        if not any(str(path).startswith(str(a) + os.sep) for a in allowed) or not path.is_file():
            raise HTTPException(400, f"Unknown {what} image.")
        img = Image.open(path)
        img.load()
    else:
        raise HTTPException(400, f"Please choose a {what} image.")
    if keep_alpha:
        from PIL import ImageOps
        return ImageOps.exif_transpose(img)
    return engine.prepare_upload(img)


# ---------------------------------------------------------------- API
@app.get("/api/status")
def status() -> dict:
    return {"status": load_state["status"], "error": load_state["error"],
            "queue": job_queue.qsize(), **engine.info()}


@app.get("/api/gallery")
def gallery() -> dict:
    def items(folder: Path) -> list[dict]:
        if not folder.exists():
            return []
        return [{"url": f"/gallery/{f.relative_to(GALLERY).as_posix()}",
                 "ref": f"gallery/{f.relative_to(GALLERY).as_posix()}", "name": f.stem}
                for f in sorted(folder.iterdir()) if f.suffix.lower() in IMG_EXT]

    return {
        "people": items(GALLERY / "people"),
        "garments": {t: items(GALLERY / "garments" / t) for t in CLOTH_TYPES},
    }


@app.post("/api/mask")
async def make_mask(person: Optional[UploadFile] = File(None), person_ref: Optional[str] = Form(None),
                    cloth_type: str = Form("upper")) -> Response:
    if load_state["status"] != "ready":
        raise HTTPException(503, "Models are still loading, please wait a moment.")
    if cloth_type not in CLOTH_TYPES:
        raise HTTPException(400, "Invalid garment type.")
    img = await _read_image(person, person_ref, "person")
    try:
        import anyio
        mask = await anyio.to_thread.run_sync(engine.auto_mask, img, cloth_type)
    except ValueError as e:
        raise HTTPException(422, str(e))
    buf = io.BytesIO()
    mask.save(buf, "PNG")
    headers = {"X-Warning": engine.last_warning} if engine.last_warning else None
    return Response(buf.getvalue(), media_type="image/png", headers=headers)


@app.post("/api/tryon")
async def tryon(
    person: Optional[UploadFile] = File(None), person_ref: Optional[str] = Form(None),
    garment: Optional[UploadFile] = File(None), garment_ref: Optional[str] = Form(None),
    mask: Optional[UploadFile] = File(None),
    cloth_type: str = Form("upper"), steps: int = Form(50), guidance: float = Form(2.5),
    seed: int = Form(-1), keep_original: bool = Form(True),
) -> dict:
    if load_state["status"] == "error":
        raise HTTPException(500, f"Model failed to load: {load_state['error']}")
    if cloth_type not in CLOTH_TYPES:
        raise HTTPException(400, "Invalid garment type.")
    person_img = await _read_image(person, person_ref, "person")
    garment_img = await _read_image(garment, garment_ref, "garment", keep_alpha=True)
    mask_img = None
    if mask is not None and mask.filename:
        mask_img = Image.open(io.BytesIO(await mask.read())).convert("L")
    job_id = uuid.uuid4().hex
    jobs[job_id] = {
        "id": job_id, "status": "queued", "progress": 0.0, "message": "Waiting in queue",
        "_params": dict(person=person_img, garment=garment_img, cloth_type=cloth_type, steps=steps,
                        guidance=guidance, seed=seed, mask=mask_img, keep_original=keep_original),
    }
    job_queue.put(job_id)
    return {"job_id": job_id}


# ---------------------------------------------------------------- phone camera relay
# The PC page shows a QR code -> phone opens /phone on the same Wi-Fi -> phone takes a photo with
# its native camera (file input with `capture`, works over plain HTTP) -> PC page picks it up.
PHONE_DIR = ROOT / "runtime" / "phone"
PHONE_DIR.mkdir(parents=True, exist_ok=True)
phone_inbox: dict[str, dict] = {}


def _check_session(session: str) -> None:
    if not (8 <= len(session) <= 40 and session.isalnum()):
        raise HTTPException(400, "Bad session.")


def _lan_ip() -> Optional[str]:
    import socket
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("10.255.255.255", 1))  # no packet is sent; just picks the LAN interface
            return s.getsockname()[0]
    except OSError:
        return None


@app.get("/api/phone/link")
def phone_link(session: str) -> dict:
    _check_session(session)
    import qrcode
    import qrcode.image.svg

    host = os.environ.get("VTON_HOST", "127.0.0.1")
    port = int(os.environ.get("VTON_PORT", "7860"))
    ip = _lan_ip()
    lan = host == "0.0.0.0" and ip is not None
    url = f"http://{ip if lan else '127.0.0.1'}:{port}/phone?s={session}"
    img = qrcode.make(url, image_factory=qrcode.image.svg.SvgPathImage, box_size=10, border=2)
    return {"url": url, "lan": lan, "qr_svg": img.to_string(encoding="unicode")}


@app.get("/phone")
def phone_page() -> FileResponse:
    return FileResponse(STATIC / "phone.html")


@app.post("/api/phone/{session}/{kind}")
async def phone_upload(session: str, kind: str, photo: UploadFile = File(...)) -> dict:
    _check_session(session)
    if kind not in ("person", "garment"):
        raise HTTPException(400, "kind must be person or garment")
    img = await _read_image(photo, None, kind, keep_alpha=False)
    name = f"{session}_{kind}_{int(time.time() * 1000)}.jpg"
    img.save(PHONE_DIR / name, quality=93)
    phone_inbox.setdefault(session, {})[kind] = {"url": f"/phone-files/{name}", "t": time.time()}
    return {"ok": True}


@app.get("/api/phone/{session}")
def phone_poll(session: str) -> dict:
    _check_session(session)
    return phone_inbox.get(session, {})


@app.get("/api/jobs/{job_id}")
def job(job_id: str) -> dict:
    j = jobs.get(job_id)
    if not j:
        raise HTTPException(404, "Job not found.")
    return {k: v for k, v in j.items() if not k.startswith("_")}


@app.get("/api/history")
def history(limit: int = 60) -> list[dict]:
    metas = sorted(OUTPUTS.glob("*/*.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:limit]
    out = []
    for m in metas:
        try:
            meta = json.loads(m.read_text())
            out.append(_urls(meta["day"], meta["id"]) | meta)
        except Exception:  # noqa: BLE001
            continue
    return out


@app.delete("/api/history/{day}/{stem}")
def delete_history(day: str, stem: str) -> dict:
    if not day.isdigit() or not all(c.isalnum() or c == "_" for c in stem):
        raise HTTPException(400, "Bad id.")
    removed = 0
    for f in (OUTPUTS / day).glob(f"{stem}*"):
        f.unlink()
        removed += 1
    return {"removed": removed}


app.mount("/gallery", StaticFiles(directory=GALLERY, check_dir=False), name="gallery")
app.mount("/outputs", StaticFiles(directory=OUTPUTS), name="outputs")
app.mount("/static", StaticFiles(directory=STATIC), name="static")
app.mount("/phone-files", StaticFiles(directory=PHONE_DIR), name="phone-files")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


if __name__ == "__main__":
    import uvicorn

    host = os.environ.get("VTON_HOST", "127.0.0.1")
    port = int(os.environ.get("VTON_PORT", "7860"))
    print(f"\n  FitRoom starting -> http://{host}:{port}\n")
    uvicorn.run(app, host=host, port=port, log_level="warning")
