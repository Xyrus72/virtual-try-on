"use strict";
(() => {
  const $ = (id) => document.getElementById(id);
  const state = {
    person: null,      // {blob?, ref?, url}
    garment: null,     // {blob?, ref?, url}
    type: "upper",
    gallery: { people: [], garments: { upper: [], lower: [], overall: [] } },
    maskActive: false,
    maskFromAuto: false,
    brush: "add",
    ready: false,
    busy: false,
    lastSeed: null,
    history: [],
    lbItem: null,
  };

  // ------------------------------------------------------------ toast
  let toastTimer;
  function toast(msg, isErr = false) {
    const t = $("toast");
    t.textContent = msg;
    t.className = "toast show" + (isErr ? " err" : "");
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => (t.className = "toast"), isErr ? 6000 : 3000);
  }

  async function api(url, opts = {}) {
    const r = await fetch(url, opts);
    if (!r.ok) {
      let msg = `${r.status} ${r.statusText}`;
      try { const j = await r.json(); if (j.detail) msg = typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail); } catch {}
      throw new Error(msg);
    }
    return r;
  }

  // ------------------------------------------------------------ status
  async function pollStatus() {
    try {
      const s = await (await api("/api/status")).json();
      const el = $("status");
      if (s.status === "ready") {
        state.ready = true;
        el.className = "status ready";
        $("statusText").textContent = s.gpu ? `Ready · ${s.gpu.replace("NVIDIA GeForce ", "")}` : "Ready · CPU (slow)";
        el.title = `${s.model}\n${s.dtype}${s.vram_free_gb != null ? ` · ${s.vram_free_gb}/${s.vram_total_gb} GB VRAM free` : ""}`;
      } else if (s.status === "error") {
        el.className = "status error";
        $("statusText").textContent = "Model failed to load";
        el.title = s.error || "";
        toast(`Model failed to load: ${s.error}`, true);
        return;
      } else {
        el.className = "status";
        $("statusText").textContent = "Loading AI models…";
      }
      updateGo();
      setTimeout(pollStatus, state.ready ? 20000 : 1500);
    } catch {
      $("status").className = "status error";
      $("statusText").textContent = "Server offline";
      setTimeout(pollStatus, 3000);
    }
  }

  // ------------------------------------------------------------ gallery
  function thumb(item, onPick, group) {
    const b = document.createElement("button");
    b.type = "button";
    b.className = "thumb";
    b.setAttribute("aria-pressed", "false");
    b.title = item.name;
    b.dataset.ref = item.ref;
    const img = new Image();
    img.loading = "lazy";
    img.src = item.url;
    img.alt = item.name;
    b.append(img);
    b.addEventListener("click", () => {
      group.querySelectorAll(".thumb").forEach((t) => t.setAttribute("aria-pressed", "false"));
      b.setAttribute("aria-pressed", "true");
      onPick(item);
    });
    return b;
  }

  function renderPeople() {
    const g = $("peopleGrid");
    g.replaceChildren(...state.gallery.people.map((it) => thumb(it, (i) => setPerson({ ref: i.ref, url: i.url }), g)));
  }
  function renderGarments() {
    const g = $("garmentGrid");
    const list = state.gallery.garments[state.type] || [];
    g.replaceChildren(...list.map((it) => thumb(it, (i) => setGarment({ ref: i.ref, url: i.url }), g)));
    if (state.garment?.ref) {
      const sel = g.querySelector(`[data-ref="${CSS.escape(state.garment.ref)}"]`);
      if (sel) sel.setAttribute("aria-pressed", "true");
    }
  }
  async function loadGallery() {
    try {
      state.gallery = await (await api("/api/gallery")).json();
      renderPeople();
      renderGarments();
    } catch (e) { toast(`Could not load gallery: ${e.message}`, true); }
  }

  // ------------------------------------------------------------ person + mask canvas
  const personImg = $("personImg");
  const maskCanvas = $("maskCanvas");
  const mctx = maskCanvas.getContext("2d");

  function setPerson(p) {
    stopCamera();
    if (state.person?.blob) URL.revokeObjectURL(state.person.url);
    state.person = p;
    $("personDrop").hidden = true;
    $("personWrap").hidden = false;
    personImg.hidden = false;
    personImg.onload = () => {
      const scale = Math.min(1, 1024 / Math.max(personImg.naturalWidth, personImg.naturalHeight));
      maskCanvas.width = Math.round(personImg.naturalWidth * scale);
      maskCanvas.height = Math.round(personImg.naturalHeight * scale);
      sizeCanvasToImage();
      clearMask();
    };
    personImg.src = p.url;
    $("maskTools").hidden = false;
    if (!p.ref || !p.ref.startsWith("gallery/")) $("peopleGrid").querySelectorAll(".thumb").forEach((t) => t.setAttribute("aria-pressed", "false"));
    updateGo();
  }

  function sizeCanvasToImage() {
    // place the canvas exactly over the letterboxed image content so brush coordinates line up
    const W = personImg.clientWidth, H = personImg.clientHeight;
    const nw = personImg.naturalWidth, nh = personImg.naturalHeight;
    if (!W || !nw) return;
    const s = Math.min(W / nw, H / nh);
    const w = nw * s, h = nh * s;
    Object.assign(maskCanvas.style, { width: `${w}px`, height: `${h}px`, left: `${(W - w) / 2}px`, top: `${(H - h) / 2}px` });
  }
  new ResizeObserver(sizeCanvasToImage).observe($("personWrap"));

  function clearMask() {
    mctx.clearRect(0, 0, maskCanvas.width, maskCanvas.height);
    state.maskActive = false;
    state.maskFromAuto = false;
    $("personWrap").classList.remove("editing");
    $("autoMaskBtn").textContent = "Show auto mask";
  }

  async function showAutoMask() {
    if (!state.person) return;
    if (!state.ready) return toast("Models are still loading — one moment.");
    const btn = $("autoMaskBtn");
    btn.disabled = true;
    btn.textContent = "Detecting…";
    try {
      const fd = new FormData();
      appendImage(fd, "person", state.person);
      fd.append("cloth_type", state.type);
      const resp = await api("/api/mask", { method: "POST", body: fd });
      const warn = resp.headers.get("X-Warning");
      if (warn) toast(warn);
      const blob = await resp.blob();
      const img = await blobToImage(blob);
      const tmp = document.createElement("canvas");
      tmp.width = maskCanvas.width;
      tmp.height = maskCanvas.height;
      const t = tmp.getContext("2d");
      t.drawImage(img, 0, 0, tmp.width, tmp.height);
      const d = t.getImageData(0, 0, tmp.width, tmp.height);
      for (let i = 0; i < d.data.length; i += 4) {
        const on = d.data[i] > 127;
        d.data[i] = 225; d.data[i + 1] = 45; d.data[i + 2] = 45; d.data[i + 3] = on ? 255 : 0;
      }
      mctx.putImageData(d, 0, 0);
      state.maskActive = true;
      state.maskFromAuto = true;
      $("personWrap").classList.add("editing");
      btn.textContent = "Refresh auto mask";
    } catch (e) {
      toast(e.message, true);
      btn.textContent = "Show auto mask";
    } finally { btn.disabled = false; }
  }

  // brush
  let drawing = false;
  function paint(ev) {
    const r = maskCanvas.getBoundingClientRect();
    const sx = maskCanvas.width / r.width;
    const x = (ev.clientX - r.left) * sx;
    const y = (ev.clientY - r.top) * (maskCanvas.height / r.height);
    const rad = (+$("brushSize").value / 2) * sx;
    mctx.globalCompositeOperation = state.brush === "add" ? "source-over" : "destination-out";
    mctx.fillStyle = "rgb(225,45,45)";
    mctx.beginPath();
    mctx.arc(x, y, rad, 0, Math.PI * 2);
    mctx.fill();
  }
  maskCanvas.addEventListener("pointerdown", (e) => { drawing = true; maskCanvas.setPointerCapture(e.pointerId); state.maskActive = true; paint(e); });
  maskCanvas.addEventListener("pointermove", (e) => drawing && paint(e));
  maskCanvas.addEventListener("pointerup", () => (drawing = false));
  maskCanvas.addEventListener("pointercancel", () => (drawing = false));

  document.querySelectorAll("[data-brush]").forEach((b) =>
    b.addEventListener("click", () => {
      state.brush = b.dataset.brush;
      document.querySelectorAll("[data-brush]").forEach((x) => x.setAttribute("aria-pressed", String(x === b)));
      $("personWrap").classList.add("editing");
    })
  );
  $("autoMaskBtn").addEventListener("click", showAutoMask);
  $("clearMaskBtn").addEventListener("click", clearMask);

  function maskBlob() {
    const out = document.createElement("canvas");
    out.width = maskCanvas.width;
    out.height = maskCanvas.height;
    const o = out.getContext("2d");
    const d = mctx.getImageData(0, 0, out.width, out.height);
    for (let i = 0; i < d.data.length; i += 4) {
      const v = d.data[i + 3] > 20 ? 255 : 0;
      d.data[i] = d.data[i + 1] = d.data[i + 2] = v; d.data[i + 3] = 255;
    }
    o.putImageData(d, 0, 0);
    return new Promise((res) => out.toBlob(res, "image/png"));
  }

  // ------------------------------------------------------------ laptop / USB camera
  // Preview is mirrored like a mirror; the saved photo is not (so logos/text read correctly).
  // A 3:4 guide shows exactly the part of the frame that is kept. Countdown lets you step back.
  let stream = null;
  let countdown = null; // {cancel: bool}
  const video = $("cameraVideo");

  function cameraError(e) {
    const n = e && e.name;
    if (n === "NotAllowedError" || n === "SecurityError")
      return "Camera permission is blocked. Click the camera icon in the address bar, allow it, then try again.";
    if (n === "NotFoundError" || n === "OverconstrainedError")
      return "No camera found on this PC. Use the Phone camera button instead.";
    if (n === "NotReadableError" || n === "AbortError")
      return "The camera is busy in another app (Zoom, Teams, Camera…). Close it and try again.";
    return `Camera unavailable: ${e && e.message ? e.message : e}`;
  }

  async function openStream(deviceId) {
    const base = { width: { ideal: 1920 }, height: { ideal: 1080 } };
    const video = deviceId ? { ...base, deviceId: { exact: deviceId } } : { ...base, facingMode: "user" };
    try {
      return await navigator.mediaDevices.getUserMedia({ video, audio: false });
    } catch (e) {
      if (e.name === "OverconstrainedError") return navigator.mediaDevices.getUserMedia({ video: true, audio: false });
      throw e;
    }
  }

  async function fillDevices(activeId) {
    try {
      const cams = (await navigator.mediaDevices.enumerateDevices()).filter((d) => d.kind === "videoinput");
      const sel = $("camDevice");
      sel.replaceChildren(...cams.map((d, i) => {
        const o = document.createElement("option");
        o.value = d.deviceId;
        o.textContent = d.label || `Camera ${i + 1}`;
        o.selected = d.deviceId === activeId;
        return o;
      }));
      sel.hidden = cams.length < 2;
    } catch {}
  }

  function positionGuide() {
    const box = $("cam");
    const vw = video.videoWidth, vh = video.videoHeight;
    if (!vw) return;
    const ar = `${vw} / ${vh}`;
    if (box.style.aspectRatio !== ar) box.style.aspectRatio = ar; // widescreen webcams fill the panel
    if (!box.clientWidth) return;
    const s = Math.min(box.clientWidth / vw, box.clientHeight / vh); // object-fit: contain
    const dw = vw * s, dh = vh * s;
    const gw = Math.min(dw, dh * 3 / 4), gh = Math.min(dh, dw * 4 / 3);
    Object.assign($("camGuide").style, {
      width: `${gw}px`, height: `${gh}px`,
      left: `${(box.clientWidth - gw) / 2}px`, top: `${(box.clientHeight - gh) / 2}px`,
    });
  }
  new ResizeObserver(positionGuide).observe($("cam"));
  video.addEventListener("loadedmetadata", positionGuide);

  async function startCamera(deviceId) {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      toast("The camera only works when the app is opened on this PC (http://127.0.0.1:7860). Use the Phone camera button instead.", true);
      return;
    }
    const btn = $("cameraBtn");
    btn.disabled = true;
    try {
      const s = await openStream(deviceId);
      if (stream) stream.getTracks().forEach((t) => t.stop());
      stream = s;
      video.srcObject = stream;
      video.play().catch(() => {}); // don't block the UI on play(); muted inline video autoplays
      $("personDrop").hidden = true;
      $("personWrap").hidden = true;
      $("maskTools").hidden = true;
      $("maskTip").hidden = true;
      $("cam").hidden = false;
      $("camBar").hidden = false;
      positionGuide();
      fillDevices(stream.getVideoTracks()[0].getSettings().deviceId);
    } catch (e) {
      toast(cameraError(e), true);
    } finally { btn.disabled = false; }
  }

  function stopCamera() {
    if (countdown) countdown.cancel = true;
    countdown = null;
    $("camCount").hidden = true;
    if (stream) stream.getTracks().forEach((t) => t.stop());
    stream = null;
    video.srcObject = null;
    $("cam").hidden = true;
    $("camBar").hidden = true;
    $("maskTip").hidden = false;
    $("camShoot").disabled = false;
  }

  function cancelCamera() {
    stopCamera();
    const has = !!state.person;
    $("personDrop").hidden = has;
    $("personWrap").hidden = !has;
    $("maskTools").hidden = !has;
  }

  async function shoot() {
    if (!stream || countdown) return;
    const secs = +$("camTimer").value;
    const cd = (countdown = { cancel: false });
    $("camShoot").disabled = true;
    const count = $("camCount");
    for (let i = secs; i > 0; i--) {
      count.hidden = false;
      count.textContent = i;
      await new Promise((r) => setTimeout(r, 1000));
      if (cd.cancel) return;
    }
    count.hidden = true;
    countdown = null;
    // capture the 3:4 area shown by the guide, un-mirrored, at full camera resolution
    const vw = video.videoWidth, vh = video.videoHeight;
    const cw = Math.round(Math.min(vw, vh * 3 / 4)), ch = Math.round(Math.min(vh, vw * 4 / 3));
    const c = document.createElement("canvas");
    c.width = cw; c.height = ch;
    c.getContext("2d").drawImage(video, (vw - cw) / 2, (vh - ch) / 2, cw, ch, 0, 0, cw, ch);
    const flash = $("camFlash");
    flash.classList.add("on");
    requestAnimationFrame(() => requestAnimationFrame(() => flash.classList.remove("on")));
    const blob = await new Promise((r) => c.toBlob(r, "image/jpeg", 0.95));
    setPerson({ blob, url: URL.createObjectURL(blob) }); // also stops the camera
    toast("Photo taken. Pick a garment and press Try it on.");
  }

  $("cameraBtn").addEventListener("click", () => (stream ? shoot() : startCamera()));
  $("camShoot").addEventListener("click", shoot);
  $("camCancel").addEventListener("click", cancelCamera);
  $("camDevice").addEventListener("change", (e) => startCamera(e.target.value));
  document.addEventListener("keydown", (e) => {
    if (!stream || e.target.matches("input, select, textarea")) return;
    if (e.code === "Space") { e.preventDefault(); shoot(); }
  });
  window.addEventListener("pagehide", stopCamera);

  // ------------------------------------------------------------ phone camera (QR relay over Wi-Fi)
  const phoneSession = (() => {
    let s = null;
    try { s = localStorage.getItem("fitroomPhone"); } catch {}
    if (!s || !/^[A-Za-z0-9]{8,40}$/.test(s)) {
      s = Array.from(crypto.getRandomValues(new Uint8Array(12)), (b) => b.toString(16).padStart(2, "0")).join("");
      try { localStorage.setItem("fitroomPhone", s); } catch {}
    }
    return s;
  })();
  const phoneSeen = { person: 0, garment: 0 };
  let phonePollTimer = null;

  async function openPhone() {
    $("phoneModal").hidden = false;
    try {
      const link = await (await api(`/api/phone/link?session=${phoneSession}`)).json();
      $("phoneQr").innerHTML = link.qr_svg; // server-generated SVG (our own endpoint)
      $("phoneUrl").textContent = link.url;
      $("phoneWarn").hidden = link.lan;
    } catch (e) { toast(`Could not create phone link: ${e.message}`, true); }
    startPhonePolling();
  }
  function closePhone() { $("phoneModal").hidden = true; }

  function startPhonePolling() {
    if (phonePollTimer) return;
    const tick = async () => {
      try {
        const inbox = await (await api(`/api/phone/${phoneSession}`)).json();
        for (const kind of ["person", "garment"]) {
          const item = inbox[kind];
          if (!item || item.t <= phoneSeen[kind]) continue;
          const first = phoneSeen[kind] === 0 && performance.now() < 3000; // ignore stale photo from a previous page load
          phoneSeen[kind] = item.t;
          if (first) continue;
          const blob = await (await api(item.url)).blob();
          const entry = { blob, url: URL.createObjectURL(blob) };
          if (kind === "person") setPerson(entry); else setGarment(entry);
          closePhone();
          toast(kind === "person" ? "Photo received from your phone." : "Garment received from your phone.");
        }
      } catch {}
      phonePollTimer = setTimeout(tick, 1500);
    };
    tick();
  }
  $("phoneBtn").addEventListener("click", openPhone);
  $("phoneClose").addEventListener("click", closePhone);
  $("phoneModal").addEventListener("click", (e) => { if (e.target.id === "phoneModal") closePhone(); });

  // ------------------------------------------------------------ garment
  function setGarment(g) {
    if (state.garment?.blob) URL.revokeObjectURL(state.garment.url);
    state.garment = g;
    const pv = $("garmentPreview");
    const img = new Image();
    img.src = g.url;
    img.alt = "Selected garment";
    pv.replaceChildren(img);
    pv.classList.add("has-img");
    if (g.blob) $("garmentGrid").querySelectorAll(".thumb").forEach((t) => t.setAttribute("aria-pressed", "false"));
    updateGo();
  }

  document.querySelectorAll("#typeTabs button").forEach((b) =>
    b.addEventListener("click", () => {
      if (state.type === b.dataset.type) return;
      state.type = b.dataset.type;
      document.querySelectorAll("#typeTabs button").forEach((x) => x.setAttribute("aria-pressed", String(x === b)));
      renderGarments();
      if (state.maskFromAuto) showAutoMask(); // mask depends on garment type
    })
  );

  // ------------------------------------------------------------ uploads / drag & drop
  function fileToEntry(file) {
    if (!file || !file.type.startsWith("image/")) { toast("Please choose an image file.", true); return null; }
    return { blob: file, url: URL.createObjectURL(file) };
  }
  $("personFile").addEventListener("change", (e) => { const f = fileToEntry(e.target.files[0]); if (f) setPerson(f); e.target.value = ""; });
  $("garmentFile").addEventListener("change", (e) => { const f = fileToEntry(e.target.files[0]); if (f) setGarment(f); e.target.value = ""; });
  $("personDrop").addEventListener("click", () => $("personFile").click());
  $("garmentPreview").addEventListener("click", () => $("garmentFile").click());

  function dropZone(el, onFile) {
    el.addEventListener("dragover", (e) => { e.preventDefault(); el.classList.add("over"); });
    el.addEventListener("dragleave", () => el.classList.remove("over"));
    el.addEventListener("drop", (e) => {
      e.preventDefault(); el.classList.remove("over");
      const f = fileToEntry(e.dataTransfer.files[0]);
      if (f) onFile(f);
    });
  }
  dropZone($("personStage"), setPerson);
  dropZone($("garmentPreview"), setGarment);

  // ------------------------------------------------------------ try-on
  function appendImage(fd, name, entry) {
    if (entry.blob) fd.append(name, entry.blob, `${name}.png`);
    else fd.append(`${name}_ref`, entry.ref);
  }

  function updateGo() {
    $("goBtn").disabled = !(state.person && state.garment) || state.busy;
    $("goBtn").textContent = state.busy ? "Working…" : state.ready ? "Try it on" : "Try it on (models loading)";
  }

  async function runTryOn(seedOverride) {
    if (!state.person || !state.garment || state.busy) return;
    state.busy = true;
    updateGo();
    $("rerollBtn").disabled = true;
    const stage = $("resultStage");
    stage.classList.add("busy");
    $("progress").hidden = false;
    setProgress(0, "Sending…");
    try {
      const fd = new FormData();
      appendImage(fd, "person", state.person);
      appendImage(fd, "garment", state.garment);
      if (state.maskActive) fd.append("mask", await maskBlob(), "mask.png");
      fd.append("cloth_type", state.type);
      fd.append("steps", $("steps").value);
      fd.append("guidance", $("guidance").value);
      fd.append("seed", seedOverride ?? $("seed").value);
      fd.append("keep_original", $("keepOriginal").checked ? "true" : "false");
      const { job_id } = await (await api("/api/tryon", { method: "POST", body: fd })).json();
      const res = await waitJob(job_id);
      showResult(res);
      loadHistory();
    } catch (e) {
      toast(e.message, true);
    } finally {
      state.busy = false;
      stage.classList.remove("busy");
      $("progress").hidden = true;
      updateGo();
      $("rerollBtn").disabled = !state.lastSeed && state.lastSeed !== 0;
    }
  }

  function setProgress(p, msg) {
    $("progressBar").style.width = `${Math.round(p * 100)}%`;
    $("progressText").textContent = msg;
  }

  async function waitJob(id) {
    const t0 = performance.now();
    for (;;) {
      await new Promise((r) => setTimeout(r, 600));
      const j = await (await api(`/api/jobs/${id}`)).json();
      const secs = ((performance.now() - t0) / 1000).toFixed(0);
      if (j.status === "done") return j.result;
      if (j.status === "error") throw new Error(j.error || "Try-on failed");
      setProgress(j.progress || 0, `${j.message || "Working"} · ${secs}s`);
    }
  }

  function showResult(r) {
    const bust = `?t=${Date.now()}`;
    $("afterImg").src = r.image_url + bust;
    $("beforeImg").src = r.person_url + bust;
    $("resultEmpty").hidden = true;
    $("compare").hidden = false;
    setSplit(0.5);
    const dl = $("downloadBtn");
    dl.href = r.image_url;
    dl.download = `fitroom_${r.id}.jpg`;
    dl.hidden = false;
    state.lastSeed = r.seed;
    if (r.warning) toast(r.warning);
    $("resultMeta").textContent = `${r.seconds}s · ${r.steps} steps · seed ${r.seed}`;
  }

  // before/after slider
  function setSplit(f) {
    f = Math.max(0, Math.min(1, f));
    $("beforeClip").style.clipPath = `inset(0 ${(1 - f) * 100}% 0 0)`;
    $("handle").style.left = `${f * 100}%`;
  }
  const cmp = $("compare");
  let sliding = false;
  const slide = (e) => { const r = cmp.getBoundingClientRect(); setSplit((e.clientX - r.left) / r.width); };
  cmp.addEventListener("pointerdown", (e) => { sliding = true; cmp.setPointerCapture(e.pointerId); slide(e); });
  cmp.addEventListener("pointermove", (e) => sliding && slide(e));
  cmp.addEventListener("pointerup", () => (sliding = false));

  $("goBtn").addEventListener("click", () => runTryOn());
  $("rerollBtn").addEventListener("click", () => runTryOn(-1));

  // settings outputs
  [["steps", "stepsOut"], ["guidance", "guidanceOut"]].forEach(([i, o]) => {
    const upd = () => ($(o).textContent = $(i).value);
    $(i).addEventListener("input", upd); upd();
  });

  // ------------------------------------------------------------ history + lightbox
  async function loadHistory() {
    try {
      state.history = await (await api("/api/history")).json();
      $("historyCount").textContent = state.history.length ? `${state.history.length} saved` : "";
      $("historyStrip").replaceChildren(...state.history.map((h) => {
        const b = document.createElement("button");
        b.type = "button";
        b.title = `${h.cloth_type} · seed ${h.seed}`;
        const img = new Image(); img.loading = "lazy"; img.src = h.image_url; img.alt = "Saved look";
        b.append(img);
        b.addEventListener("click", () => openLightbox(h));
        return b;
      }));
    } catch {}
  }
  function openLightbox(h) {
    state.lbItem = h;
    $("lbImg").src = h.image_url;
    $("lbDownload").href = h.image_url;
    $("lbDownload").download = `fitroom_${h.id}.jpg`;
    $("lightbox").hidden = false;
  }
  const closeLb = () => ($("lightbox").hidden = true);
  $("lbClose").addEventListener("click", closeLb);
  $("lightbox").addEventListener("click", (e) => { if (e.target.id === "lightbox") closeLb(); });
  document.addEventListener("keydown", (e) => { if (e.key === "Escape") { closeLb(); closePhone(); } });
  $("lbReuse").addEventListener("click", () => {
    const h = state.lbItem;
    setPerson({ ref: h.person_url.slice(1), url: h.person_url });
    closeLb();
    toast("Photo loaded — pick another garment.");
  });
  $("lbDelete").addEventListener("click", async () => {
    const h = state.lbItem;
    if (!confirm("Delete this look?")) return;
    try {
      await api(`/api/history/${h.day}/${h.id}`, { method: "DELETE" });
      closeLb();
      loadHistory();
    } catch (e) { toast(e.message, true); }
  });

  function blobToImage(blob) {
    return new Promise((res, rej) => {
      const img = new Image();
      img.onload = () => res(img);
      img.onerror = rej;
      img.src = URL.createObjectURL(blob);
    });
  }

  // ------------------------------------------------------------ boot
  pollStatus();
  loadGallery();
  loadHistory();
})();
