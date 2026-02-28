// Smart Checkout Web Client (Local)
// - webcam -> canvas 640x640 -> jpeg blob -> [8 bytes frame_id] + jpeg bytes -> websocket
// - gating: only send new frame when server responds
// - tracking local por IoU para "descartar hasta que desaparezca"
// - carrito simple con qty, subtotal y total
//
// Fix aplicado:
// 1) Throttle del render de detecciones (evita re-render del DOM mientras haces click)
// 2) Badge en el botón "Carrito (n)" para confirmar que sí se agregó aunque el drawer esté cerrado
//
// Mejora UX:
// 3) ✅ Toast de confirmación al agregar/descartar/editar carrito (feedback inmediato sin abrir el drawer)

const el = (id) => document.getElementById(id);

const video = el("video");
const overlay = el("overlay");
const octx = overlay.getContext("2d");

const wsUrlInput = el("wsUrl");
const btnConnect = el("btnConnect");
const btnDisconnect = el("btnDisconnect");
const btnCart = el("btnCart");
const statusEl = el("status");
const metricsEl = el("metrics");
const detectionsList = el("detectionsList");
const queueInfo = el("queueInfo");

const cartDrawer = el("cartDrawer");
const btnCloseCart = el("btnCloseCart");
const btnClearCart = el("btnClearCart");
const cartItemsEl = el("cartItems");
const cartTotalEl = el("cartTotal");
const cartCountEl = el("cartCount");

// ✅ Toast container
const toastsEl = el("toasts");

let ws = null;
let stream = null;

const CAP_SIZE = 640;               // canvas capture size
const JPEG_QUALITY = 0.8;
const DISCARD_DISAPPEAR_FRAMES = 15; // si un objeto no se ve por N respuestas -> se "libera"
const TRACK_IOU_THRESHOLD = 0.30;
const MAX_UI_PENDING = 6;           // para no saturar

// ✅ Throttle UI render para evitar que el DOM se regenere mientras haces click
const DETECTIONS_RENDER_MS = 200; // 5 renders/seg aprox
let lastDetectionsRenderAt = 0;
let renderScheduled = false;

let waitingResponse = false;
let lastResponseAt = 0;
let fpsCounter = { t0: performance.now(), n: 0, fps: 0 };

const captureCanvas = document.createElement("canvas");
captureCanvas.width = CAP_SIZE;
captureCanvas.height = CAP_SIZE;
const cctx = captureCanvas.getContext("2d", { willReadFrequently: false });

// ------------------------
// Estado: tracking + UI
// ------------------------
let nextTrackId = 1;

/**
 * Track:
 * {
 *   id, class_name, class_id,
 *   bbox: {x,y,w,h} px (en canvas CAP_SIZE),
 *   conf, product_info,
 *   lastSeenTick, missing,
 *   status: "pending" | "added" | "discarded"
 * }
 */
let tracks = new Map(); // id -> track
let tick = 0;           // incrementa por cada respuesta del servidor

// Carrito: sku -> { sku, name, price, currency, description, qty }
let cart = new Map();

// ------------------------
// Utilidades
// ------------------------
function setStatus(kind, text) {
  statusEl.textContent = text;
  if (kind === "ok") statusEl.style.color = "var(--ok)";
  else if (kind === "warn") statusEl.style.color = "var(--warn)";
  else if (kind === "bad") statusEl.style.color = "var(--bad)";
  else statusEl.style.color = "var(--muted)";
}

function formatMoney(value, currency = "USD") {
  const v = Number(value || 0);
  try {
    return new Intl.NumberFormat(undefined, { style: "currency", currency }).format(v);
  } catch {
    return `$${v.toFixed(2)}`;
  }
}

function clamp(v, a, b) {
  return Math.max(a, Math.min(b, v));
}

function normToPx(bbox_norm) {
  // bbox_norm viene como [x_center, y_center, w, h] normalizados [0..1] en el frame de inferencia
  const [cx, cy, w, h] = bbox_norm;
  const pw = w * CAP_SIZE;
  const ph = h * CAP_SIZE;
  const px = (cx * CAP_SIZE) - pw / 2;
  const py = (cy * CAP_SIZE) - ph / 2;
  return {
    x: clamp(px, 0, CAP_SIZE),
    y: clamp(py, 0, CAP_SIZE),
    w: clamp(pw, 0, CAP_SIZE),
    h: clamp(ph, 0, CAP_SIZE)
  };
}

function iou(a, b) {
  const x1 = Math.max(a.x, b.x);
  const y1 = Math.max(a.y, b.y);
  const x2 = Math.min(a.x + a.w, b.x + b.w);
  const y2 = Math.min(a.y + a.h, b.y + b.h);
  const interW = Math.max(0, x2 - x1);
  const interH = Math.max(0, y2 - y1);
  const inter = interW * interH;
  const union = a.w * a.h + b.w * b.h - inter;
  return union <= 0 ? 0 : inter / union;
}

function buildMessage(frameIdBigInt, jpegUint8) {
  // Header: uint64 little-endian
  const header = new ArrayBuffer(8);
  const dv = new DataView(header);
  dv.setBigUint64(0, frameIdBigInt, true);

  const out = new Uint8Array(8 + jpegUint8.byteLength);
  out.set(new Uint8Array(header), 0);
  out.set(jpegUint8, 8);
  return out.buffer;
}

function blobToUint8(blob) {
  return new Promise((resolve, reject) => {
    const fr = new FileReader();
    fr.onerror = () => reject(fr.error);
    fr.onload = () => resolve(new Uint8Array(fr.result));
    fr.readAsArrayBuffer(blob);
  });
}

// ------------------------
// ✅ Toasts
// ------------------------
function toast(kind, title, desc = "", amount = "", ttlMs = 1600) {
  if (!toastsEl) return;

  const node = document.createElement("div");
  node.className = `toast ${kind || ""}`.trim();

  node.innerHTML = `
    <div class="row">
      <div class="left">
        <div class="title">${escapeHtml(title)}</div>
        ${desc ? `<div class="desc">${escapeHtml(desc)}</div>` : ``}
      </div>
      ${amount ? `<div class="amount">${escapeHtml(amount)}</div>` : ``}
    </div>
  `;

  toastsEl.appendChild(node);

  // auto remove
  setTimeout(() => {
    node.classList.add("out");
    setTimeout(() => node.remove(), 180);
  }, ttlMs);
}

function escapeHtml(s) {
  return String(s ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

// ------------------------
// Render scheduling (throttle)
// ------------------------
function scheduleDetectionsRender(force = false) {
  const now = performance.now();

  if (!force && (now - lastDetectionsRenderAt) < DETECTIONS_RENDER_MS) {
    if (!renderScheduled) {
      renderScheduled = true;
      setTimeout(() => {
        renderScheduled = false;
        lastDetectionsRenderAt = performance.now();
        renderDetectionsPanel();
      }, DETECTIONS_RENDER_MS);
    }
    return;
  }

  lastDetectionsRenderAt = now;
  renderDetectionsPanel();
}

function updateCartBadge(total, count) {
  btnCart.textContent = count > 0 ? `Carrito (${count})` : "Carrito";
  btnCart.title = count > 0 ? `Total: ${formatMoney(total, "USD")}` : "";
}

// ------------------------
// Tracking: match detections -> tracks
// ------------------------
function updateTracksFromDetections(detections) {
  tick += 1;

  for (const t of tracks.values()) t.missing += 1;

  for (const det of detections) {
    const bboxPx = normToPx(det.bbox_norm);
    const clsName = det.class_name;

    let bestId = null;
    let bestIou = 0;

    for (const [id, tr] of tracks.entries()) {
      if (tr.class_name !== clsName) continue;
      const score = iou(tr.bbox, bboxPx);
      if (score > bestIou) {
        bestIou = score;
        bestId = id;
      }
    }

    if (bestId !== null && bestIou >= TRACK_IOU_THRESHOLD) {
      const tr = tracks.get(bestId);
      tr.bbox = bboxPx;
      tr.conf = det.confidence;
      tr.product_info = det.product_info || tr.product_info;
      tr.lastSeenTick = tick;
      tr.missing = 0;
      tr.class_id = det.class_id;
    } else {
      const id = nextTrackId++;
      tracks.set(id, {
        id,
        class_id: det.class_id,
        class_name: clsName,
        bbox: bboxPx,
        conf: det.confidence,
        product_info: det.product_info || null,
        lastSeenTick: tick,
        missing: 0,
        status: "pending"
      });
    }
  }

  for (const [id, tr] of tracks.entries()) {
    if (tr.missing >= DISCARD_DISAPPEAR_FRAMES) tracks.delete(id);
  }
}

// ------------------------
// UI: detections queue + cart
// ------------------------
function renderDetectionsPanel() {
  const pending = [...tracks.values()].filter(t => t.status === "pending");

  queueInfo.textContent = `${pending.length} pendientes`;
  detectionsList.innerHTML = "";

  pending.sort((a, b) => (b.conf ?? 0) - (a.conf ?? 0));

  const show = pending.slice(0, MAX_UI_PENDING);
  for (const t of show) {
    const info = t.product_info || {
      sku: "UNK-000",
      name: `Unknown (${t.class_name})`,
      price: 0,
      currency: "USD",
      description: "Producto no encontrado en catálogo."
    };

    const card = document.createElement("div");
    card.className = "card";

    card.innerHTML = `
      <div class="cardTop">
        <div>
          <div class="cardTitle">${info.name}</div>
          <div class="small">SKU: ${info.sku} · Clase: ${t.class_name}</div>
        </div>
        <div class="badge">${formatMoney(info.price, info.currency || "USD")}</div>
      </div>
      <div class="cardBody">
        <div>${info.description || ""}</div>
        <div class="small">conf: ${(t.conf ?? 0).toFixed(2)} · track_id: ${t.id}</div>
      </div>
      <div class="cardActions">
        <button class="add">Agregar</button>
        <button class="secondary discard">Descartar</button>
      </div>
    `;

    card.querySelector("button.add").onclick = () => {
      addToCart(info);
      t.status = "added";
      renderCart();

      // ✅ feedback inmediato sin abrir carrito
      toast("ok", "Agregado al carrito", `${info.name} · ${info.sku}`, `+${formatMoney(info.price, info.currency || "USD")}`);

      scheduleDetectionsRender(true);
    };

    card.querySelector("button.discard").onclick = () => {
      t.status = "discarded";

      // ✅ feedback inmediato
      toast("warn", "Descartado", `${info.name} · track #${t.id}`, "");

      scheduleDetectionsRender(true);
    };

    detectionsList.appendChild(card);
  }

  if (pending.length > show.length) {
    const more = document.createElement("div");
    more.className = "small";
    more.textContent = `… y ${pending.length - show.length} más (se ocultan para no saturar).`;
    detectionsList.appendChild(more);
  }
}

function addToCart(info) {
  const sku = info.sku || "UNK-000";
  const current = cart.get(sku);
  if (current) current.qty += 1;
  else {
    cart.set(sku, {
      sku,
      name: info.name,
      price: Number(info.price || 0),
      currency: info.currency || "USD",
      description: info.description || "",
      qty: 1
    });
  }
}

function renderCart() {
  cartItemsEl.innerHTML = "";

  let total = 0;
  let count = 0;

  for (const item of cart.values()) {
    const subtotal = item.price * item.qty;
    total += subtotal;
    count += item.qty;

    const row = document.createElement("div");
    row.className = "cartRow";

    row.innerHTML = `
      <div>
        <strong>${item.name}</strong>
        <div class="muted">${item.description}</div>
        <div class="muted">SKU: ${item.sku} · ${formatMoney(item.price, item.currency)} c/u</div>
      </div>
      <div class="qty">
        <button class="secondary minus">-</button>
        <div><strong>${item.qty}</strong></div>
        <button class="secondary plus">+</button>
      </div>
    `;

    row.querySelector(".minus").onclick = () => {
      item.qty -= 1;
      if (item.qty <= 0) {
        cart.delete(item.sku);
        toast("warn", "Eliminado del carrito", `${item.name} · ${item.sku}`, "");
      } else {
        toast("warn", "Cantidad actualizada", `${item.name}`, `−1`);
      }
      renderCart();
    };

    row.querySelector(".plus").onclick = () => {
      item.qty += 1;
      toast("ok", "Cantidad actualizada", `${item.name}`, `+1`);
      renderCart();
    };

    cartItemsEl.appendChild(row);
  }

  cartTotalEl.textContent = formatMoney(total, "USD");
  cartCountEl.textContent = `${count} items`;

  updateCartBadge(total, count);
}

function clearCart() {
  if (cart.size > 0) toast("bad", "Carrito vaciado", "Se eliminaron todos los productos.", "");
  cart.clear();
  renderCart();
}

// ------------------------
// Overlay drawing
// ------------------------
function drawOverlay() {
  const rect = video.getBoundingClientRect();
  overlay.width = Math.floor(rect.width);
  overlay.height = Math.floor(rect.height);

  octx.clearRect(0, 0, overlay.width, overlay.height);

  const sx = overlay.width / CAP_SIZE;
  const sy = overlay.height / CAP_SIZE;

  for (const t of tracks.values()) {
    if (t.missing > 0) continue;

    const { x, y, w, h } = t.bbox;

    let stroke = "rgba(78,161,255,0.9)";
    if (t.status === "added") stroke = "rgba(34,197,94,0.9)";
    if (t.status === "discarded") stroke = "rgba(245,158,11,0.9)";

    octx.strokeStyle = stroke;
    octx.lineWidth = 2;
    octx.strokeRect(x * sx, y * sy, w * sx, h * sy);

    const label = `${t.class_name} (${(t.conf ?? 0).toFixed(2)}) #${t.id}`;
    octx.font = "12px ui-sans-serif, system-ui, sans-serif";
    const pad = 4;
    const tw = octx.measureText(label).width;
    octx.fillStyle = "rgba(0,0,0,0.55)";
    octx.fillRect(x * sx, (y * sy) - 18, tw + pad * 2, 16);
    octx.fillStyle = "rgba(255,255,255,0.92)";
    octx.fillText(label, (x * sx) + pad, (y * sy) - 6);
  }
}

// ------------------------
// Capture & send loop (gated)
// ------------------------
async function sendFrameIfReady() {
  if (!ws || ws.readyState !== WebSocket.OPEN) return;
  if (waitingResponse) return;
  if (!video.videoWidth || !video.videoHeight) return;

  waitingResponse = true;

  const vw = video.videoWidth;
  const vh = video.videoHeight;

  const side = Math.min(vw, vh);
  const sx = Math.floor((vw - side) / 2);
  const sy = Math.floor((vh - side) / 2);

  cctx.drawImage(video, sx, sy, side, side, 0, 0, CAP_SIZE, CAP_SIZE);

  const blob = await new Promise((resolve) =>
    captureCanvas.toBlob(resolve, "image/jpeg", JPEG_QUALITY)
  );

  if (!blob) {
    waitingResponse = false;
    return;
  }

  const jpegUint8 = await blobToUint8(blob);

  const frameId = BigInt(Date.now());
  const msg = buildMessage(frameId, jpegUint8);

  try {
    ws.send(msg);
  } catch (e) {
    waitingResponse = false;
    setStatus("bad", "Error enviando frame");
  }
}

function animationLoop() {
  sendFrameIfReady().catch(() => {});
  drawOverlay();
  requestAnimationFrame(animationLoop);
}

// ------------------------
// WebSocket handlers
// ------------------------
function connectWS() {
  const url = wsUrlInput.value.trim();
  ws = new WebSocket(url);
  ws.binaryType = "arraybuffer";

  setStatus("warn", "Conectando…");
  btnConnect.disabled = true;

  ws.onopen = () => {
    setStatus("ok", "Conectado");
    btnDisconnect.disabled = false;
    lastResponseAt = performance.now();
    toast("ok", "Conectado", "WebSocket listo para inferencias.", "");
  };

  ws.onmessage = (ev) => {
    try {
      const data = JSON.parse(ev.data);

      if (data.error) {
        setStatus("warn", `Servidor: ${data.error}`);
        waitingResponse = false;
        toast("bad", "Error del servidor", String(data.error), "");
        return;
      }

      setStatus("ok", "Conectado");
      lastResponseAt = performance.now();

      const dets = Array.isArray(data.detections) ? data.detections : [];
      updateTracksFromDetections(dets);

      scheduleDetectionsRender(false);

      fpsCounter.n += 1;
      const now = performance.now();
      if (now - fpsCounter.t0 >= 1000) {
        fpsCounter.fps = fpsCounter.n;
        fpsCounter.n = 0;
        fpsCounter.t0 = now;
      }

      const lat = (typeof data.inference_time === "number") ? data.inference_time : null;
      const items = dets.length;

      metricsEl.innerHTML = `
        <span>fps: ${fpsCounter.fps}</span>
        <span>lat: ${lat !== null ? `${lat.toFixed(1)} ms` : "-"}</span>
        <span>items: ${items}</span>
      `;
    } catch {
      setStatus("warn", "Respuesta inválida");
      toast("bad", "Respuesta inválida", "No se pudo parsear JSON.", "");
    } finally {
      waitingResponse = false;
    }
  };

  ws.onclose = () => {
    setStatus("bad", "Desconectado");
    btnConnect.disabled = false;
    btnDisconnect.disabled = true;
    waitingResponse = false;
    ws = null;
    toast("warn", "Desconectado", "WebSocket cerrado.", "");
  };

  ws.onerror = () => {
    setStatus("bad", "Error WebSocket");
    toast("bad", "Error WebSocket", "Revisa URL / backend.", "");
  };
}

function disconnectWS() {
  if (ws) ws.close();
}

// ------------------------
// Webcam
// ------------------------
async function startCamera() {
  stream = await navigator.mediaDevices.getUserMedia({
    video: { width: { ideal: 1280 }, height: { ideal: 720 } },
    audio: false
  });
  video.srcObject = stream;
  await video.play();
  toast("ok", "Cámara lista", "Capturando video local.", "");
}

function stopCamera() {
  if (!stream) return;
  for (const tr of stream.getTracks()) tr.stop();
  stream = null;
}

// ------------------------
// UI events
// ------------------------
btnConnect.onclick = () => connectWS();
btnDisconnect.onclick = () => disconnectWS();

btnCart.onclick = () => {
  cartDrawer.classList.remove("hidden");
  renderCart();
};
btnCloseCart.onclick = () => cartDrawer.classList.add("hidden");

btnClearCart.onclick = () => clearCart();

window.addEventListener("beforeunload", () => {
  try { disconnectWS(); } catch {}
  try { stopCamera(); } catch {}
});

// ------------------------
// Boot
// ------------------------
(async function boot() {
  setStatus("muted", "Desconectado");
  updateCartBadge(0, 0);
  renderCart();
  await startCamera();
  requestAnimationFrame(animationLoop);
})();
