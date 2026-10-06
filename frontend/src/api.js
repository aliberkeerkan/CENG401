// API layer between the UI and the FastAPI backend.
//
// Backend contract (see backend/app/main.py, docs at <API_URL>/docs):
//   GET  /health                          -> { status, operations }
//   POST /upload   multipart "file"       -> { image_id, width, height, scaled, source_format }
//   GET  /images/{id}/original            -> image/png
//   POST /process  { image_id, operations }-> image/png
//   POST /export   { image_id, operations, format, quality } -> file
//
// On start the client checks /health. If the backend is not running, it
// falls back to simulating the same requests in the browser, so the UI
// can still be developed without the server.

import { applyPipeline } from "./imageOps.js";

export const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";
const MAX_SIDE = 1600;

let mode = "checking"; // "checking" | "server" | "mock"
let listeners = [];
let statusListeners = [];
let counter = 0;
const mockStore = new Map();

export function onRequest(fn) {
  listeners.push(fn);
  return () => { listeners = listeners.filter((l) => l !== fn); };
}

export function onStatus(fn) {
  statusListeners.push(fn);
  fn(mode);
  return () => { statusListeners = statusListeners.filter((l) => l !== fn); };
}

function setMode(m) {
  mode = m;
  statusListeners.forEach((l) => l(m));
}

function log(entry) {
  const e = { id: ++counter, time: new Date().toLocaleTimeString(), simulated: mode !== "server", ...entry };
  listeners.forEach((l) => l(e));
}

export async function checkBackend() {
  try {
    const ctrl = new AbortController();
    const timer = setTimeout(() => ctrl.abort(), 1500);
    const r = await fetch(`${API_URL}/health`, { signal: ctrl.signal });
    clearTimeout(timer);
    setMode(r.ok ? "server" : "mock");
  } catch {
    setMode("mock");
  }
  return mode;
}

// ---------- helpers ----------

function imageDataFromBitmap(bitmap) {
  const c = document.createElement("canvas");
  c.width = bitmap.width;
  c.height = bitmap.height;
  const g = c.getContext("2d");
  g.drawImage(bitmap, 0, 0);
  return g.getImageData(0, 0, c.width, c.height);
}

async function blobToImageData(blob) {
  const bitmap = await createImageBitmap(blob, { colorSpaceConversion: "none", premultiplyAlpha: "none" });
  return imageDataFromBitmap(bitmap);
}

function imageDataToBlob(image) {
  const c = document.createElement("canvas");
  c.width = image.width;
  c.height = image.height;
  c.getContext("2d").putImageData(image, 0, 0);
  return new Promise((resolve) => c.toBlob(resolve, "image/png"));
}

async function errorMessage(r) {
  try {
    const body = await r.json();
    if (typeof body.detail === "string") return body.detail;
    if (Array.isArray(body.detail)) return body.detail.map((d) => d.msg).join("; ");
  } catch { /* not JSON */ }
  return `Request failed with status ${r.status}.`;
}

function unreachable() {
  setMode("mock");
  return new Error("The backend stopped responding. Start it again, then reload the image.");
}

// ---------- server mode ----------

async function serverUpload(file, name) {
  const t0 = performance.now();
  const form = new FormData();
  form.append("file", file, name);
  let r;
  try {
    r = await fetch(`${API_URL}/upload`, { method: "POST", body: form });
  } catch {
    throw unreachable();
  }
  if (!r.ok) {
    const msg = await errorMessage(r);
    log({ method: "POST", path: "/upload", status: r.status, body: `multipart: ${name}`, result: msg, ms: Math.round(performance.now() - t0) });
    throw new Error(msg);
  }
  const info = await r.json();
  log({
    method: "POST", path: "/upload", status: r.status,
    body: `multipart: ${name} (${(file.size / 1024).toFixed(0)} KB)`,
    result: JSON.stringify(info),
    ms: Math.round(performance.now() - t0),
  });

  const t1 = performance.now();
  const png = await fetch(`${API_URL}/images/${info.image_id}/original`);
  const image = await blobToImageData(await png.blob());
  log({
    method: "GET", path: `/images/${info.image_id}/original`, status: png.status,
    body: "", result: `image/png ${image.width}×${image.height}`,
    ms: Math.round(performance.now() - t1),
  });
  return { imageId: info.image_id, image, scaled: info.scaled };
}

async function serverProcess(imageId, operations) {
  const t0 = performance.now();
  const payload = { image_id: imageId, operations };
  let r;
  try {
    r = await fetch(`${API_URL}/process`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
  } catch {
    throw unreachable();
  }
  if (!r.ok) {
    const msg = await errorMessage(r);
    log({ method: "POST", path: "/process", status: r.status, body: JSON.stringify(payload), result: msg, ms: Math.round(performance.now() - t0) });
    throw new Error(msg);
  }
  const image = await blobToImageData(await r.blob());
  log({
    method: "POST", path: "/process", status: r.status,
    body: JSON.stringify(payload), result: `image/png ${image.width}×${image.height}`,
    ms: Math.round(performance.now() - t0),
  });
  return image;
}

// ---------- mock mode (no backend running) ----------

function decodeInBrowser(file) {
  return new Promise((resolve, reject) => {
    const url = URL.createObjectURL(file);
    const img = new Image();
    img.onload = () => {
      let { naturalWidth: w, naturalHeight: h } = img;
      const scale = Math.min(1, MAX_SIDE / Math.max(w, h));
      w = Math.round(w * scale); h = Math.round(h * scale);
      const c = document.createElement("canvas");
      c.width = w; c.height = h;
      const g = c.getContext("2d");
      g.drawImage(img, 0, 0, w, h);
      URL.revokeObjectURL(url);
      resolve({ data: g.getImageData(0, 0, w, h), scaled: scale < 1 });
    };
    img.onerror = () => { URL.revokeObjectURL(url); reject(new Error("decode")); };
    img.src = url;
  });
}

async function mockUpload(file, name) {
  const t0 = performance.now();
  const isTiff = /tiff?$/i.test(name) || file.type === "image/tiff";
  try {
    const { data, scaled } = await decodeInBrowser(file);
    const id = "img_" + Math.random().toString(36).slice(2, 8);
    mockStore.set(id, data);
    log({
      method: "POST", path: "/upload", status: 200,
      body: `multipart: ${name} (${(file.size / 1024).toFixed(0)} KB)`,
      result: `{ image_id: "${id}", width: ${data.width}, height: ${data.height} }`,
      ms: Math.round(performance.now() - t0),
    });
    return { imageId: id, image: data, scaled };
  } catch {
    log({ method: "POST", path: "/upload", status: 415, body: `multipart: ${name}`, result: "Unsupported image", ms: 0 });
    throw new Error(
      isTiff
        ? "This browser can't display TIFF files. Start the backend, which converts TIFF to PNG, or use a JPEG or PNG."
        : "This file couldn't be read as an image. Use a JPEG, PNG or TIFF file."
    );
  }
}

async function mockProcess(imageId, operations) {
  const t0 = performance.now();
  const out = applyPipeline(mockStore.get(imageId), operations);
  log({
    method: "POST", path: "/process", status: 200,
    body: JSON.stringify({ image_id: imageId, operations }),
    result: `image/png ${out.width}×${out.height}`,
    ms: Math.round(performance.now() - t0),
  });
  return out;
}

// ---------- public API used by the UI ----------

export function upload(file) {
  return mode === "server" ? serverUpload(file, file.name) : mockUpload(file, file.name);
}

export async function uploadSample(image) {
  const blob = await imageDataToBlob(image);
  const file = new File([blob], "sample.png", { type: "image/png" });
  return mode === "server" ? serverUpload(file, file.name) : mockUpload(file, file.name);
}

export function processImage(imageId, operations) {
  return mode === "server" ? serverProcess(imageId, operations) : mockProcess(imageId, operations);
}

// ---------- export ----------

export const EXPORT_FORMATS = {
  png: { label: "PNG", ext: "png", mime: "image/png" },
  jpeg: { label: "JPEG", ext: "jpg", mime: "image/jpeg" },
  tiff: { label: "TIFF", ext: "tif", mime: "image/tiff" },
};

async function serverExport(imageId, operations, format) {
  const t0 = performance.now();
  const payload = { image_id: imageId, operations, format, quality: 92 };
  let r;
  try {
    r = await fetch(`${API_URL}/export`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
  } catch {
    throw unreachable();
  }
  if (!r.ok) {
    const msg = await errorMessage(r);
    log({ method: "POST", path: "/export", status: r.status, body: JSON.stringify(payload), result: msg, ms: Math.round(performance.now() - t0) });
    throw new Error(msg);
  }
  const blob = await r.blob();
  log({
    method: "POST", path: "/export", status: r.status,
    body: JSON.stringify(payload),
    result: `${blob.type} ${(blob.size / 1024).toFixed(0)} KB`,
    ms: Math.round(performance.now() - t0),
  });
  return blob;
}

async function mockExport(imageId, operations, format) {
  if (format === "tiff") {
    throw new Error("TIFF export needs the backend. Start it, or export as PNG or JPEG.");
  }
  const t0 = performance.now();
  const out = applyPipeline(mockStore.get(imageId), operations);
  const c = document.createElement("canvas");
  c.width = out.width;
  c.height = out.height;
  const g = c.getContext("2d");
  if (format === "jpeg") {  // JPEG has no transparency: flatten onto white
    g.fillStyle = "#ffffff";
    g.fillRect(0, 0, c.width, c.height);
    const tmp = document.createElement("canvas");
    tmp.width = out.width; tmp.height = out.height;
    tmp.getContext("2d").putImageData(out, 0, 0);
    g.drawImage(tmp, 0, 0);
  } else {
    g.putImageData(out, 0, 0);
  }
  const blob = await new Promise((res) => c.toBlob(res, EXPORT_FORMATS[format].mime, 0.92));
  log({
    method: "POST", path: "/export", status: 200,
    body: JSON.stringify({ image_id: imageId, operations, format }),
    result: `${blob.type} ${(blob.size / 1024).toFixed(0)} KB`,
    ms: Math.round(performance.now() - t0),
  });
  return blob;
}

export function exportImage(imageId, operations, format = "png") {
  return mode === "server"
    ? serverExport(imageId, operations, format)
    : mockExport(imageId, operations, format);
}

export function saveBlob(blob, filename) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
