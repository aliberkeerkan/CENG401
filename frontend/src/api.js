// API layer. The UI only talks to these two functions, so switching to
// the real FastAPI backend later means replacing this file only.
//
// Planned backend contract:
//   POST /upload   multipart file           -> { image_id, width, height }
//   POST /process  { image_id, operations } -> image/png
//
// For now both calls are simulated in the browser and logged, so the
// team can see exactly which requests the frontend will send.

import { applyPipeline } from "./imageOps.js";

const MAX_SIDE = 1600; // preview size limit (risk table: large images)
const store = new Map();
let listeners = [];
let counter = 0;

export function onRequest(fn) {
  listeners.push(fn);
  return () => { listeners = listeners.filter((l) => l !== fn); };
}

function log(entry) {
  const e = { id: ++counter, time: new Date().toLocaleTimeString(), ...entry };
  listeners.forEach((l) => l(e));
}

function decode(file) {
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
    img.onerror = () => {
      URL.revokeObjectURL(url);
      reject(new Error("decode"));
    };
    img.src = url;
  });
}

export async function upload(file) {
  const t0 = performance.now();
  const isTiff = /tiff?$/i.test(file.name) || file.type === "image/tiff";
  try {
    const { data, scaled } = await decode(file);
    const id = "img_" + Math.random().toString(36).slice(2, 8);
    store.set(id, data);
    log({
      method: "POST", path: "/upload", status: 200,
      body: `multipart: ${file.name} (${(file.size / 1024).toFixed(0)} KB)`,
      result: `{ image_id: "${id}", width: ${data.width}, height: ${data.height} }`,
      ms: Math.round(performance.now() - t0),
    });
    return { imageId: id, image: data, scaled };
  } catch {
    log({
      method: "POST", path: "/upload", status: 415,
      body: `multipart: ${file.name}`,
      result: isTiff ? "TIFF needs server-side conversion to PNG" : "Unsupported image",
      ms: Math.round(performance.now() - t0),
    });
    throw new Error(
      isTiff
        ? "This browser can't display TIFF files. The backend will convert TIFF to PNG; until then, use a JPEG or PNG."
        : "This file couldn't be read as an image. Use a JPEG, PNG or TIFF file."
    );
  }
}

export function registerSample(image) {
  const id = "sample";
  store.set(id, image);
  log({
    method: "POST", path: "/upload", status: 200,
    body: "built-in sample image",
    result: `{ image_id: "sample", width: ${image.width}, height: ${image.height} }`,
    ms: 0,
  });
  return id;
}

export async function processImage(imageId, operations, { silent = false } = {}) {
  const t0 = performance.now();
  const original = store.get(imageId);
  const out = applyPipeline(original, operations);
  if (!silent) {
    log({
      method: "POST", path: "/process", status: 200,
      body: JSON.stringify({ image_id: imageId, operations }),
      result: `image/png ${out.width}×${out.height}`,
      ms: Math.round(performance.now() - t0),
    });
  }
  return out;
}
