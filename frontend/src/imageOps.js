// Pixel operations used by the in-browser mock backend.
// Each function follows the equation numbers in the project plan (Section 5).
// When the FastAPI backend is ready, the same operations run in Python
// and this file is only used for instant slider previews.

const clamp = (v) => (v < 0 ? 0 : v > 255 ? 255 : v);

// Eq. (6): Y = 0.299R + 0.587G + 0.114B
export function grayscale(src) {
  const out = new ImageData(src.width, src.height);
  const s = src.data, d = out.data;
  for (let i = 0; i < s.length; i += 4) {
    const y = Math.round(0.299 * s[i] + 0.587 * s[i + 1] + 0.114 * s[i + 2]);
    d[i] = d[i + 1] = d[i + 2] = y;
    d[i + 3] = s[i + 3];
  }
  return out;
}

// Eq. (8): g = clip(f + beta)
export function brightness(src, { beta }) {
  const out = new ImageData(src.width, src.height);
  const s = src.data, d = out.data;
  for (let i = 0; i < s.length; i += 4) {
    d[i] = clamp(s[i] + beta);
    d[i + 1] = clamp(s[i + 1] + beta);
    d[i + 2] = clamp(s[i + 2] + beta);
    d[i + 3] = s[i + 3];
  }
  return out;
}

// Eq. (9): g = clip(alpha (f - c) + c), pivot c = image mean intensity
export function contrast(src, { alpha }) {
  const c = stats(src).mean;
  const out = new ImageData(src.width, src.height);
  const s = src.data, d = out.data;
  for (let i = 0; i < s.length; i += 4) {
    d[i] = clamp(Math.round(alpha * (s[i] - c) + c));
    d[i + 1] = clamp(Math.round(alpha * (s[i + 1] - c) + c));
    d[i + 2] = clamp(Math.round(alpha * (s[i + 2] - c) + c));
    d[i + 3] = s[i + 3];
  }
  return out;
}

export const OPS = { grayscale, brightness, contrast };

export function applyOp(img, op) {
  return OPS[op.type](img, op.params || {});
}

export function applyPipeline(original, ops) {
  return ops.reduce((img, op) => applyOp(img, op), original);
}

// Eqs. (12)-(14): normalized histogram of luma, mean, median, std
export function stats(img) {
  const hist = new Array(256).fill(0);
  const s = img.data;
  let n = 0;
  for (let i = 0; i < s.length; i += 4) {
    const y = Math.round(0.299 * s[i] + 0.587 * s[i + 1] + 0.114 * s[i + 2]);
    hist[y]++;
    n++;
  }
  let mean = 0;
  for (let k = 0; k < 256; k++) mean += k * (hist[k] / n);
  let v = 0;
  for (let k = 0; k < 256; k++) v += (k - mean) ** 2 * (hist[k] / n);
  let cum = 0, median = 0;
  for (let k = 0; k < 256; k++) {
    cum += hist[k] / n;
    if (cum >= 0.5) { median = k; break; }
  }
  return { hist, mean, median, std: Math.sqrt(v), n };
}

// Synthetic test image so the prototype works without an upload
export function sampleImage(w = 640, h = 420) {
  const c = document.createElement("canvas");
  c.width = w; c.height = h;
  const g = c.getContext("2d");
  const grad = g.createLinearGradient(0, 0, w, h);
  grad.addColorStop(0, "#1d3b53");
  grad.addColorStop(0.55, "#4f8a8b");
  grad.addColorStop(1, "#f2c14e");
  g.fillStyle = grad;
  g.fillRect(0, 0, w, h);
  // stepped gray wedge: useful for seeing brightness/contrast clipping
  for (let i = 0; i < 8; i++) {
    const v = Math.round((i / 7) * 255);
    g.fillStyle = `rgb(${v},${v},${v})`;
    g.fillRect(40 + i * 50, h - 90, 50, 50);
  }
  g.fillStyle = "#d1495b";
  g.beginPath(); g.arc(470, 150, 80, 0, Math.PI * 2); g.fill();
  g.fillStyle = "#edae49";
  g.fillRect(90, 70, 150, 150);
  g.strokeStyle = "#ffffff";
  g.lineWidth = 3;
  for (let x = 300; x < 400; x += 12) {
    g.beginPath(); g.moveTo(x, 60); g.lineTo(x, 230); g.stroke();
  }
  return g.getImageData(0, 0, w, h);
}
