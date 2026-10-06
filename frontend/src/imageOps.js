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

// ---------- geometry (Eqs. 4-5); applied to all four channels incl. alpha ----------

export function crop(src, { x, y, width, height }) {
  if (x < 0 || y < 0 || x + width > src.width || y + height > src.height) {
    throw new Error(`Crop area ${x},${y} ${width}x${height} is outside the ${src.width}x${src.height} image.`);
  }
  const out = new ImageData(width, height);
  for (let row = 0; row < height; row++) {
    const from = ((y + row) * src.width + x) * 4;
    out.data.set(src.data.subarray(from, from + width * 4), row * width * 4);
  }
  return out;
}

// pixel-centre alignment: x = (x' + 1/2) * M / M' - 1/2   (same as Python)
function sourceCoords(nOut, nIn) {
  const c = new Float64Array(nOut);
  for (let i = 0; i < nOut; i++) c[i] = (i + 0.5) * (nIn / nOut) - 0.5;
  return c;
}

export function resize(src, { width, height, method = "bilinear" }) {
  const w = src.width, h = src.height, s = src.data;
  const out = new ImageData(width, height), d = out.data;
  const xs = sourceCoords(width, w), ys = sourceCoords(height, h);

  if (method === "nearest") {
    const xi = Array.from(xs, (v) => Math.min(Math.max(Math.floor(v + 0.5), 0), w - 1));
    const yi = Array.from(ys, (v) => Math.min(Math.max(Math.floor(v + 0.5), 0), h - 1));
    for (let y = 0; y < height; y++)
      for (let x = 0; x < width; x++) {
        const o = (y * width + x) * 4, i = (yi[y] * w + xi[x]) * 4;
        d[o] = s[i]; d[o + 1] = s[i + 1]; d[o + 2] = s[i + 2]; d[o + 3] = s[i + 3];
      }
    return out;
  }

  for (let y = 0; y < height; y++) {
    const yy = Math.min(Math.max(ys[y], 0), h - 1);
    const y0 = Math.floor(yy), y1 = Math.min(y0 + 1, h - 1), b = yy - y0;
    for (let x = 0; x < width; x++) {
      const xx = Math.min(Math.max(xs[x], 0), w - 1);
      const x0 = Math.floor(xx), x1 = Math.min(x0 + 1, w - 1), a = xx - x0;
      const i00 = (y0 * w + x0) * 4, i10 = (y0 * w + x1) * 4;
      const i01 = (y1 * w + x0) * 4, i11 = (y1 * w + x1) * 4;
      const o = (y * width + x) * 4;
      for (let c = 0; c < 4; c++) {
        // Eq. (5), same evaluation order as the Python version
        const v = (1 - a) * (1 - b) * s[i00 + c] + a * (1 - b) * s[i10 + c]
          + (1 - a) * b * s[i01 + c] + a * b * s[i11 + c];
        d[o + c] = clamp(Math.floor(v + 0.5));
      }
    }
  }
  return out;
}

// ---------- filters (Eqs. 10 and 33); RGB only, alpha copied ----------

// border rule: reflection without repeating the edge (d c b | a b c d | c b a)
function reflectIndex(n, r) {
  const idx = new Int32Array(n + 2 * r);
  const period = 2 * (n - 1);
  for (let j = 0; j < idx.length; j++) {
    if (n === 1) { idx[j] = 0; continue; }
    let v = Math.abs(j - r) % period;
    idx[j] = v >= n ? period - v : v;
  }
  return idx;
}

export function gaussianKernel(sigma) {
  const r = Math.max(1, Math.ceil(3 * sigma));
  const k = new Float64Array(2 * r + 1);
  let sum = 0;
  for (let t = -r; t <= r; t++) {
    k[t + r] = Math.exp(-(t * t) / (2 * sigma * sigma));
  }
  for (let i = 0; i < k.length; i++) sum += k[i];
  for (let i = 0; i < k.length; i++) k[i] = k[i] / sum;
  return k;
}

export function gaussianBlur(src, { sigma }) {
  const k = gaussianKernel(sigma), r = (k.length - 1) / 2;
  const w = src.width, h = src.height, s = src.data;
  const xi = reflectIndex(w, r), yi = reflectIndex(h, r);
  const tmp = new Float64Array(w * h * 3);
  for (let y = 0; y < h; y++)
    for (let x = 0; x < w; x++)
      for (let c = 0; c < 3; c++) {
        let acc = 0;
        for (let i = 0; i < k.length; i++) acc = acc + k[i] * s[(y * w + xi[x + i]) * 4 + c];
        tmp[(y * w + x) * 3 + c] = acc;
      }
  const out = new ImageData(w, h), d = out.data;
  for (let y = 0; y < h; y++)
    for (let x = 0; x < w; x++) {
      for (let c = 0; c < 3; c++) {
        let acc = 0;
        for (let i = 0; i < k.length; i++) acc = acc + k[i] * tmp[(yi[y + i] * w + x) * 3 + c];
        d[(y * w + x) * 4 + c] = clamp(Math.floor(acc + 0.5));
      }
      d[(y * w + x) * 4 + 3] = s[(y * w + x) * 4 + 3];
    }
  return out;
}

// Median with a sliding histogram (Huang's algorithm): fast enough for live preview
export function median(src, { size }) {
  const w = src.width, h = src.height, s = src.data, r = (size - 1) / 2;
  const half = (size * size + 1) / 2;
  const xi = reflectIndex(w, r), yi = reflectIndex(h, r);
  const out = new ImageData(w, h), d = out.data;
  const hist = new Int32Array(256);
  for (let i = 3; i < s.length; i += 4) d[i] = s[i];
  for (let c = 0; c < 3; c++) {
    for (let y = 0; y < h; y++) {
      hist.fill(0);
      for (let dy = 0; dy < size; dy++) {
        const row = yi[y + dy] * w;
        for (let dx = 0; dx < size; dx++) hist[s[(row + xi[dx]) * 4 + c]]++;
      }
      let m = 0, lt = 0;
      while (lt + hist[m] < half) { lt += hist[m]; m++; }
      d[(y * w) * 4 + c] = m;
      for (let x = 1; x < w; x++) {
        const colOut = xi[x - 1], colIn = xi[x + size - 1];
        for (let dy = 0; dy < size; dy++) {
          const row = yi[y + dy] * w;
          const vOut = s[(row + colOut) * 4 + c];
          hist[vOut]--; if (vOut < m) lt--;
          const vIn = s[(row + colIn) * 4 + c];
          hist[vIn]++; if (vIn < m) lt++;
        }
        if (lt >= half) {
          do { m--; lt -= hist[m]; } while (lt >= half);
        } else {
          while (lt + hist[m] < half) { lt += hist[m]; m++; }
        }
        d[(y * w + x) * 4 + c] = m;
      }
    }
  }
  return out;
}

export const OPS = {
  crop, resize, grayscale, brightness, contrast,
  gaussian_blur: gaussianBlur, median,
};

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
