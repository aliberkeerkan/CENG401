import React, { useEffect, useMemo, useRef, useState } from "react";
import katex from "katex";
import "katex/dist/katex.min.css";
import {
  upload, uploadFromUrl, processImage, exportImage, saveBlob, EXPORT_FORMATS,
  onRequest, onStatus, checkBackend, API_URL,
} from "./api.js";
import { BUILTIN_SAMPLES } from "./builtinSamples.js";
import { applyOp, stats, gaussianKernel } from "./imageOps.js";

// Tool definitions. `heavy` tools get a short delay before the preview is
// computed so dragging a slider stays smooth.
const TOOLS = {
  crop: {
    label: "Crop", eq: "4",
    describe: "Keeps a rectangular region without resampling. Drag on the processed image or type the values.",
  },
  resize: {
    label: "Resize", eq: "5", heavy: true,
    describe: "Resamples the image to a new size. Bilinear blends the four nearest pixels; nearest copies one.",
  },
  grayscale: {
    label: "Grayscale", eq: "6",
    describe: "Maps the three colour channels to one luma channel. Not reversible.",
  },
  brightness: {
    label: "Brightness", eq: "8",
    slider: { key: "beta", min: -100, max: 100, step: 1, name: "β (offset)" },
    describe: "Adds the same offset to every pixel. Values past 0 or 255 are clipped.",
  },
  contrast: {
    label: "Contrast", eq: "9",
    describe: "Gonzalez–Woods contrast stretch. Raise Low or lower High to compress that end; Low < 0 or High > 255 expands it and reduces contrast there.",
  },
  gaussian_blur: {
    label: "Gaussian blur", eq: "10", heavy: true,
    slider: { key: "sigma", min: 0.5, max: 10, step: 0.1, name: "σ (blur scale)" },
    describe: "Averages each pixel with its neighbours using Gaussian weights. Larger σ blurs more.",
  },
  median: {
    label: "Median filter", eq: "33", heavy: true,
    slider: { key: "size", min: 3, max: 15, step: 2, name: "Window size" },
    describe: "Replaces each pixel with the median of its window. Removes salt-and-pepper noise and keeps edges.",
  },
  salt_pepper: {
    label: "Salt & pepper", eq: "SP", heavy: true,
    describe: "Gonzalez–Woods impulse noise. Each pixel becomes black with probability P_a or white with probability P_b.",
  },
  quantization: {
    label: "Quantization", eq: "Q",
    slider: { key: "bits", min: 1, max: 8, step: 1, name: "Bits b" },
    describe: "Uniform quantizer. Fewer bits make a larger step Δ, so the error e ~ U(−Δ/2, Δ/2) is stronger. 8 bits leaves the image unchanged.",
  },
};

function defaultParams(tool, img) {
  const w = img ? img.width : 1, h = img ? img.height : 1;
  switch (tool) {
    case "crop":
      return { x: Math.round(w * 0.1), y: Math.round(h * 0.1), width: Math.max(1, Math.round(w * 0.8)), height: Math.max(1, Math.round(h * 0.8)) };
    case "resize": return { width: w, height: h, method: "bilinear", lock: true };
    case "brightness": return { beta: 0 };
    case "contrast": return { low: 0, high: 255 };
    case "gaussian_blur": return { sigma: 1.5 };
    case "median": return { size: 3 };
    case "salt_pepper": return { pepper: 0.03, salt: 0.03 };
    case "quantization": return { bits: 3 };
    default: return {};
  }
}

function isNeutral(tool, p, img) {
  if (!img) return true;
  if (tool === "brightness") return p.beta === 0;
  if (tool === "contrast") return p.low === 0 && p.high === 255;
  if (tool === "resize") return p.width === img.width && p.height === img.height;
  if (tool === "crop") return p.x === 0 && p.y === 0 && p.width === img.width && p.height === img.height;
  if (tool === "salt_pepper") return p.pepper === 0 && p.salt === 0;
  if (tool === "quantization") return p.bits === 8;
  return false;
}

// parameters sent to the backend (UI-only fields removed)
function opParams(tool, p) {
  if (tool === "resize") return { width: p.width, height: p.height, method: p.method };
  return { ...p };
}

const fmt = (v, d = 2) => Number(v).toFixed(d);
const clampInt = (v, lo, hi) => Math.min(Math.max(Math.round(Number(v) || 0), lo), hi);

function MathTex({ tex, display = false }) {
  const html = useMemo(
    () => katex.renderToString(tex, { throwOnError: false, displayMode: display, strict: "ignore" }),
    [tex, display],
  );
  return <span className="math" dangerouslySetInnerHTML={{ __html: html }} />;
}

function Formula({ tool, params, pivot, img }) {
  const W = img ? img.width : 0, H = img ? img.height : 0;
  let tex = null;
  switch (tool) {
    case "crop":
      tex = `g(x,y)=f(x+${params.x},\\,y+${params.y}),\\quad 0\\le x<${params.width},\\; 0\\le y<${params.height}`;
      break;
    case "resize":
      tex = params.method === "nearest"
        ? `\\begin{aligned} g(x',y')&=f(\\mathrm{round}(x),\\mathrm{round}(y))\\\\ x&=(x'+\\tfrac12)\\cdot\\frac{${W}}{${params.width}}-\\tfrac12 \\end{aligned}`
        : `\\begin{aligned} x&=(x'+\\tfrac12)\\cdot\\frac{${W}}{${params.width}}-\\tfrac12\\\\ y&=(y'+\\tfrac12)\\cdot\\frac{${H}}{${params.height}}-\\tfrac12 \\end{aligned}`;
      break;
    case "grayscale":
      tex = `Y=0.299R+0.587G+0.114B`;
      break;
    case "brightness": {
      const sign = params.beta < 0 ? "-" : "+";
      tex = `g(x,y)=\\mathrm{clip}\\bigl(f(x,y)${sign}${Math.abs(params.beta)}\\bigr)`;
      break;
    }
    case "contrast": {
      const L = fmt(params.low, 0);
      const H = fmt(params.high, 0);
      tex = `\\begin{aligned} g&=\\mathrm{clip}\\!\\left(\\frac{255}{${H}-${L}}(f-${L})\\right)\\\\ r_1&=${L},\\; r_2=${H} \\end{aligned}`;
      break;
    }
    case "gaussian_blur": {
      const n = gaussianKernel(params.sigma).length;
      tex = `\\begin{aligned} w(s,t)&\\propto\\exp\\!\\left(-\\frac{s^{2}+t^{2}}{2\\sigma^{2}}\\right)\\\\ \\sigma&=${fmt(params.sigma, 1)},\\;\\text{window }${n}\\times${n} \\end{aligned}`;
      break;
    }
    case "median":
      tex = `\\hat{f}(x,y)=\\mathrm{median}\\{g(s,t)\\},\\; S_{xy}=${params.size}\\times${params.size}`;
      break;
    case "salt_pepper":
      tex = `\\begin{aligned} p(z)&=\\begin{cases} ${fmt(params.pepper)} & z=0 \\\\ ${fmt(params.salt)} & z=255 \\\\ 1-P_a-P_b & \\text{else}\\end{cases}\\\\ g&=\\begin{cases} 0 & u<P_a \\\\ 255 & P_a\\le u<P_a+P_b \\\\ f & \\text{else}\\end{cases}\\end{aligned}`;
      break;
    case "quantization": {
      const levels = 2 ** params.bits;
      const step = params.bits >= 8 ? 1 : 255 / (levels - 1);
      tex = `\\begin{aligned} \\Delta&=\\dfrac{255}{2^{${params.bits}}-1}=${fmt(step, 1)}\\\\ g&=\\mathrm{clip}\\!\\left(\\Delta\\cdot\\mathrm{round}(f/\\Delta)\\right)\\\\ \\sigma_e^{2}&=\\Delta^{2}/12 \\end{aligned}`;
      break;
    }
    default:
      return null;
  }
  return <MathTex tex={tex} display={tool === "contrast" || tool === "resize" || tool === "gaussian_blur" || tool === "salt_pepper" || tool === "quantization"} />;
}

function CropOverlay({ img, rect, onChange }) {
  const ref = useRef(null);
  const start = useRef(null);

  function toImage(e) {
    const b = ref.current.getBoundingClientRect();
    return {
      x: clampInt(((e.clientX - b.left) / b.width) * img.width, 0, img.width),
      y: clampInt(((e.clientY - b.top) / b.height) * img.height, 0, img.height),
    };
  }
  function update(e) {
    const p = toImage(e), s = start.current;
    const x = Math.min(s.x, p.x), y = Math.min(s.y, p.y);
    onChange({
      x: Math.min(x, img.width - 1), y: Math.min(y, img.height - 1),
      width: Math.max(1, Math.abs(p.x - s.x)), height: Math.max(1, Math.abs(p.y - s.y)),
    });
  }
  return (
    <div
      ref={ref}
      className="crop-overlay"
      onPointerDown={(e) => { e.currentTarget.setPointerCapture(e.pointerId); start.current = toImage(e); update(e); }}
      onPointerMove={(e) => { if (start.current) update(e); }}
      onPointerUp={() => { start.current = null; }}
    >
      <div
        className="crop-rect"
        style={{
          left: `${(rect.x / img.width) * 100}%`,
          top: `${(rect.y / img.height) * 100}%`,
          width: `${(rect.width / img.width) * 100}%`,
          height: `${(rect.height / img.height) * 100}%`,
        }}
      />
    </div>
  );
}

function SampleCarousel({ index, onIndexChange, onSelect, loading, selectedId }) {
  const n = BUILTIN_SAMPLES.length;
  const slots = [-2, -1, 0, 1, 2].map((offset) => {
    const i = (index + offset + n) % n;
    return { offset, i, sample: BUILTIN_SAMPLES[i] };
  });
  const current = BUILTIN_SAMPLES[index];

  return (
    <section className="sample-carousel" aria-label="Built-in sample images">
      <div className="sample-carousel-head">
        <h3>Sample images</h3>
        <p className="hint">Click a square to bring it forward. Click the large photo to load it.</p>
      </div>
      <div className="carousel">
        <button type="button" className="carousel-nav" aria-label="Previous sample"
          onClick={() => onIndexChange((index - 1 + n) % n)} disabled={loading}>
          ‹
        </button>
        <div className="carousel-stage">
          {slots.map(({ offset, i, sample }) => {
            const center = offset === 0;
            return (
              <button
                key={sample.id}
                type="button"
                className={
                  "carousel-card" +
                  (center ? " center" : offset === -1 || offset === 1 ? " near" : " far") +
                  (selectedId === sample.id ? " loaded" : "")
                }
                aria-label={center ? `Load ${sample.label}` : `Show ${sample.label}`}
                onClick={() => (center ? onSelect(sample) : onIndexChange(i))}
                disabled={loading}
              >
                <img src={sample.url} alt={center ? sample.label : ""} draggable={false} />
              </button>
            );
          })}
        </div>
        <button type="button" className="carousel-nav" aria-label="Next sample"
          onClick={() => onIndexChange((index + 1) % n)} disabled={loading}>
          ›
        </button>
      </div>
      <div className="carousel-caption">
        <strong>{current.label}</strong>
        <span>{loading ? "Loading…" : current.hint}</span>
      </div>
    </section>
  );
}

function ImageCanvas({ image, label, note, overlay, footer }) {
  const ref = useRef(null);
  useEffect(() => {
    if (!image || !ref.current) return;
    const c = ref.current;
    c.width = image.width;
    c.height = image.height;
    c.getContext("2d").putImageData(image, 0, 0);
  }, [image]);
  return (
    <figure className="frame">
      <figcaption>
        <span>{label}</span>
        {note && <span className="note">{note}</span>}
      </figcaption>
      <div className="frame-body">
        <div className="canvas-wrap">
          <canvas ref={ref} aria-label={label} />
          {overlay}
        </div>
      </div>
      {footer && <div className="frame-foot">{footer}</div>}
    </figure>
  );
}

function Histogram({ image }) {
  const s = useMemo(() => (image ? stats(image) : null), [image]);
  if (!s) return null;
  const max = Math.max(...s.hist);
  const pts = s.hist.map((v, k) => `${k},${100 - (v / max) * 100}`).join(" ");
  return (
    <section className="panel">
      <h3>Histogram of result</h3>
      <svg viewBox="0 0 255 100" preserveAspectRatio="none" className="hist" role="img"
        aria-label="Luma histogram of the processed image">
        <polygon points={`0,100 ${pts} 255,100`} />
        <line x1={s.mean} x2={s.mean} y1="0" y2="100" className="mean-line" />
      </svg>
      <div className="axis"><span>0</span><span>255</span></div>
      <dl className="stats">
        <div><dt>Mean μ</dt><dd>{fmt(s.mean, 1)}</dd></div>
        <div><dt>Median</dt><dd>{s.median}</dd></div>
        <div><dt>Std σ</dt><dd>{fmt(s.std, 1)}</dd></div>
      </dl>
      <p className="hint">Equations (12)–(14). The vertical line marks the mean.</p>
    </section>
  );
}

function opLabel(op) {
  const p = op.params || {};
  switch (op.type) {
    case "crop": return `Crop ${p.width}×${p.height} at (${p.x}, ${p.y})`;
    case "resize": return `Resize to ${p.width}×${p.height} (${p.method})`;
    case "grayscale": return "Grayscale";
    case "brightness": return `Brightness β = ${p.beta > 0 ? "+" : ""}${p.beta}`;
    case "contrast": return `Contrast stretch [${fmt(p.low, 0)}, ${fmt(p.high, 0)}]`;
    case "gaussian_blur": return `Gaussian blur σ = ${fmt(p.sigma, 1)}`;
    case "median": return `Median filter ${p.size}×${p.size}`;
    case "salt_pepper": return `Salt & pepper P_a = ${fmt(p.pepper)}, P_b = ${fmt(p.salt)}`;
    case "quantization": return `Quantization ${p.bits}-bit`;
    default: return op.type;
  }
}

function NumberField({ label, value, min, max, onChange }) {
  return (
    <label className="field">
      <span>{label}</span>
      <input type="number" value={value} min={min} max={max}
        onChange={(e) => onChange(clampInt(e.target.value, min, max))} />
    </label>
  );
}

function ToolControls({ tool, params, setParams, img }) {
  const t = TOOLS[tool];
  if (tool === "crop") {
    const set = (k, v) => {
      const p = { ...params, [k]: v };
      p.x = Math.min(p.x, img.width - 1);
      p.y = Math.min(p.y, img.height - 1);
      p.width = Math.min(Math.max(1, p.width), img.width - p.x);
      p.height = Math.min(Math.max(1, p.height), img.height - p.y);
      setParams(p);
    };
    return (
      <div className="fields">
        <NumberField label="X" value={params.x} min={0} max={img.width - 1} onChange={(v) => set("x", v)} />
        <NumberField label="Y" value={params.y} min={0} max={img.height - 1} onChange={(v) => set("y", v)} />
        <NumberField label="Width" value={params.width} min={1} max={img.width} onChange={(v) => set("width", v)} />
        <NumberField label="Height" value={params.height} min={1} max={img.height} onChange={(v) => set("height", v)} />
      </div>
    );
  }
  if (tool === "resize") {
    const ratio = img.height / img.width;
    const setW = (v) => setParams({ ...params, width: v, height: params.lock ? clampInt(v * ratio, 1, 4000) : params.height });
    const setH = (v) => setParams({ ...params, height: v, width: params.lock ? clampInt(v / ratio, 1, 4000) : params.width });
    return (
      <>
        <div className="fields">
          <NumberField label="Width" value={params.width} min={1} max={4000} onChange={setW} />
          <NumberField label="Height" value={params.height} min={1} max={4000} onChange={setH} />
        </div>
        <label className="check">
          <input type="checkbox" checked={params.lock} onChange={(e) => setParams({ ...params, lock: e.target.checked })} />
          Keep aspect ratio
        </label>
        <label className="select inline">
          <span>Interpolation</span>
          <select value={params.method} onChange={(e) => setParams({ ...params, method: e.target.value })}>
            <option value="bilinear">Bilinear</option>
            <option value="nearest">Nearest neighbour</option>
          </select>
        </label>
      </>
    );
  }
  if (tool === "salt_pepper") {
    const setPepper = (v) => setParams({ ...params, pepper: Math.min(v, 1 - params.salt) });
    const setSalt = (v) => setParams({ ...params, salt: Math.min(v, 1 - params.pepper) });
    return (
      <div className="contrast-sliders">
        <label className="slider">
          <span>Pepper P_a (black)<output>{fmt(params.pepper)}</output></span>
          <input type="range" min={0} max={0.25} step={0.01} value={params.pepper}
            onChange={(e) => setPepper(Number(e.target.value))} />
        </label>
        <label className="slider">
          <span>Salt P_b (white)<output>{fmt(params.salt)}</output></span>
          <input type="range" min={0} max={0.25} step={0.01} value={params.salt}
            onChange={(e) => setSalt(Number(e.target.value))} />
        </label>
      </div>
    );
  }
  if (tool === "contrast") {
    const setLow = (v) => setParams({ ...params, low: Math.min(v, params.high - 1) });
    const setHigh = (v) => setParams({ ...params, high: Math.max(v, params.low + 1) });
    return (
      <div className="contrast-sliders">
        <label className="slider">
          <span>Low r₁ (darks)<output>{params.low}</output></span>
          <input type="range" min={-100} max={254} step={1} value={params.low}
            onChange={(e) => setLow(Number(e.target.value))} />
        </label>
        <label className="slider">
          <span>High r₂ (brights)<output>{params.high}</output></span>
          <input type="range" min={1} max={355} step={1} value={params.high}
            onChange={(e) => setHigh(Number(e.target.value))} />
        </label>
      </div>
    );
  }
  if (!t.slider) return null;
  const s = t.slider;
  const value = params[s.key];
  const shown = tool === "median" ? `${value} × ${value}` : s.step < 1 ? fmt(value, s.step < 0.1 ? 2 : 1) : value;
  return (
    <label className="slider">
      <span>{s.name}<output>{shown}</output></span>
      <input type="range" min={s.min} max={s.max} step={s.step} value={value}
        onChange={(e) => setParams({ [s.key]: Number(e.target.value) })} />
    </label>
  );
}

export default function App() {
  const [imageId, setImageId] = useState(null);
  const [original, setOriginal] = useState(null);
  const [committed, setCommitted] = useState(null);
  const [ops, setOps] = useState([]);
  const [past, setPast] = useState([]);
  const [future, setFuture] = useState([]);
  const [tool, setTool] = useState("brightness");
  const [params, setParams] = useState(defaultParams("brightness"));
  const [preview, setPreview] = useState(null);
  const [computing, setComputing] = useState(false);
  // true once the user has changed the controls (or opened a filter tool)
  const [dirty, setDirty] = useState(false);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [log, setLog] = useState([]);
  const [dragging, setDragging] = useState(false);
  const [backend, setBackend] = useState("checking");
  const [busy, setBusy] = useState(false);
  const [fileName, setFileName] = useState("image");
  const [exportFormat, setExportFormat] = useState("png");
  const [exported, setExported] = useState("");
  const [toolsOpen, setToolsOpen] = useState(true);
  const [sampleIndex, setSampleIndex] = useState(0);
  const [selectedSampleId, setSelectedSampleId] = useState("");
  const [sampleLoading, setSampleLoading] = useState(false);
  const fileRef = useRef(null);
  const timer = useRef(0);
  const historyTimer = useRef(0);
  const pendingSnap = useRef(null);
  const skipParamReset = useRef(false);

  useEffect(() => onRequest((e) => setLog((l) => [e, ...l].slice(0, 40))), []);
  useEffect(() => onStatus(setBackend), []);
  useEffect(() => { checkBackend(); }, []);

  // new result (undo, redo, reset, new image): reset the tool's controls
  useEffect(() => {
    if (skipParamReset.current) {
      skipParamReset.current = false;
      return;
    }
    if (committed) setParams(defaultParams(tool, committed));
    setDirty(false);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [committed]);

  function snapshot() {
    return { tool, params, ops: ops.slice(), dirty };
  }

  function remember(before) {
    setPast((h) => [...h, before].slice(-40));
    setFuture([]);
  }

  function editParams(p) {
    if (!pendingSnap.current) pendingSnap.current = snapshot();
    setParams(p);
    setDirty(true);
    clearTimeout(historyTimer.current);
    historyTimer.current = setTimeout(() => {
      const before = pendingSnap.current;
      pendingSnap.current = null;
      if (before) remember(before);
    }, 350);
  }

  const pivot = useMemo(() => (committed ? stats(committed).mean : 128), [committed]);
  const neutral = isNeutral(tool, params, committed);

  // live preview of the tool being adjusted (local only, not sent to the backend)
  useEffect(() => {
    if (!committed) return;
    clearTimeout(timer.current);
    if (tool === "crop" || neutral || !dirty) {
      setPreview(null); setComputing(false); return;
    }
    const run = () => {
      try {
        setPreview(applyOp(committed, { type: tool, params: opParams(tool, params) }));
      } catch {
        setPreview(null);
      }
      setComputing(false);
    };
    if (TOOLS[tool].heavy) {
      setComputing(true);
      timer.current = setTimeout(run, 150);
    } else {
      timer.current = setTimeout(run, 0);
    }
    return () => clearTimeout(timer.current);
  }, [committed, tool, params, neutral, dirty]);

  function start(id, img, note = "", name = "image", sampleId = "") {
    setImageId(id);
    setFileName(name.replace(/\.[^.]+$/, "") || "image");
    setExported("");
    setOriginal(img);
    setCommitted(img);
    setOps([]);
    setPast([]);
    setFuture([]);
    pendingSnap.current = null;
    setError("");
    setNotice(note);
    setSelectedSampleId(sampleId);
  }

  async function handleFile(file) {
    if (!file) return;
    try {
      const { imageId: id, image, scaled } = await upload(file);
      start(id, image, scaled ? "Large image: reduced to 1600 px on the longest side." : "", file.name, "");
    } catch (e) {
      setError(e.message);
    }
  }

  async function loadBuiltin(sample) {
    setSampleLoading(true);
    setError("");
    try {
      const { imageId: id, image, scaled } = await uploadFromUrl(sample.url, sample.file);
      start(
        id,
        image,
        scaled ? "Large image: reduced to 1600 px on the longest side." : "",
        sample.file,
        sample.id,
      );
    } catch (e) {
      setError(e.message);
    } finally {
      setSampleLoading(false);
    }
  }

  async function selectTool(next) {
    if (next === tool) return;
    clearTimeout(historyTimer.current);
    remember(pendingSnap.current || snapshot());
    pendingSnap.current = null;
    let base = committed;
    if (dirty && !isNeutral(tool, params, committed)) {
      skipParamReset.current = true;
      const op = { type: tool, params: opParams(tool, params) };
      const out = await commit([...ops, op]);
      if (!out) {
        skipParamReset.current = false;
        return;
      }
      base = out;
    }
    setTool(next);
    setParams(defaultParams(next, base));
    setDirty(next === "gaussian_blur" || next === "median" || next === "grayscale" || next === "salt_pepper" || next === "quantization");
  }

  // Returns true when the backend accepted the new operation list.
  async function commit(nextOps) {
    setBusy(true);
    try {
      const out = await processImage(imageId, nextOps);
      setOps(nextOps);
      setCommitted(out);
      setError("");
      setExported("");
      return out;
    } catch (e) {
      setError(e.message);
      return false;
    } finally {
      setBusy(false);
    }
  }

  async function applySnapshot(s) {
    setTool(s.tool);
    setParams(s.params);
    setDirty(s.dirty);
    if (JSON.stringify(s.ops) !== JSON.stringify(ops)) {
      skipParamReset.current = true;
      await commit(s.ops);
    }
  }

  function undo() {
    if (!past.length || busy) return;
    const prev = past[past.length - 1];
    setFuture((f) => [snapshot(), ...f]);
    setPast((h) => h.slice(0, -1));
    applySnapshot(prev);
  }

  function redoOp() {
    if (!future.length || busy) return;
    const next = future[0];
    setPast((h) => [...h, snapshot()]);
    setFuture((f) => f.slice(1));
    applySnapshot(next);
  }

  function reset() {
    clearTimeout(historyTimer.current);
    pendingSnap.current = null;
    setPast([]);
    setFuture([]);
    setOps([]);
    setParams(defaultParams(tool, original));
    setDirty(false);
    if (committed !== original) {
      skipParamReset.current = true;
      setCommitted(original);
    }
  }

  async function handleExport() {
    setBusy(true);
    setExported("");
    const live = dirty && !isNeutral(tool, params, committed)
      ? [{ type: tool, params: opParams(tool, params) }]
      : [];
    try {
      const blob = await exportImage(imageId, [...ops, ...live], exportFormat);
      const name = `${fileName}_processed.${EXPORT_FORMATS[exportFormat].ext}`;
      saveBlob(blob, name);
      setExported(`Downloaded ${name}`);
      setError("");
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy(false);
    }
  }

  const t = TOOLS[tool];
  const shown = preview || committed;
  const processedNote = computing
    ? "Computing…"
    : shown ? `${shown.width}×${shown.height}` : "";

  const exportFooter = committed && (
    <>
      <div className="export-row">
        <select aria-label="Format" value={exportFormat} onChange={(e) => setExportFormat(e.target.value)}>
          {Object.entries(EXPORT_FORMATS).map(([k, v]) => (
            <option key={k} value={k} disabled={k === "tiff" && backend !== "server"}>
              {v.label}{k === "tiff" && backend !== "server" ? " (needs backend)" : ""}
            </option>
          ))}
        </select>
        <button className="primary" onClick={handleExport} disabled={busy}>Download</button>
      </div>
      {exported && <p className="hint">{exported}</p>}
    </>
  );

  return (
    <div className="app">
      <header className="top">
        <div>
          <h1>Image Lab</h1>
          <p className="sub">Prototype of the interactive web-based image processing platform</p>
        </div>
        <span
          className={"status " + backend}
          title={backend === "server" ? `Connected to ${API_URL}` : `No backend at ${API_URL}; requests are simulated in the browser`}
        >
          <span className="dot" aria-hidden="true" />
          {backend === "server" ? "Backend connected" : backend === "checking" ? "Checking backend…" : "Backend offline: simulated in browser"}
        </span>
      </header>

      <SampleCarousel
        index={sampleIndex}
        onIndexChange={setSampleIndex}
        onSelect={loadBuiltin}
        loading={sampleLoading}
        selectedId={selectedSampleId}
      />

      {!original ? (
        <main className="empty">
          <div
            className={"drop" + (dragging ? " over" : "")}
            onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
            onDragLeave={() => setDragging(false)}
            onDrop={(e) => { e.preventDefault(); setDragging(false); handleFile(e.dataTransfer.files[0]); }}
          >
            <h2>Load an image to start</h2>
            <p>Pick a sample above, drop a JPEG, PNG or TIFF here, or choose a file from your computer.</p>
            <div className="row">
              <button className="primary" onClick={() => fileRef.current.click()}>Choose image</button>
            </div>
            {error && <p className="error" role="alert">{error}</p>}
          </div>
        </main>
      ) : (
        <main className={"work" + (toolsOpen ? "" : " tools-closed")}>
          <aside className={"tools panel" + (toolsOpen ? "" : " closed")}>
            <div className="tools-head">
              {toolsOpen && <h3>Tools</h3>}
              <button
                type="button"
                className="tools-toggle"
                aria-expanded={toolsOpen}
                aria-label={toolsOpen ? "Hide tools" : "Show tools"}
                onClick={() => setToolsOpen((open) => !open)}
              >
                {toolsOpen ? "‹" : "›"}
              </button>
              {!toolsOpen && <span className="tools-rail">Tools</span>}
            </div>
            {toolsOpen && (
            <>
            <div className="tool-list" role="tablist" aria-label="Tools">
              {Object.entries(TOOLS).map(([k, v]) => (
                <button key={k} role="tab" aria-selected={tool === k}
                  className={"tool" + (tool === k ? " on" : "")} onClick={() => selectTool(k)}>
                  {v.label}
                </button>
              ))}
            </div>

            <div className="formula">
              <Formula tool={tool} params={params} pivot={pivot} img={committed} />
              <span className="eqno">({t.eq})</span>
            </div>
            <p className="hint">{t.describe}</p>

            <ToolControls tool={tool} params={params} setParams={editParams} img={committed} />

            <h3 className="gap">History</h3>
            {ops.length === 0 && (neutral || !dirty) ? (
              <p className="hint">No operations yet. Each parameter change can be undone.</p>
            ) : (
              <ol className="history">
                {ops.map((op, i) => <li key={i}>{opLabel(op)}</li>)}
                {dirty && !neutral && <li>{opLabel({ type: tool, params })}</li>}
              </ol>
            )}
            <div className="row">
              <button onClick={undo} disabled={!past.length || busy}>Undo</button>
              <button onClick={redoOp} disabled={!future.length || busy}>Redo</button>
              <button onClick={reset} disabled={(!past.length && !ops.length && (neutral || !dirty)) || busy}>Reset</button>
            </div>

            {error && <p className="error" role="alert">{error}</p>}
            </>
            )}
          </aside>

          <section className="canvases">
            {notice && <p className="notice">{notice}</p>}
            <div className="pair">
              <ImageCanvas
                image={original}
                label="Original"
                note={`${original.width}×${original.height}`}
                footer={(
                  <div className="frame-foot-row">
                    <button onClick={() => fileRef.current.click()}>Load another image</button>
                  </div>
                )}
              />
              <ImageCanvas
                image={shown}
                label="Processed"
                note={processedNote}
                overlay={tool === "crop" && committed && (
                  <CropOverlay img={committed} rect={params} onChange={editParams} />
                )}
                footer={exportFooter}
              />
            </div>
            {tool === "crop" && <p className="hint">Drag on the processed image to select the area to keep.</p>}
          </section>

          <div className="bottom">
            <Histogram image={shown} />
            <section className="panel log">
              <h3>Requests to backend</h3>
              <p className="hint">
                {backend === "server"
                  ? "Live requests to the FastAPI backend. The picture updates as you move the controls."
                  : "Simulated requests: what the frontend will send once the backend is running."}
              </p>
              <ul>
                {log.map((e) => (
                  <li key={e.id}>
                    <div className="req">
                      <span className="method">{e.method}</span>
                      <span className="path">{e.path}</span>
                      <span className={"code" + (e.status >= 400 ? " bad" : "")}>{e.status}</span>
                      <span className="ms">{e.ms} ms</span>
                    </div>
                    {e.body && <code>{e.body}</code>}
                    <code className="res">{e.result}</code>
                  </li>
                ))}
              </ul>
            </section>
          </div>
        </main>
      )}

      <input ref={fileRef} type="file" accept="image/jpeg,image/png,image/tiff,.tif,.tiff" hidden
        onChange={(e) => { handleFile(e.target.files[0]); e.target.value = ""; }} />
    </div>
  );
}
