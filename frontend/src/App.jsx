import React, { useEffect, useMemo, useRef, useState } from "react";
import {
  upload, uploadSample, processImage, exportImage, saveBlob, EXPORT_FORMATS,
  onRequest, onStatus, checkBackend, API_URL,
} from "./api.js";
import { applyOp, stats, sampleImage, gaussianKernel } from "./imageOps.js";

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
    slider: { key: "alpha", min: 0.2, max: 3, step: 0.05, name: "α (gain)" },
    describe: "Stretches intensities around the image mean c. α > 1 increases contrast.",
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
};

function defaultParams(tool, img) {
  const w = img ? img.width : 1, h = img ? img.height : 1;
  switch (tool) {
    case "crop":
      return { x: Math.round(w * 0.1), y: Math.round(h * 0.1), width: Math.max(1, Math.round(w * 0.8)), height: Math.max(1, Math.round(h * 0.8)) };
    case "resize": return { width: w, height: h, method: "bilinear", lock: true };
    case "brightness": return { beta: 0 };
    case "contrast": return { alpha: 1 };
    case "gaussian_blur": return { sigma: 1.5 };
    case "median": return { size: 3 };
    default: return {};
  }
}

function isNeutral(tool, p, img) {
  if (!img) return true;
  if (tool === "brightness") return p.beta === 0;
  if (tool === "contrast") return p.alpha === 1;
  if (tool === "resize") return p.width === img.width && p.height === img.height;
  if (tool === "crop") return p.x === 0 && p.y === 0 && p.width === img.width && p.height === img.height;
  return false;
}

// parameters sent to the backend (UI-only fields removed)
function opParams(tool, p) {
  if (tool === "resize") return { width: p.width, height: p.height, method: p.method };
  return { ...p };
}

const fmt = (v, d = 2) => Number(v).toFixed(d);
const clampInt = (v, lo, hi) => Math.min(Math.max(Math.round(Number(v) || 0), lo), hi);

function Formula({ tool, params, pivot, img }) {
  const W = img ? img.width : 0, H = img ? img.height : 0;
  switch (tool) {
    case "crop":
      return (
        <span className="math">
          <i>g</i>(<i>x</i>,<i>y</i>) = <i>f</i>(<i>x</i> + <b>{params.x}</b>, <i>y</i> + <b>{params.y}</b>),
          0 ≤ <i>x</i> &lt; <b>{params.width}</b>, 0 ≤ <i>y</i> &lt; <b>{params.height}</b>
        </span>
      );
    case "resize":
      return params.method === "nearest" ? (
        <span className="math">
          <i>g</i>(<i>x</i>′,<i>y</i>′) = <i>f</i>(round(<i>x</i>), round(<i>y</i>)),
          <i> x</i> = (<i>x</i>′ + ½)·<b>{W}</b>/<b>{params.width}</b> − ½
        </span>
      ) : (
        <span className="math">
          bilinear from <i>f</i> at <i>x</i> = (<i>x</i>′ + ½)·<b>{W}</b>/<b>{params.width}</b> − ½,
          <i> y</i> = (<i>y</i>′ + ½)·<b>{H}</b>/<b>{params.height}</b> − ½
        </span>
      );
    case "grayscale":
      return <span className="math"><i>Y</i> = 0.299<i>R</i> + 0.587<i>G</i> + 0.114<i>B</i></span>;
    case "brightness":
      return (
        <span className="math">
          <i>g</i>(<i>x</i>,<i>y</i>) = clip(<i>f</i>(<i>x</i>,<i>y</i>) {params.beta < 0 ? "−" : "+"} <b>{Math.abs(params.beta)}</b>)
        </span>
      );
    case "contrast":
      return (
        <span className="math">
          <i>g</i> = clip(<b>{fmt(params.alpha)}</b>(<i>f</i> − <b>{fmt(pivot, 1)}</b>) + <b>{fmt(pivot, 1)}</b>)
        </span>
      );
    case "gaussian_blur": {
      const n = gaussianKernel(params.sigma).length;
      return (
        <span className="math">
          <i>w</i>(<i>s</i>,<i>t</i>) ∝ exp(−(<i>s</i>² + <i>t</i>²) / 2<i>σ</i>²), <i>σ</i> = <b>{fmt(params.sigma, 1)}</b>,
          window <b>{n}×{n}</b>
        </span>
      );
    }
    case "median":
      return (
        <span className="math">
          <i>f̂</i>(<i>x</i>,<i>y</i>) = median{"{"}<i>g</i>(<i>s</i>,<i>t</i>){"}"}, <i>S<sub>xy</sub></i> = <b>{params.size}×{params.size}</b>
        </span>
      );
    default:
      return null;
  }
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
    case "contrast": return `Contrast α = ${fmt(p.alpha)}`;
    case "gaussian_blur": return `Gaussian blur σ = ${fmt(p.sigma, 1)}`;
    case "median": return `Median filter ${p.size}×${p.size}`;
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
  const [redo, setRedo] = useState([]);
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
  const fileRef = useRef(null);
  const timer = useRef(0);

  useEffect(() => onRequest((e) => setLog((l) => [e, ...l].slice(0, 40))), []);
  useEffect(() => onStatus(setBackend), []);
  useEffect(() => { checkBackend(); }, []);

  // new result (apply, undo, redo, reset, new image): reset the tool's controls
  useEffect(() => {
    if (committed) setParams(defaultParams(tool, committed));
    setDirty(false);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [committed]);

  function editParams(p) {
    setParams(p);
    setDirty(true);
  }

  const pivot = useMemo(() => (committed ? stats(committed).mean : 128), [committed]);
  const neutral = isNeutral(tool, params, committed);

  // live preview of the tool being adjusted (local only, not sent to the backend)
  useEffect(() => {
    if (!committed) return;
    clearTimeout(timer.current);
    if (tool === "crop" || tool === "grayscale" || neutral || !dirty) {
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

  function start(id, img, note = "", name = "image") {
    setImageId(id);
    setFileName(name.replace(/\.[^.]+$/, "") || "image");
    setExported("");
    setOriginal(img);
    setCommitted(img);
    setOps([]);
    setRedo([]);
    setError("");
    setNotice(note);
  }

  async function handleFile(file) {
    if (!file) return;
    try {
      const { imageId: id, image, scaled } = await upload(file);
      start(id, image, scaled ? "Large image: reduced to 1600 px on the longest side." : "", file.name);
    } catch (e) {
      setError(e.message);
    }
  }

  async function loadSample() {
    try {
      const { imageId: id, image } = await uploadSample(sampleImage());
      start(id, image, "", "sample");
    } catch (e) {
      setError(e.message);
    }
  }

  function selectTool(t) {
    setTool(t);
    setParams(defaultParams(t, committed));
    // filters have no "do nothing" value, so show their default effect right away
    setDirty(t === "gaussian_blur" || t === "median");
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
      return true;
    } catch (e) {
      setError(e.message);
      return false;
    } finally {
      setBusy(false);
    }
  }

  async function apply() {
    if (neutral) return;
    const op = { type: tool, params: opParams(tool, params) };
    if (await commit([...ops, op])) setRedo([]);
  }

  async function undo() {
    if (!ops.length) return;
    const last = ops[ops.length - 1];
    if (await commit(ops.slice(0, -1))) setRedo((r) => [last, ...r]);
  }

  async function redoOp() {
    if (!redo.length) return;
    const [first, ...rest] = redo;
    if (await commit([...ops, first])) setRedo(rest);
  }

  function reset() {
    setRedo([]);
    setOps([]);
    setCommitted(original);
  }

  async function handleExport() {
    setBusy(true);
    setExported("");
    try {
      const blob = await exportImage(imageId, ops, exportFormat);
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
    ? "Computing preview…"
    : preview
      ? `Preview ${preview.width}×${preview.height}, not applied`
      : committed ? `${committed.width}×${committed.height} · ${ops.length} step${ops.length === 1 ? "" : "s"}` : "";

  const exportFooter = committed && (
    <>
      <div className="export-row">
        <label className="select">
          <span>Format</span>
          <select value={exportFormat} onChange={(e) => setExportFormat(e.target.value)}>
            {Object.entries(EXPORT_FORMATS).map(([k, v]) => (
              <option key={k} value={k} disabled={k === "tiff" && backend !== "server"}>
                {v.label}{k === "tiff" && backend !== "server" ? " (needs backend)" : ""}
              </option>
            ))}
          </select>
        </label>
        <button className="primary grow" onClick={handleExport} disabled={busy}>Download result</button>
      </div>
      <p className="hint">
        {preview || (tool === "crop" && !neutral)
          ? "Downloads the applied steps only. Apply or cancel the current change first."
          : exported || `Saves the result of ${ops.length} applied step${ops.length === 1 ? "" : "s"}.`}
      </p>
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

      {!original ? (
        <main className="empty">
          <div
            className={"drop" + (dragging ? " over" : "")}
            onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
            onDragLeave={() => setDragging(false)}
            onDrop={(e) => { e.preventDefault(); setDragging(false); handleFile(e.dataTransfer.files[0]); }}
          >
            <h2>Load an image to start</h2>
            <p>Drop a JPEG, PNG or TIFF file here, or choose one from your computer.</p>
            <div className="row">
              <button className="primary" onClick={() => fileRef.current.click()}>Choose image</button>
              <button onClick={loadSample}>Use sample image</button>
            </div>
            {error && <p className="error" role="alert">{error}</p>}
          </div>
        </main>
      ) : (
        <main className="work">
          <aside className="tools panel">
            <h3>Tools</h3>
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

            <div className="row">
              <button className="primary" onClick={apply} disabled={neutral || busy}>Apply {t.label.toLowerCase()}</button>
              {tool !== "grayscale" && (
                <button onClick={() => { setParams(defaultParams(tool, committed)); setDirty(false); }} disabled={busy}>Cancel</button>
              )}
            </div>

            <h3 className="gap">History</h3>
            {ops.length === 0 ? (
              <p className="hint">No operations yet. Applied steps appear here in order.</p>
            ) : (
              <ol className="history">{ops.map((op, i) => <li key={i}>{opLabel(op)}</li>)}</ol>
            )}
            <div className="row">
              <button onClick={undo} disabled={!ops.length || busy}>Undo</button>
              <button onClick={redoOp} disabled={!redo.length || busy}>Redo</button>
              <button onClick={reset} disabled={!ops.length || busy}>Reset</button>
            </div>

            <h3 className="gap">Image</h3>
            <div className="row">
              <button onClick={() => fileRef.current.click()}>Load another image</button>
            </div>
            {error && <p className="error" role="alert">{error}</p>}
          </aside>

          <section className="canvases">
            {notice && <p className="notice">{notice}</p>}
            <div className="pair">
              <ImageCanvas image={original} label="Original" note={`${original.width}×${original.height}`} />
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
                  ? "Live requests to the FastAPI backend. Previews stay local until you apply."
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
