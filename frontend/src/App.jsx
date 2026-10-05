// Umut tarafindan test amaciyla eklendi
import React, { useEffect, useMemo, useRef, useState } from "react";
import { upload, processImage, registerSample, onRequest } from "./api.js";
import { applyOp, stats, sampleImage } from "./imageOps.js";

const TOOLS = {
  grayscale: {
    label: "Grayscale",
    eq: "6",
    params: {},
    describe: "Maps the three colour channels to one luma channel. Not reversible.",
  },
  brightness: {
    label: "Brightness",
    eq: "8",
    params: { beta: 0 },
    slider: { key: "beta", min: -100, max: 100, step: 1, name: "β (offset)" },
    describe: "Adds the same offset to every pixel. Values past 0 or 255 are clipped.",
  },
  contrast: {
    label: "Contrast",
    eq: "9",
    params: { alpha: 1 },
    slider: { key: "alpha", min: 0.2, max: 3, step: 0.05, name: "α (gain)" },
    describe: "Stretches intensities around the image mean c. α > 1 increases contrast.",
  },
};

function fmt(v, d = 2) {
  return Number(v).toFixed(d);
}

function Formula({ tool, params, pivot }) {
  if (tool === "grayscale")
    return (
      <span className="math">
        <i>Y</i> = 0.299<i>R</i> + 0.587<i>G</i> + 0.114<i>B</i>
      </span>
    );
  if (tool === "brightness") {
    const b = params.beta;
    return (
      <span className="math">
        <i>g</i>(<i>x</i>,<i>y</i>) = clip(<i>f</i>(<i>x</i>,<i>y</i>) {b < 0 ? "−" : "+"}{" "}
        <b>{Math.abs(b)}</b>)
      </span>
    );
  }
  return (
    <span className="math">
      <i>g</i> = clip(<b>{fmt(params.alpha)}</b>(<i>f</i> − <b>{fmt(pivot, 1)}</b>) +{" "}
      <b>{fmt(pivot, 1)}</b>)
    </span>
  );
}

function ImageCanvas({ image, label, note }) {
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
        <canvas ref={ref} aria-label={label} />
      </div>
    </figure>
  );
}

function Histogram({ image }) {
  const s = useMemo(() => (image ? stats(image) : null), [image]);
  if (!s) return null;
  const max = Math.max(...s.hist);
  const pts = s.hist
    .map((v, k) => `${k},${100 - (v / max) * 100}`)
    .join(" ");
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
  if (op.type === "grayscale") return "Grayscale";
  if (op.type === "brightness") return `Brightness β = ${op.params.beta > 0 ? "+" : ""}${op.params.beta}`;
  return `Contrast α = ${fmt(op.params.alpha)}`;
}

export default function App() {
  const [imageId, setImageId] = useState(null);
  const [original, setOriginal] = useState(null);
  const [committed, setCommitted] = useState(null);
  const [ops, setOps] = useState([]);
  const [redo, setRedo] = useState([]);
  const [tool, setTool] = useState("brightness");
  const [params, setParams] = useState(TOOLS.brightness.params);
  const [preview, setPreview] = useState(null);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [log, setLog] = useState([]);
  const [dragging, setDragging] = useState(false);
  const fileRef = useRef(null);
  const raf = useRef(0);

  useEffect(() => onRequest((e) => setLog((l) => [e, ...l].slice(0, 40))), []);

  const pivot = useMemo(() => (committed ? stats(committed).mean : 128), [committed]);
  const isNeutral =
    (tool === "brightness" && params.beta === 0) ||
    (tool === "contrast" && params.alpha === 1);

  // live preview of the tool being adjusted (local only, not logged)
  useEffect(() => {
    if (!committed) return;
    cancelAnimationFrame(raf.current);
    raf.current = requestAnimationFrame(() => {
      if (tool === "grayscale" || isNeutral) setPreview(null);
      else setPreview(applyOp(committed, { type: tool, params }));
    });
  }, [committed, tool, params, isNeutral]);

  async function start(id, img, note = "") {
    setImageId(id);
    setOriginal(img);
    setCommitted(img);
    setOps([]);
    setRedo([]);
    setError("");
    setNotice(note);
    selectTool(tool);
  }

  async function handleFile(file) {
    if (!file) return;
    try {
      const { imageId: id, image, scaled } = await upload(file);
      start(id, image, scaled ? "Large image: preview reduced to 1600 px on the longest side." : "");
    } catch (e) {
      setError(e.message);
    }
  }

  function loadSample() {
    const img = sampleImage();
    start(registerSample(img), img);
  }

  function selectTool(t) {
    setTool(t);
    setParams({ ...TOOLS[t].params });
  }

  async function commit(nextOps) {
    setOps(nextOps);
    const out = await processImage(imageId, nextOps);
    setCommitted(out);
  }

  async function apply() {
    if (isNeutral) return;
    const op = { type: tool, params: { ...params } };
    setRedo([]);
    await commit([...ops, op]);
    setParams({ ...TOOLS[tool].params });
  }

  async function undo() {
    if (!ops.length) return;
    setRedo((r) => [ops[ops.length - 1], ...r]);
    await commit(ops.slice(0, -1));
  }

  async function redoOp() {
    if (!redo.length) return;
    const [first, ...rest] = redo;
    setRedo(rest);
    await commit([...ops, first]);
  }

  async function reset() {
    setRedo([]);
    setOps([]);
    setCommitted(original);
    setParams({ ...TOOLS[tool].params });
  }

  const t = TOOLS[tool];
  const shown = preview || committed;

  return (
    <div className="app">
      <header className="top">
        <div>
          <h1>Image Lab</h1>
          <p className="sub">Frontend prototype for the interactive image processing platform</p>
        </div>
        <span className="status" title="Requests are simulated in the browser until the FastAPI backend is connected">
          <span className="dot" aria-hidden="true" /> Backend simulated in browser
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
            <div className="tool-list" role="tablist">
              {Object.entries(TOOLS).map(([k, v]) => (
                <button
                  key={k}
                  role="tab"
                  aria-selected={tool === k}
                  className={"tool" + (tool === k ? " on" : "")}
                  onClick={() => selectTool(k)}
                >
                  {v.label}
                </button>
              ))}
            </div>

            <div className="formula">
              <Formula tool={tool} params={params} pivot={pivot} />
              <span className="eqno">({t.eq})</span>
            </div>
            <p className="hint">{t.describe}</p>

            {t.slider && (
              <label className="slider">
                <span>
                  {t.slider.name}
                  <output>{t.slider.step < 1 ? fmt(params[t.slider.key]) : params[t.slider.key]}</output>
                </span>
                <input
                  type="range"
                  min={t.slider.min}
                  max={t.slider.max}
                  step={t.slider.step}
                  value={params[t.slider.key]}
                  onChange={(e) => setParams({ [t.slider.key]: Number(e.target.value) })}
                />
              </label>
            )}

            <div className="row">
              <button className="primary" onClick={apply} disabled={isNeutral}>
                Apply {t.label.toLowerCase()}
              </button>
              {t.slider && (
                <button onClick={() => setParams({ ...t.params })} disabled={isNeutral}>Cancel</button>
              )}
            </div>

            <h3 className="gap">History</h3>
            {ops.length === 0 ? (
              <p className="hint">No operations yet. Applied steps appear here in order.</p>
            ) : (
              <ol className="history">
                {ops.map((op, i) => <li key={i}>{opLabel(op)}</li>)}
              </ol>
            )}
            <div className="row">
              <button onClick={undo} disabled={!ops.length}>Undo</button>
              <button onClick={redoOp} disabled={!redo.length}>Redo</button>
              <button onClick={reset} disabled={!ops.length}>Reset</button>
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
              <ImageCanvas image={shown} label="Processed" note={preview ? "Preview, not applied" : `${ops.length} step${ops.length === 1 ? "" : "s"}`} />
            </div>
          </section>

          <div className="bottom">
            <Histogram image={shown} />
            <section className="panel log">
              <h3>Requests to backend</h3>
              <p className="hint">What the frontend will send to FastAPI. Slider previews stay local.</p>
              <ul>
                {log.map((e) => (
                  <li key={e.id}>
                    <div className="req">
                      <span className="method">{e.method}</span>
                      <span className="path">{e.path}</span>
                      <span className={"code" + (e.status >= 400 ? " bad" : "")}>{e.status}</span>
                      <span className="ms">{e.ms} ms</span>
                    </div>
                    <code>{e.body}</code>
                    <code className="res">{e.result}</code>
                  </li>
                ))}
              </ul>
            </section>
          </div>
        </main>
      )}

      <input
        ref={fileRef}
        type="file"
        accept="image/jpeg,image/png,image/tiff,.tif,.tiff"
        hidden
        onChange={(e) => { handleFile(e.target.files[0]); e.target.value = ""; }}
      />
    </div>
  );
}
