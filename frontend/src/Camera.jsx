// Camera view: the live picture of a chosen camera, processed on the backend
// frame by frame and shown in one window with the frame rate in its top-left
// corner. Only tools that keep up with a live picture are offered.
import React, { useEffect, useRef, useState } from "react";
import { processFrame } from "./api.js";

// Frames go to the backend at most this many pixels on the longest side:
// the filters take time in proportion to the pixels.
const MAX_SIDE = 640;
const JPEG_QUALITY = 0.8;
// Window filters stop at 7 x 7 here; larger windows drop below a live rate.
const WINDOW_LIMIT = 7;

// The tools the camera offers, by category.
export const CAMERA_GROUPS = [
  { id: "basic", label: "Basic tools", tools: ["grayscale", "brightness", "contrast", "gaussian_blur", "sharpen"] },
  {
    id: "noise", label: "Noise",
    tools: ["salt_pepper_noise", "quantization_noise", "uniform_noise", "rayleigh_noise",
      "exponential_noise", "gamma_noise", "poisson_noise"],
  },
  {
    id: "filters", label: "Filters",
    tools: ["arithmetic_mean", "geometric_mean", "harmonic_mean", "median", "minimum", "maximum", "midpoint"],
  },
];

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
const newSeed = () => Math.floor(Math.random() * 2 ** 32);

function cameraErrorMessage(e) {
  if (!navigator.mediaDevices?.getUserMedia) {
    return "The camera needs a secure page: open the app at http://localhost:5173 or over https.";
  }
  switch (e?.name) {
    case "NotAllowedError": return "Camera access was blocked. Allow it for this page and try again.";
    case "NotFoundError":
    case "OverconstrainedError": return "That camera was not found. Choose another one.";
    case "NotReadableError": return "The camera is in use by another program.";
    default: return `The camera could not be started${e?.message ? `: ${e.message}` : "."}`;
  }
}

// Draws the current video frame on `canvas`, scaled to at most `maxSide`.
// Returns false until the video has a picture.
function drawVideoFrame(video, canvas, maxSide) {
  if (!video || video.readyState < 2 || !video.videoWidth) return false;
  const scale = Math.min(1, maxSide / Math.max(video.videoWidth, video.videoHeight));
  const w = Math.max(1, Math.round(video.videoWidth * scale));
  const h = Math.max(1, Math.round(video.videoHeight * scale));
  if (canvas.width !== w) canvas.width = w;
  if (canvas.height !== h) canvas.height = h;
  canvas.getContext("2d").drawImage(video, 0, 0, w, h);
  return true;
}

const canvasToJpeg = (canvas) => new Promise((resolve) => canvas.toBlob(resolve, "image/jpeg", JPEG_QUALITY));

// Random noise draws a new pattern on every frame, like a real sensor.
const freshSeeds = (ops) => ops.map((op) => (
  op.params && "seed" in op.params ? { ...op, params: { ...op.params, seed: newSeed() } } : op
));

export default function CameraPanel({
  tools, defaultParams, opParams, isNeutral, opLabel, ToolControls, Formula, TexBits, backend,
}) {
  const [devices, setDevices] = useState([]);
  const [deviceId, setDeviceId] = useState("");
  const [running, setRunning] = useState(false);
  const [starting, setStarting] = useState(false);
  const [error, setError] = useState("");
  const [fps, setFps] = useState(null);
  const [frameSize, setFrameSize] = useState(null);
  // the applied chain, plus the tool being adjusted on top of it
  const [chain, setChain] = useState([]);
  const [group, setGroup] = useState("basic");
  const [tool, setTool] = useState(null);
  const [params, setParams] = useState({});

  const videoRef = useRef(null);
  const viewRef = useRef(null);
  const grabRef = useRef(null);
  const streamRef = useRef(null);
  // the loop reads the newest chain and sliders from here instead of restarting
  const live = useRef({});
  const size = frameSize || { width: 1, height: 1 };
  live.current = { chain, tool, params, neutral: !tool || isNeutral(tool, params, size) };

  // camera list; names appear once the page has camera permission
  async function refreshDevices() {
    if (!navigator.mediaDevices?.enumerateDevices) return;
    const cams = (await navigator.mediaDevices.enumerateDevices()).filter((d) => d.kind === "videoinput");
    setDevices(cams);
    setDeviceId((id) => (id && cams.some((c) => c.deviceId === id) ? id : cams[0]?.deviceId || ""));
  }

  useEffect(() => {
    refreshDevices();
    const md = navigator.mediaDevices;
    md?.addEventListener?.("devicechange", refreshDevices);
    return () => {
      md?.removeEventListener?.("devicechange", refreshDevices);
      streamRef.current?.getTracks().forEach((t) => t.stop());
    };
  }, []);

  function stop() {
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
    if (videoRef.current) videoRef.current.srcObject = null;
    setRunning(false);
    setFps(null);
  }

  async function start(id = deviceId) {
    setError("");
    if (backend !== "server") {
      setError("The camera needs the backend. Start it and reload the page.");
      return;
    }
    stop();
    setStarting(true);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { ...(id ? { deviceId: { exact: id } } : {}), width: { ideal: 640 }, height: { ideal: 480 } },
        audio: false,
      });
      streamRef.current = stream;
      const video = videoRef.current;
      video.srcObject = stream;
      await video.play();
      // with permission granted the cameras now have names
      await refreshDevices();
      const used = stream.getVideoTracks()[0]?.getSettings().deviceId;
      if (used) setDeviceId(used);
      setRunning(true);
    } catch (e) {
      stop();
      setError(cameraErrorMessage(e));
    } finally {
      setStarting(false);
    }
  }

  // The loop: grab a frame, send it with the chain, draw the result, then
  // grab the next one. Frames are never queued, so the picture stays current
  // even when a filter is slow; the frame rate drops instead.
  useEffect(() => {
    if (!running) return undefined;
    let alive = true;
    (async () => {
      const grab = grabRef.current || (grabRef.current = document.createElement("canvas"));
      const shown = [];  // display times of the last second, for the frame rate
      let lastFpsUpdate = 0;
      let failing = false;
      while (alive) {
        if (document.hidden) { await sleep(300); continue; }
        try {
          if (!drawVideoFrame(videoRef.current, grab, MAX_SIDE)) { await sleep(30); continue; }
          const s = live.current;
          const ops = s.neutral ? s.chain : [...s.chain, { type: s.tool, params: opParams(s.tool, s.params) }];
          const out = await processFrame(await canvasToJpeg(grab), freshSeeds(ops), { fast: true });
          if (!alive) break;
          const view = viewRef.current;
          if (view) {
            if (view.width !== out.width) view.width = out.width;
            if (view.height !== out.height) view.height = out.height;
            view.getContext("2d").putImageData(out, 0, 0);
          }
          const now = performance.now();
          shown.push(now);
          while (shown.length && now - shown[0] > 1000) shown.shift();
          if (now - lastFpsUpdate > 250) {
            lastFpsUpdate = now;
            setFps(shown.length > 1 ? (shown.length - 1) * 1000 / (now - shown[0]) : null);
            setFrameSize((f) => (f && f.width === out.width && f.height === out.height ? f : { width: out.width, height: out.height }));
          }
          if (failing) { failing = false; setError(""); }
        } catch (e) {
          if (!alive) break;
          failing = true;
          setError(e.message);
          await sleep(800);  // do not hammer the backend with a request that keeps failing
        }
      }
    })();
    return () => { alive = false; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [running]);

  function chooseTool(t) {
    setTool(t);
    setParams(defaultParams(t, null));
  }

  function addToChain() {
    if (live.current.neutral) return;
    setChain((c) => [...c, { type: tool, params: opParams(tool, params) }]);
    setParams(defaultParams(tool, null));
  }

  const g = CAMERA_GROUPS.find((x) => x.id === group);
  const t = tool ? tools[tool] : null;
  const limits = { size: WINDOW_LIMIT };

  return (
    <div className="camera-page">
      <aside className="panel camera-tools" aria-label="Camera and tools">
        <h3>Camera</h3>
        <label className="select">
          <span>Device</span>
          <select value={deviceId} disabled={starting || !devices.length}
            onChange={(e) => { setDeviceId(e.target.value); if (running) start(e.target.value); }}>
            {/* the browser lists the cameras only after camera permission */}
            {!devices.length && <option value="">Default camera</option>}
            {devices.map((d, i) => <option key={d.deviceId || i} value={d.deviceId}>{d.label || `Camera ${i + 1}`}</option>)}
          </select>
        </label>
        <div className="row">
          {running
            ? <button onClick={stop}>Stop camera</button>
            : <button className="primary" onClick={() => start()} disabled={starting || backend !== "server"}>
                {starting ? "Starting…" : "Start camera"}
              </button>}
        </div>
        {backend !== "server" && <p className="hint">The camera needs the backend.</p>}
        {error && <p className="error" role="alert">{error}</p>}

        <h3 className="gap">Tools</h3>
        <div className="camera-groups" role="tablist" aria-label="Tool groups">
          {CAMERA_GROUPS.map((x) => (
            <button key={x.id} role="tab" aria-selected={group === x.id}
              className={"tool" + (group === x.id ? " on" : "")} onClick={() => setGroup(x.id)}>{x.label}</button>
          ))}
        </div>
        <div className="tool-list" role="tablist" aria-label={g.label}>
          {g.tools.map((k) => (
            <button key={k} role="tab" aria-selected={tool === k}
              className={"tool" + (tool === k ? " on" : "")} onClick={() => chooseTool(k)}>{tools[k].label}</button>
          ))}
        </div>

        {t ? (
          <>
            <div className="formula">
              <Formula tool={tool} params={params} img={null} />
              {t.eq && <span className="eqno">({t.eq})</span>}
            </div>
            <p className="hint"><TexBits text={t.describe} /></p>
            <ToolControls tool={tool} params={params} setParams={setParams} img={null} limits={limits} camera />
            <p className="hint">The camera shows this tool live. Add it to keep it while you try another one.</p>
            <div className="row">
              <button className="primary" onClick={addToChain} disabled={live.current.neutral}>Add to chain</button>
              <button onClick={() => setTool(null)}>Close tool</button>
            </div>
          </>
        ) : (
          <p className="hint">Choose a tool to see its effect on the camera picture.</p>
        )}

        <h3 className="gap">Chain</h3>
        {chain.length === 0 ? (
          <p className="hint">No steps yet. Added tools run on every frame, in this order.</p>
        ) : (
          <ol className="history camera-chain">
            {chain.map((op, i) => (
              <li key={i}>
                <span><TexBits text={opLabel(op)} /></span>
                <button type="button" className="icon-btn" aria-label={`Remove step ${i + 1}`} title="Remove"
                  onClick={() => setChain((c) => c.filter((_, j) => j !== i))}>×</button>
              </li>
            ))}
          </ol>
        )}
        <div className="row">
          <button onClick={() => setChain([])} disabled={!chain.length}>Clear chain</button>
        </div>
      </aside>

      <section className="camera-stage">
        <figure className="frame camera-frame">
          <div className="camera-view">
            {/* the wrapper hugs the picture, so the frame rate sits on its corner */}
            <div className="camera-picture" hidden={!running}>
              <canvas ref={viewRef} aria-label="Processed camera picture" />
              <span className="fps" aria-live="off">{fps === null ? "– FPS" : `${fps.toFixed(1)} FPS`}</span>
            </div>
            {!running && (
              <div className="camera-empty">
                <p>Choose a camera and press Start camera.</p>
                <p className="hint">Frames are processed at up to {MAX_SIDE} px on the backend; the frame rate is the number of processed frames shown per second.</p>
              </div>
            )}
          </div>
        </figure>
      </section>

      {/* the camera feed; drawn from here onto a canvas, never shown itself */}
      <video ref={videoRef} muted playsInline aria-hidden="true"
        style={{ position: "fixed", width: 1, height: 1, opacity: 0, pointerEvents: "none" }} />
    </div>
  );
}
