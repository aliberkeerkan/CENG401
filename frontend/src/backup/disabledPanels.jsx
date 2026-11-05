/**
 * Parked on purpose. "Histogram of result" and "Requests to backend" were
 * taken off the page because they used the space the Compare view needs.
 * Nothing imports this file, so neither panel runs.
 *
 * To bring them back, in App.jsx:
 *   1. import { Histogram, RequestLog } from "./backup/disabledPanels.jsx";
 *   2. import onRequest again from "./api.js" (it still records every request);
 *   3. restore the state and the listener:
 *        const [log, setLog] = useState([]);
 *        useEffect(() => onRequest((e) => setLog((l) => [e, ...l].slice(0, 100))), []);
 *   4. put this back inside <div className="stage">, under the canvases:
 *        <div className="bottom">
 *          <Histogram image={shown} />
 *          <RequestLog log={log} backend={backend} />
 *        </div>
 * The .hist, .bottom and .log rules in styles.css were left in place.
 */
import React, { useMemo } from "react";
import { stats } from "../imageOps.js";

const fmt = (v, d = 2) => Number(v).toFixed(d);

export function Histogram({ image }) {
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

export function RequestLog({ log, backend }) {
  return (
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
  );
}
