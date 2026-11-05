// Compare view: the base image, then up to five tools (A–E) each applied to
// that same image on its own, so their results can be read side by side.
import React, { useEffect, useRef, useState } from "react";
import { processImage } from "./api.js";

const LETTERS = ["A", "B", "C", "D", "E"];

// opLabel marks a few symbols with $...$; the card summary is plain text
function readable(text) {
  return String(text).replace(/\$([^$]+)\$/g, (_, tex) =>
    tex.replace(/\\ge/g, "≥").replace(/\\le/g, "≤").replace(/[{}\\]/g, ""));
}

function CompareCard({
  panel, groups, imageId, ops, backend, baseImage,
  tools, ToolControls, opParams, opLabel, ImageCanvas, onTool, onParams, onRemove,
}) {
  const [result, setResult] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [settingsOpen, setSettingsOpen] = useState(true);
  const req = useRef(0);
  const spec = tools[panel.tool];
  const offline = spec?.backendOnly && backend !== "server";

  useEffect(() => {
    if (!imageId || !panel.tool || offline) {
      setResult(null);
      setBusy(false);
      return undefined;
    }
    const id = ++req.current;
    setBusy(true);
    setResult(null);
    const timer = setTimeout(async () => {
      try {
        const out = await processImage(imageId, [
          ...ops, { type: panel.tool, params: opParams(panel.tool, panel.params) },
        ]);
        if (id !== req.current) return;
        setResult(out);
        setError("");
      } catch (e) {
        if (id !== req.current) return;
        setResult(null);
        setError(e.message);
      } finally {
        if (id === req.current) setBusy(false);
      }
    }, 300);
    return () => clearTimeout(timer);
  }, [imageId, ops, panel.tool, panel.params, offline, opParams]);

  return (
    <article className="compare-card">
      <header className="compare-head">
        <span className="badge">{panel.letter}</span>
        <select aria-label={`Tool for ${panel.letter}`} value={panel.tool}
          onChange={(e) => onTool(panel.id, e.target.value)}>
          {groups.map((g) => (
            <optgroup key={g.id} label={g.label}>
              {g.tools.map((k) => (
                <option key={k} value={k}>{tools[k].label}</option>
              ))}
            </optgroup>
          ))}
        </select>
        <button type="button" className="icon-x" onClick={() => onRemove(panel.id)} aria-label={`Remove ${panel.letter}`}>×</button>
      </header>
      <p className="compare-summary">
        {readable(opLabel({ type: panel.tool, params: panel.params }))}
      </p>
      {offline ? (
        <p className="hint">This tool needs the backend.</p>
      ) : error ? (
        <p className="error" role="alert">{error}</p>
      ) : result ? (
        <ImageCanvas image={result} bare />
      ) : (
        <p className="hint compare-wait">{busy ? "Computing…" : ""}</p>
      )}
      <button type="button" className="settings-toggle" onClick={() => setSettingsOpen((o) => !o)}
        aria-expanded={settingsOpen}>
        {settingsOpen ? "Hide settings" : "Settings"}
      </button>
      {settingsOpen && (
        <ToolControls tool={panel.tool} params={panel.params} img={baseImage}
          setParams={(p) => onParams(panel.id, p)} />
      )}
    </article>
  );
}

export default function ComparePanel({
  imageId, baseImage, ops, backend,
  tools, categories, defaultParams, opParams, ToolControls, opLabel, ImageCanvas,
}) {
  const nextId = useRef(1);
  const groups = categories
    .map((c) => ({
      ...c,
      tools: c.tools.filter((k) => k !== "crop" && k !== "resize" && tools[k]
        && (backend === "server" || !tools[k].backendOnly)),
    }))
    .filter((c) => c.tools.length);

  const [panels, setPanels] = useState(() => [{
    id: 1, letter: "A", tool: "median", params: defaultParams("median", baseImage),
  }]);

  function add() {
    const used = new Set(panels.map((p) => p.letter));
    const letter = LETTERS.find((l) => !used.has(l));
    if (!letter) return;
    const id = ++nextId.current;
    setPanels((list) => [...list, {
      id, letter, tool: "median", params: defaultParams("median", baseImage),
    }]);
  }

  function remove(id) {
    setPanels((list) => list.filter((p) => p.id !== id));
  }

  function chooseTool(id, tool) {
    setPanels((list) => list.map((p) => (
      p.id === id ? { ...p, tool, params: defaultParams(tool, baseImage) } : p
    )));
  }

  function setParams(id, params) {
    setPanels((list) => list.map((p) => (p.id === id ? { ...p, params } : p)));
  }

  return (
    <section className="compare">
      <p className="hint">
        Each letter is one tool applied to the base image on its own, not on top of the others.
        Add up to five and compare them.
      </p>
      <div className="compare-grid">
        <article className="compare-card">
          <header className="compare-head">
            <span className="badge">Base</span>
            <strong>Base Image</strong>
          </header>
          <p className="compare-summary">
            {baseImage ? `${baseImage.width}×${baseImage.height}` : ""}
            {ops.length ? ` · ${ops.length} applied step${ops.length === 1 ? "" : "s"}` : " · no steps applied"}
          </p>
          <ImageCanvas image={baseImage} bare />
        </article>

        {panels.map((p) => (
          <CompareCard key={p.id} panel={p} groups={groups} imageId={imageId} ops={ops} backend={backend}
            baseImage={baseImage} tools={tools} ToolControls={ToolControls} opParams={opParams}
            opLabel={opLabel} ImageCanvas={ImageCanvas}
            onTool={chooseTool} onParams={setParams} onRemove={remove} />
        ))}

        {panels.length < LETTERS.length && (
          <button type="button" className="compare-add" onClick={add}>
            <span>+</span>
            Add comparison
          </button>
        )}
      </div>
    </section>
  );
}
