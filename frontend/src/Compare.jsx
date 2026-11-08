// Compare view: the base image, then up to five tools (1–5) each applied to
// that same image on its own, so their results can be read side by side.
// The card with the open eye is also shown large above the grid.
import React, { useCallback, useEffect, useRef, useState } from "react";
import { processImage } from "./api.js";

const NUMBERS = [1, 2, 3, 4, 5];

// opLabel marks a few symbols with $...$; the card summary is plain text
function readable(text) {
  return String(text).replace(/\$([^$]+)\$/g, (_, tex) =>
    tex.replace(/\\ge/g, "≥").replace(/\\le/g, "≤").replace(/[{}\\]/g, ""));
}

function EyeIcon({ open }) {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true" fill="none" stroke="currentColor"
      strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
      <path d="M2 12s3.5-6 10-6 10 6 10 6-3.5 6-10 6S2 12 2 12z" />
      <circle cx="12" cy="12" r="2.6" />
      {!open && <path d="M4 5l16 14" />}
    </svg>
  );
}

function EyeButton({ open, label, onClick, stays }) {
  const pressed = stays
    ? `${label} is shown above`
    : `${label} is shown above. Show the base image instead`;
  return (
    <button type="button" className={"eye" + (open ? " on" : "")} onClick={onClick}
      aria-pressed={open}
      aria-label={open ? pressed : `Show ${label} above`}
      title={open ? "Shown above" : "Show above"}>
      <EyeIcon open={open} />
    </button>
  );
}

function ToolPicker({ n, value, groups, tools, onChange }) {
  const [open, setOpen] = useState(false);
  const activeGroup = groups.find((g) => g.tools.includes(value))?.id;
  const [openGroups, setOpenGroups] = useState(() => new Set(activeGroup ? [activeGroup] : []));
  const rootRef = useRef(null);
  const label = tools[value]?.label || "Choose a tool";

  useEffect(() => {
    if (!open) return undefined;
    function onDoc(e) {
      if (!rootRef.current?.contains(e.target)) setOpen(false);
    }
    function onKey(e) {
      if (e.key === "Escape") setOpen(false);
    }
    document.addEventListener("pointerdown", onDoc);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("pointerdown", onDoc);
      document.removeEventListener("keydown", onKey);
    };
  }, [open]);

  function togglePicker() {
    setOpen((v) => {
      if (!v && activeGroup) {
        setOpenGroups((prev) => (prev.has(activeGroup) ? prev : new Set(prev).add(activeGroup)));
      }
      return !v;
    });
  }

  function toggleGroup(id) {
    setOpenGroups((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }

  return (
    <div className={"tool-pick" + (open ? " open" : "")} ref={rootRef}>
      <button type="button" className="tool-pick-btn" aria-expanded={open} aria-label={`Tool for ${n}`}
        onClick={togglePicker}>
        <span className="tool-pick-label">{label}</span>
        <span className={"tool-mark" + (open ? " on" : "")} aria-hidden="true" />
      </button>
      {open && (
        <div className="tool-pick-panel">
          {groups.map((g) => {
            const expanded = openGroups.has(g.id);
            return (
              <div key={g.id} className="tool-pick-group">
                <button type="button" className="tool-pick-group-btn" aria-expanded={expanded}
                  onClick={() => toggleGroup(g.id)}>
                  <span className={"tool-mark" + (expanded ? " on" : "")} aria-hidden="true" />
                  <span>{g.label}</span>
                </button>
                {expanded && (
                  <div className="tool-pick-tools" role="listbox" aria-label={g.label}>
                    {g.tools.map((k) => (
                      <button key={k} type="button" role="option" aria-selected={k === value}
                        className={"tool-pick-opt" + (k === value ? " on" : "")}
                        onClick={() => { onChange(k); setOpen(false); }}>
                        {tools[k].label}
                      </button>
                    ))}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}

function CompareCard({
  panel, groups, imageId, ops, backend, baseImage, watched,
  tools, ToolControls, opParams, opLabel, ImageCanvas, onTool, onParams, onRemove, onShow, onImage,
}) {
  const [result, setResult] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [settingsOpen, setSettingsOpen] = useState(true);
  const req = useRef(0);
  const spec = tools[panel.tool];
  const offline = spec?.backendOnly && backend !== "server";
  const summary = readable(opLabel({ type: panel.tool, params: panel.params }));

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

  useEffect(() => {
    onImage(panel.id, {
      image: offline || error ? null : result,
      detail: offline ? "This tool needs the backend." : error || (busy || !result ? "Computing…" : `${result.width}×${result.height}`),
    });
  }, [panel.id, result, error, busy, offline, onImage]);

  return (
    <article className={"compare-card" + (watched ? " watched" : "")}>
      <header className="compare-head">
        <span className="badge">{panel.n}</span>
        <ToolPicker n={panel.n} value={panel.tool} groups={groups} tools={tools}
          onChange={(tool) => onTool(panel.id, tool)} />
        <button type="button" className="icon-x" onClick={() => onRemove(panel.id)} aria-label={`Remove ${panel.n}`}>×</button>
      </header>
      <p className="compare-summary">{summary}</p>
      <div className="compare-shot">
        {offline ? (
          <p className="hint">This tool needs the backend.</p>
        ) : error ? (
          <p className="error" role="alert">{error}</p>
        ) : result ? (
          <ImageCanvas image={result} bare />
        ) : (
          <p className="hint compare-wait">{busy ? "Computing…" : ""}</p>
        )}
        <EyeButton open={watched} label={`${panel.n}, ${summary}`} onClick={() => onShow(panel.id)} />
      </div>
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
  tools, categories, defaultParams, opParams, ToolControls, opLabel, ImageCanvas, onFeature,
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
    id: 1, n: 1, tool: "median", params: defaultParams("median", baseImage),
  }]);
  const [watch, setWatch] = useState("base");
  const [shots, setShots] = useState({});

  const remember = useCallback((id, shot) => {
    setShots((prev) => (
      prev[id] && prev[id].image === shot.image && prev[id].detail === shot.detail ? prev : { ...prev, [id]: shot }
    ));
  }, []);

  function add() {
    const used = new Set(panels.map((p) => p.n));
    const n = NUMBERS.find((k) => !used.has(k));
    if (!n) return;
    const id = ++nextId.current;
    setPanels((list) => [...list, {
      id, n, tool: "median", params: defaultParams("median", baseImage),
    }]);
  }

  function remove(id) {
    setPanels((list) => list.filter((p) => p.id !== id));
    setShots((prev) => {
      if (!(id in prev)) return prev;
      const next = { ...prev };
      delete next[id];
      return next;
    });
    setWatch((cur) => (cur === id ? "base" : cur));
  }

  function show(id) {
    setWatch((cur) => (cur === id ? "base" : id));
  }

  function chooseTool(id, tool) {
    setPanels((list) => list.map((p) => (
      p.id === id ? { ...p, tool, params: defaultParams(tool, baseImage) } : p
    )));
  }

  function setParams(id, params) {
    setPanels((list) => list.map((p) => (p.id === id ? { ...p, params } : p)));
  }

  const baseDetail = (baseImage ? `${baseImage.width}×${baseImage.height}` : "")
    + (ops.length ? ` · ${ops.length} applied step${ops.length === 1 ? "" : "s"}` : " · no steps applied");

  useEffect(() => {
    if (watch !== "base" && !panels.some((p) => p.id === watch)) {
      setWatch("base");
      return;
    }
    if (watch === "base") {
      onFeature({ title: "Base Image", detail: baseDetail, image: baseImage });
      return;
    }
    const panel = panels.find((p) => p.id === watch);
    const shot = shots[watch];
    const summary = panel ? readable(opLabel({ type: panel.tool, params: panel.params })) : "";
    onFeature({
      title: panel ? `${panel.n} · ${summary}` : "Base Image",
      detail: shot?.detail || "Computing…",
      image: shot?.image || null,
    });
  }, [watch, panels, shots, baseImage, baseDetail, onFeature, opLabel]);

  const ordered = [...panels].sort((a, b) => a.n - b.n);

  return (
    <section className="compare">
      <p className="hint">
        Each number is one tool applied to the base image on its own, not on top of the others.
        The open eye is shown large above. Add up to five and compare them.
      </p>
      <div className="compare-grid">
        <article className={"compare-card" + (watch === "base" ? " watched" : "")}>
          <header className="compare-head">
            <span className="badge">Base</span>
            <strong>Base Image</strong>
          </header>
          <p className="compare-summary">{baseDetail}</p>
          <div className="compare-shot">
            <ImageCanvas image={baseImage} bare />
            <EyeButton open={watch === "base"} stays label="Base Image" onClick={() => setWatch("base")} />
          </div>
        </article>

        {ordered.map((p) => (
          <CompareCard key={p.id} panel={p} groups={groups} imageId={imageId} ops={ops} backend={backend}
            baseImage={baseImage} watched={watch === p.id} tools={tools} ToolControls={ToolControls}
            opParams={opParams} opLabel={opLabel} ImageCanvas={ImageCanvas}
            onTool={chooseTool} onParams={setParams} onRemove={remove} onShow={show} onImage={remember} />
        ))}

        {panels.length < NUMBERS.length && (
          <button type="button" className="compare-add" onClick={add}>
            <span>+</span>
            Add comparison
          </button>
        )}
      </div>
    </section>
  );
}
