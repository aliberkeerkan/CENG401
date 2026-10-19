import React, { useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import katex from "katex";
import "katex/dist/katex.min.css";
import {
  upload, uploadSample, uploadFromUrl, processImage, fetchOriginal, exportImage, saveBlob, EXPORT_FORMATS,
  onRequest, onStatus, checkBackend, API_URL,
} from "./api.js";
import { BUILTIN_SAMPLES } from "./builtinSamples.js";
import { applyOp, stats, sampleImage, gaussianKernel, rgbRange } from "./imageOps.js";

// Tool definitions. `heavy` tools get a short delay before the preview is
// computed so dragging a slider stays smooth. `backendOnly` tools are not
// implemented in imageOps.js: their preview comes from the backend via
// processImage, and they are disabled while the backend is offline.
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
    sliders: [{ key: "beta", min: -100, max: 100, step: 1, name: "β (offset)" }],
    describe: "Adds the same offset to every pixel. Values past 0 or 255 are clipped.",
  },
  contrast: {
    label: "Contrast", eq: "9",
    sliders: [
      { key: "r_min", min: 0, max: 254, step: 1, name: "r_min (maps to 0)" },
      { key: "r_max", min: 1, max: 255, step: 1, name: "r_max (maps to 255)" },
    ],
    describe: "Maps the input range [r_min, r_max] linearly onto [0, 255]. The sliders start at the image's darkest and brightest values, so applying stretches it to the full range. Values outside the range are clipped.",
  },
  gaussian_blur: {
    label: "Gaussian blur", eq: "10", heavy: true,
    sliders: [{ key: "sigma", min: 0.5, max: 10, step: 0.1, name: "σ (blur scale)" }],
    describe: "Averages each pixel with its neighbours using Gaussian weights. Larger σ blurs more.",
  },
  median: {
    label: "Median filter", eq: "33", heavy: true,
    sliders: [{ key: "size", min: 3, max: 15, step: 2, name: "Window size" }],
    describe: "Replaces each pixel with the median of its window. Removes salt-and-pepper noise and keeps edges.",
  },
  arithmetic_mean: {
    label: "Mean filter", eq: "27", heavy: true, backendOnly: true,
    sliders: [{ key: "size", min: 3, max: 15, step: 2, name: "Window size" }],
    describe: "Softens additive noise such as Gaussian or uniform noise, but blurs the edges too.",
  },
  geometric_mean: {
    label: "Geometric mean", eq: "28", heavy: true, backendOnly: true,
    sliders: [{ key: "size", min: 3, max: 15, step: 2, name: "Window size" }],
    describe: "Like the mean filter, but preserves detail slightly better. A single 0 (pepper) pixel in the window darkens the result.",
  },
  harmonic_mean: {
    label: "Harmonic mean", eq: "29", heavy: true, backendOnly: true,
    sliders: [{ key: "size", min: 3, max: 15, step: 2, name: "Window size" }],
    describe: "Cleans salt noise well. With pepper noise the window is pulled towards zero and the image darkens.",
  },
  minimum: {
    label: "Min filter", eq: "31", heavy: true, backendOnly: true,
    sliders: [{ key: "size", min: 3, max: 15, step: 2, name: "Window size" }],
    describe: "Takes the darkest pixel of each window. Suppresses isolated bright (salt) impulses; the image darkens overall.",
  },
  maximum: {
    label: "Max filter", eq: "32", heavy: true, backendOnly: true,
    sliders: [{ key: "size", min: 3, max: 15, step: 2, name: "Window size" }],
    describe: "Takes the brightest pixel of each window. Suppresses isolated dark (pepper) impulses; the image brightens overall.",
  },
  midpoint: {
    label: "Midpoint filter", eq: "34", heavy: true, backendOnly: true,
    sliders: [{ key: "size", min: 3, max: 15, step: 2, name: "Window size" }],
    describe: "Averages the darkest and brightest pixel of each window. Best suited to bounded random noise such as uniform noise, not salt & pepper.",
  },
  salt_pepper_noise: {
    label: "Salt & pepper", eq: "18", heavy: true, backendOnly: true,
    sliders: [
      { key: "p_pepper", min: 0, max: 0.5, step: 0.01, name: "Pₚ (pepper)" },
      { key: "p_salt", min: 0, max: 0.5, step: 0.01, name: "Pₛ (salt)" },
    ],
    describe: "Sets random pixels to black (pepper) or white (salt). Other pixels keep their value.",
  },
  quantization_noise: {
    label: "Quantization noise", eq: "22", backendOnly: true,
    sliders: [{ key: "bits", min: 1, max: 8, step: 1, name: "Bit depth B" }],
    describe: "Not random: reduces the number of gray levels to 2^B, so low B shows banding. Filters can soften banding but cannot restore the lost levels.",
  },
  rayleigh_noise: {
    label: "Rayleigh noise", eq: "24", heavy: true, backendOnly: true,
    sliders: [
      { key: "a", min: -100, max: 100, step: 1, name: "a (location)" },
      { key: "b", min: 1, max: 2000, step: 10, name: "b (spread, = 2σ²)" },
    ],
    describe: "Its mean is positive, so it brightens the image. The zero-mean option removes that shift.",
  },
  uniform_noise: {
    label: "Uniform noise", eq: "25", heavy: true, backendOnly: true,
    sliders: [
      { key: "a", min: -100, max: 100, step: 1, name: "a (lower bound)" },
      { key: "b", min: -100, max: 100, step: 1, name: "b (upper bound)" },
    ],
    describe: "Adds noise drawn uniformly from [a, b]. A range symmetric around 0 adds no brightness shift.",
  },
  band_reject: {
    label: "Band-reject filter", eq: "40", heavy: true, backendOnly: true,
    sliders: [
      { key: "c0", min: 0.04, max: 0.80, step: 0.01, name: "D₀ (band centre / Dmax)" },
      { key: "width", min: 0.02, max: 0.50, step: 0.01, name: "W (band width / Dmax)" },
      { key: "n", min: 1, max: 8, step: 1, name: "n (Butterworth order)" },
    ],
    describe: "Removes a ring of frequencies of radius D₀ and width W from the spectrum. Ideal has a hard edge and rings; Butterworth and Gaussian (Gonzalez–Woods Table 4.7) are smoother.",
  },
  band_pass: {
    label: "Band-pass filter", eq: "41", heavy: true, backendOnly: true,
    sliders: [
      { key: "c0", min: 0.04, max: 0.80, step: 0.01, name: "D₀ (band centre / Dmax)" },
      { key: "width", min: 0.02, max: 0.50, step: 0.01, name: "W (band width / Dmax)" },
      { key: "n", min: 1, max: 8, step: 1, name: "n (Butterworth order)" },
    ],
    describe: "Keeps only the ring that the band-reject filter removes, to show what it takes away, e.g. a periodic pattern. Shown around mid-gray (128 = 0), since the result has no mean.",
  },
  wiener: {
    label: "Wiener filter", eq: "44", heavy: true, backendOnly: true,
    sliders: [{ key: "noise_std", min: 0, max: 60, step: 1, name: "σ_η (noise std)" }],
    describe: "Keeps each frequency in proportion to how much of it is signal rather than noise. Set σ_η to the noise level; the uniform and Rayleigh tools show the std of the noise they add.",
  },
  kalman: {
    label: "Kalman filter", eq: "45–51", heavy: true, backendOnly: true,
    sliders: [
      { key: "q", min: 0.5, max: 40, step: 0.5, name: "q (process variance)" },
      { key: "r", min: 1, max: 800, step: 1, name: "r (measurement variance)" },
    ],
    describe: "Not a Gonzalez–Woods formula: a 1-D Kalman smoother. Every row, then every column, is a 1-D signal; R, G and B are processed separately. A larger r/q smooths more; a constant image is unchanged.",
  },
};

// The tools panel first lists these groups (the plan's feature groups,
// Table 1); choosing one shows its tools.
const CATEGORIES = [
  {
    id: "basic", label: "Basic tools",
    tools: ["crop", "resize", "grayscale", "brightness", "contrast", "gaussian_blur"],
  },
  {
    id: "noise", label: "Noise",
    tools: ["salt_pepper_noise", "quantization_noise", "rayleigh_noise", "uniform_noise"],
  },
  {
    id: "filters", label: "Filters",
    tools: [
      "arithmetic_mean", "geometric_mean", "harmonic_mean",
      "median", "minimum", "maximum", "midpoint",
    ],
  },
  {
    id: "frequency", label: "Frequency domain",
    tools: ["band_reject", "band_pass", "wiener", "kalman"],
  },
];

// 20 x 20 line icons drawn with currentColor
const svgProps = {
  width: 20, height: 20, viewBox: "0 0 20 20", fill: "none", stroke: "currentColor",
  strokeWidth: 1.6, strokeLinecap: "round", strokeLinejoin: "round", "aria-hidden": true,
};
const ICONS = {
  // three sliders
  basic: (
    <svg {...svgProps}>
      <path d="M3 5h14M3 10h14M3 15h14" />
      <circle cx="7" cy="5" r="1.8" fill="var(--panel)" />
      <circle cx="13" cy="10" r="1.8" fill="var(--panel)" />
      <circle cx="9" cy="15" r="1.8" fill="var(--panel)" />
    </svg>
  ),
  // scattered pixels
  noise: (
    <svg {...svgProps}>
      <rect x="3" y="3" width="14" height="14" rx="2.5" />
      <path d="M7 7h.01M12 6.5h.01M9.5 10h.01M6.5 13h.01M13 13.5h.01M14 9.5h.01" strokeWidth="2.4" />
    </svg>
  ),
  // funnel
  filters: (
    <svg {...svgProps}>
      <path d="M3.5 4h13l-5 6.2V16l-3-1.5v-4.3z" />
    </svg>
  ),
  // a sine wave
  frequency: (
    <svg {...svgProps}>
      <path d="M2.5 10c1.6-5 3.4-5 5 0s3.4 5 5 0 3.4-5 5 0" />
    </svg>
  ),
  menu: (
    <svg {...svgProps}>
      <path d="M3.5 5.5h13M3.5 10h13M3.5 14.5h13" />
    </svg>
  ),
  back: (
    <svg {...svgProps}>
      <path d="M16 10H4.5M9 5l-5 5 5 5" />
    </svg>
  ),
  chevron: (
    <svg {...svgProps} width={16} height={16}>
      <path d="M8 5l5 5-5 5" />
    </svg>
  ),
};

// tools not implemented in imageOps.js: preview comes from the backend and
// the tool is disabled while the backend is offline
const BACKEND_ONLY = new Set(
  Object.entries(TOOLS).filter(([, v]) => v.backendOnly).map(([k]) => k)
);

// noise tools whose params carry a seed
const NOISE_TOOLS = new Set(["salt_pepper_noise", "rayleigh_noise", "uniform_noise"]);

const newSeed = () => Math.floor(Math.random() * 2 ** 32);

// The working session (image id and applied steps) is kept in this tab's
// sessionStorage. When the browser reloads the tab, for example after
// Edge put it to sleep, the image is fetched from the backend again and
// the steps are re-applied. Storage can be unavailable; then nothing is
// saved and the app works as before.
const SESSION_KEY = "imagelab-session";

function readSession() {
  try {
    return JSON.parse(sessionStorage.getItem(SESSION_KEY) || "null");
  } catch {
    return null;
  }
}

function writeSession(value) {
  try {
    if (value) sessionStorage.setItem(SESSION_KEY, JSON.stringify(value));
    else sessionStorage.removeItem(SESSION_KEY);
  } catch { /* storage unavailable */ }
}

// tools that have no "do nothing" value: show their default effect on open
const PREVIEW_ON_OPEN = new Set([
  "gaussian_blur", "median",
  "arithmetic_mean", "geometric_mean", "harmonic_mean",
  "minimum", "maximum", "midpoint",
  "salt_pepper_noise", "quantization_noise", "rayleigh_noise", "uniform_noise",
  "band_reject", "band_pass", "wiener", "kalman",
]);

function defaultParams(tool, img) {
  const w = img ? img.width : 1, h = img ? img.height : 1;
  switch (tool) {
    case "crop":
      return { x: Math.round(w * 0.1), y: Math.round(h * 0.1), width: Math.max(1, Math.round(w * 0.8)), height: Math.max(1, Math.round(h * 0.8)) };
    case "resize": return { width: w, height: h, method: "bilinear", lock: true };
    case "brightness": return { beta: 0 };
    // start at the image's own range: applying stretches it to [0, 255]
    case "contrast": {
      const [r_min, r_max] = img ? rgbRange(img) : [0, 255];
      return { r_min, r_max };
    }
    case "gaussian_blur": return { sigma: 1.5 };
    case "median": return { size: 3 };
    case "arithmetic_mean":
    case "geometric_mean":
    case "harmonic_mean":
    case "minimum":
    case "maximum":
    case "midpoint": return { size: 3 };
    // every noise step gets its own random seed when the tool is selected;
    // the seed stays fixed while the user adjusts the other parameters
    case "salt_pepper_noise": return { p_pepper: 0.05, p_salt: 0.05, seed: newSeed() };
    case "quantization_noise": return { bits: 4 };
    case "rayleigh_noise": return { a: 0, b: 400, zero_mean: false, seed: newSeed() };
    case "uniform_noise": return { a: -20, b: 20, seed: newSeed() };
    case "band_reject":
    case "band_pass": return { c0: 0.25, width: 0.12, kind: "butterworth", n: 2 };
    case "wiener": return { noise_std: 20 };
    case "kalman": return { q: 4, r: 80 };
    default: return {};
  }
}

function isNeutral(tool, p, img) {
  if (!img) return true;
  if (tool === "brightness") return p.beta === 0;
  if (tool === "contrast") return p.r_min === 0 && p.r_max === 255;
  if (tool === "resize") return p.width === img.width && p.height === img.height;
  if (tool === "crop") return p.x === 0 && p.y === 0 && p.width === img.width && p.height === img.height;
  if (tool === "salt_pepper_noise") return p.p_pepper === 0 && p.p_salt === 0;
  if (tool === "quantization_noise") return p.bits === 8;
  if (tool === "wiener") return p.noise_std === 0;
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
  const ref = useRef(null);
  const html = useMemo(
    () => katex.renderToString(tex, { throwOnError: false, displayMode: display, strict: "ignore" }),
    [tex, display],
  );
  // The tools panel is narrow: shrink a long formula until it fits,
  // down to 60% of the normal size, before the browser paints it.
  useLayoutEffect(() => {
    const el = ref.current;
    if (!el) return;
    el.style.fontSize = "";
    const scale = el.clientWidth / el.scrollWidth;
    if (scale < 1) el.style.fontSize = `${Math.max(0.6, scale * 0.98)}em`;
  }, [html]);
  return <span ref={ref} className="math" dangerouslySetInnerHTML={{ __html: html }} />;
}

// The plan's equation for each tool, with the current parameter values.
function formulaTex(tool, params, img) {
  const W = img ? img.width : 0, H = img ? img.height : 0;
  const win = `${params.size}\\times${params.size}`;
  switch (tool) {
    case "crop":
      return `g(x,y)=f(x+${params.x},\\,y+${params.y}),\\quad 0\\le x<${params.width},\\; 0\\le y<${params.height}`;
    case "resize":
      return params.method === "nearest"
        ? `\\begin{aligned} g(x',y')&=f(\\mathrm{round}(x),\\mathrm{round}(y))\\\\ x&=(x'+\\tfrac12)\\cdot\\frac{${W}}{${params.width}}-\\tfrac12 \\end{aligned}`
        : `\\begin{aligned} x&=(x'+\\tfrac12)\\cdot\\frac{${W}}{${params.width}}-\\tfrac12\\\\ y&=(y'+\\tfrac12)\\cdot\\frac{${H}}{${params.height}}-\\tfrac12 \\end{aligned}`;
    case "grayscale":
      return `Y=0.299R+0.587G+0.114B`;
    case "brightness": {
      const sign = params.beta < 0 ? "-" : "+";
      return `g(x,y)=\\mathrm{clip}\\bigl(f(x,y)${sign}${Math.abs(params.beta)}\\bigr)`;
    }
    // the output range is the full scale: s_min = 0, s_max = 255
    case "contrast":
      return `\\begin{aligned} s=T(r)&=\\frac{s_{\\max}-s_{\\min}}{r_{\\max}-r_{\\min}}\\cdot(r-r_{\\min})+s_{\\min}\\\\ r_{\\min}&=${params.r_min},\\; r_{\\max}=${params.r_max},\\; s_{\\min}=0,\\; s_{\\max}=255 \\end{aligned}`;
    case "gaussian_blur": {
      const n = gaussianKernel(params.sigma).length;
      return `\\begin{aligned} w(s,t)&\\propto\\exp\\!\\left(-\\frac{s^{2}+t^{2}}{2\\sigma^{2}}\\right)\\\\ \\sigma&=${fmt(params.sigma, 1)},\\;\\text{window }${n}\\times${n} \\end{aligned}`;
    }
    case "median":
      return `\\hat{f}(x,y)=\\mathrm{median}\\{g(s,t)\\},\\; S_{xy}=${win}`;
    case "arithmetic_mean":
      return `\\begin{aligned} \\hat{f}(x,y)&=\\frac{1}{mn}\\sum_{(s,t)\\in S_{xy}} g(s,t)\\\\ S_{xy}&=${win} \\end{aligned}`;
    case "geometric_mean":
      return `\\begin{aligned} \\hat{f}(x,y)&=\\exp\\!\\Bigl(\\frac{1}{mn}\\sum_{(s,t)\\in S_{xy}}\\ln\\bigl(g(s,t)+\\varepsilon\\bigr)\\Bigr)\\\\ S_{xy}&=${win} \\end{aligned}`;
    case "harmonic_mean":
      return `\\begin{aligned} \\hat{f}(x,y)&=\\frac{mn}{\\sum_{(s,t)\\in S_{xy}} 1/\\bigl(g(s,t)+\\varepsilon\\bigr)}\\\\ S_{xy}&=${win} \\end{aligned}`;
    case "minimum":
      return `\\hat{f}(x,y)=\\min_{(s,t)\\in S_{xy}} g(s,t),\\; S_{xy}=${win}`;
    case "maximum":
      return `\\hat{f}(x,y)=\\max_{(s,t)\\in S_{xy}} g(s,t),\\; S_{xy}=${win}`;
    case "midpoint":
      return `\\hat{f}(x,y)=\\tfrac{1}{2}\\Bigl[\\max_{(s,t)\\in S_{xy}} g(s,t)+\\min_{(s,t)\\in S_{xy}} g(s,t)\\Bigr],\\; S_{xy}=${win}`;
    case "salt_pepper_noise":
      return `\\begin{aligned} p(z)&=\\begin{cases} P_p=${fmt(params.p_pepper)} & z=0 \\\\ P_s=${fmt(params.p_salt)} & z=255 \\\\ 1-P_p-P_s & \\text{else}\\end{cases}\\\\ g&=\\begin{cases} 0 & u<P_p \\\\ 255 & P_p\\le u<P_p+P_s \\\\ f & \\text{else}\\end{cases}\\end{aligned}`;
    case "quantization_noise": {
      const delta = 255 / (2 ** params.bits - 1);
      return `\\begin{aligned} \\Delta&=\\dfrac{255}{2^{${params.bits}}-1}=${fmt(delta, 2)}\\\\ q(f)&=\\Delta\\cdot\\mathrm{round}(f/\\Delta)\\\\ \\sigma_e^{2}&\\approx\\Delta^{2}/12 \\end{aligned}`;
    }
    case "rayleigh_noise": {
      // plan 5.3.6: the positive noise mean brightens the image, the UI
      // must say so — show the live mean and standard deviation
      const mean = params.zero_mean ? 0 : params.a + Math.sqrt((Math.PI * params.b) / 4);
      const std = Math.sqrt((params.b * (4 - Math.PI)) / 4);
      const note = params.zero_mean ? "\\;\\text{(mean subtracted)}" : "";
      return `\\begin{aligned} p(z)&=\\frac{2}{b}(z-a)\\,e^{-(z-a)^{2}/b},\\; z\\ge a\\\\ g&=\\mathrm{clip}(f+z)\\\\ \\text{mean}&=${fmt(mean, 1)},\\;\\text{std}=${fmt(std, 1)}${note} \\end{aligned}`;
    }
    case "uniform_noise": {
      const std = (params.b - params.a) / Math.sqrt(12);
      return `\\begin{aligned} p(z)&=\\frac{1}{b-a},\\; ${params.a}\\le z\\le ${params.b}\\\\ g&=\\mathrm{clip}(f+z)\\\\ \\text{mean}&=${fmt((params.a + params.b) / 2, 1)},\\;\\text{std}=${fmt(std, 1)} \\end{aligned}`;
    }
    case "band_reject":
    case "band_pass": {
      const c0 = fmt(params.c0, 2);
      const w = fmt(params.width, 2);
      const prefix = tool === "band_pass"
        ? `H_{\\mathrm{BP}}&=1-H_{\\mathrm{BR}}\\\\ `
        : "";
      let hbr;
      if (params.kind === "ideal") {
        hbr = `H_{\\mathrm{BR}}&=\\begin{cases}0&D_0-W/2\\le D\\le D_0+W/2\\\\ 1&\\text{else}\\end{cases}`;
      } else if (params.kind === "gaussian") {
        hbr = `H_{\\mathrm{BR}}&=1-\\exp\\!\\Bigl(-\\tfrac12\\bigl[(D^{2}-D_0^{2})/(DW)\\bigr]^{2}\\Bigr)`;
      } else {
        hbr = `H_{\\mathrm{BR}}&=\\bigl(1+[DW/(D^{2}-D_0^{2})]^{2n}\\bigr)^{-1}`;
      }
      const extra = params.kind === "butterworth" ? `,\\; n=${params.n}` : "";
      return `\\begin{aligned} ${prefix}${hbr}\\\\ D_0&=${c0}\\,D_{\\max},\\; W=${w}\\,D_{\\max}${extra} \\end{aligned}`;
    }
    case "wiener":
      return `\\begin{aligned} H_W&=\\frac{S_f}{S_f+S_\\eta},\\quad S_f\\approx\\max\\bigl(|G|^{2}-S_\\eta,0\\bigr)\\\\ S_\\eta&=MN\\sigma_\\eta^{2},\\; \\sigma_\\eta=${params.noise_std} \\end{aligned}`;
    case "kalman":
      return `\\begin{aligned} x_i&=x_{i-1}+w,\\; z_i=x_i+v\\\\ \\mathrm{Var}(w)&=q=${fmt(params.q, 1)},\\; \\mathrm{Var}(v)=r=${params.r} \\end{aligned}`;
    default:
      return null;
  }
}

function Formula({ tool, params, img }) {
  const tex = formulaTex(tool, params, img);
  if (!tex) return null;
  return <MathTex tex={tex} display={tex.startsWith("\\begin")} />;
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
    case "contrast": return `Contrast stretch [${p.r_min}, ${p.r_max}] → [0, 255]`;
    case "gaussian_blur": return `Gaussian blur σ = ${fmt(p.sigma, 1)}`;
    case "median": return `Median filter ${p.size}×${p.size}`;
    case "arithmetic_mean": return `Mean filter ${p.size}×${p.size}`;
    case "geometric_mean": return `Geometric mean ${p.size}×${p.size}`;
    case "harmonic_mean": return `Harmonic mean ${p.size}×${p.size}`;
    case "minimum": return `Min filter ${p.size}×${p.size}`;
    case "maximum": return `Max filter ${p.size}×${p.size}`;
    case "midpoint": return `Midpoint filter ${p.size}×${p.size}`;
    case "salt_pepper_noise": return `Salt & pepper Pₚ = ${fmt(p.p_pepper, 2)}, Pₛ = ${fmt(p.p_salt, 2)}`;
    case "quantization_noise": return `Quantization noise B = ${p.bits} (${2 ** p.bits} levels)`;
    case "rayleigh_noise": return `Rayleigh noise a = ${p.a}, b = ${p.b}${p.zero_mean ? ", zero-mean" : ""}`;
    case "uniform_noise": return `Uniform noise [${p.a}, ${p.b}]`;
    case "band_reject": return `Band-reject filter (${p.kind}) D₀ = ${fmt(p.c0, 2)}, W = ${fmt(p.width, 2)}`;
    case "band_pass": return `Band-pass filter (${p.kind}) D₀ = ${fmt(p.c0, 2)}, W = ${fmt(p.width, 2)}`;
    case "wiener": return `Wiener filter σ_η = ${p.noise_std}`;
    case "kalman": return `Kalman filter q = ${fmt(p.q, 1)}, r = ${p.r}`;
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
  if (!t.sliders) return null;

  // keep invalid values out of the UI so the user never hits a 422:
  // Pp + Ps <= 1, a < b for uniform noise, b >= 1 for Rayleigh
  const setParam = (key, value) => {
    const p = { ...params, [key]: value };
    if (tool === "salt_pepper_noise") {
      const other = key === "p_pepper" ? "p_salt" : "p_pepper";
      p[key] = Math.min(value, 1 - params[other]);
    }
    if (tool === "contrast") {
      if (key === "r_min") p.r_min = Math.min(value, params.r_max - 1);
      if (key === "r_max") p.r_max = Math.max(value, params.r_min + 1);
    }
    if (tool === "uniform_noise") {
      if (key === "a") p.a = Math.min(value, params.b - 1);
      if (key === "b") p.b = Math.max(value, params.a + 1);
    }
    setParams(p);
  };

  const formatSlider = (s, value) =>
    s.key === "size" ? `${value} × ${value}` : s.step < 1 ? fmt(value, s.step < 0.1 ? 2 : 1) : value;

  return (
    <>
      {(tool === "band_reject" || tool === "band_pass") && (
        <label className="select inline">
          <span>Transfer function (Table 4.7)</span>
          <select value={params.kind} onChange={(e) => setParams({ ...params, kind: e.target.value })}>
            <option value="ideal">Ideal</option>
            <option value="butterworth">Butterworth</option>
            <option value="gaussian">Gaussian</option>
          </select>
        </label>
      )}
      {t.sliders.map((s) => {
        if (s.key === "n" && params.kind !== "butterworth") return null;
        // with zero_mean on, the mean a + √(πb/4) is subtracted, so a has
        // no effect at all — disable the slider and say why
        const disabled = tool === "rayleigh_noise" && s.key === "a" && params.zero_mean;
        return (
          <label className="slider" key={s.key}>
            <span>{s.name}<output>{formatSlider(s, params[s.key])}</output></span>
            <input type="range" min={s.min} max={s.max} step={s.step} value={params[s.key]}
              disabled={disabled}
              onChange={(e) => setParam(s.key, Number(e.target.value))} />
          </label>
        );
      })}
      {tool === "rayleigh_noise" && (
        <>
          <label className="check">
            <input type="checkbox" checked={params.zero_mean}
              onChange={(e) => setParams({ ...params, zero_mean: e.target.checked })} />
            Subtract the mean (no brightening)
          </label>
          {params.zero_mean && <p className="hint">Zero-mean mode: a has no effect.</p>}
        </>
      )}
      {NOISE_TOOLS.has(tool) && (
        <div className="row">
          <button onClick={() => setParams({ ...params, seed: newSeed() })}>New noise pattern</button>
        </div>
      )}
      {tool === "salt_pepper_noise" && (
        <p className="hint">Total corrupted pixels: {fmt((params.p_pepper + params.p_salt) * 100, 1)}%.</p>
      )}
    </>
  );
}

export default function App() {
  const [imageId, setImageId] = useState(null);
  const [original, setOriginal] = useState(null);
  const [committed, setCommitted] = useState(null);
  const [ops, setOps] = useState([]);
  const [redo, setRedo] = useState([]);
  const [category, setCategory] = useState(null);
  // direction of the last move in the tools panel, for the slide animation
  const [navDir, setNavDir] = useState("");
  const [toolsOpen, setToolsOpen] = useState(true);
  const [tool, setTool] = useState(null);
  const [params, setParams] = useState({});
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
  const [sampleIndex, setSampleIndex] = useState(0);
  const [selectedSampleId, setSelectedSampleId] = useState("");
  const [sampleLoading, setSampleLoading] = useState(false);
  const fileRef = useRef(null);
  const timer = useRef(0);
  const toolsRef = useRef(null);
  const previewReq = useRef(0);

  useEffect(() => onRequest((e) => setLog((l) => [e, ...l].slice(0, 100))), []);
  useEffect(() => onStatus(setBackend), []);
  useEffect(() => { checkBackend(); }, []);

  // keep the session up to date; only images stored on the backend can be
  // restored (in the offline mode they live in this page only)
  useEffect(() => {
    if (!imageId || backend !== "server") return;
    writeSession({ imageId, fileName, sampleId: selectedSampleId, ops, redo, category, toolsOpen });
  }, [imageId, fileName, selectedSampleId, ops, redo, category, toolsOpen, backend]);

  // after a reload, bring the last session back once the backend answers
  const restoreTried = useRef(false);
  useEffect(() => {
    if (backend !== "server" || restoreTried.current) return;
    restoreTried.current = true;
    const saved = readSession();
    if (!saved || original) return;
    (async () => {
      setBusy(true);
      try {
        const img = await fetchOriginal(saved.imageId);
        if (!img) {
          writeSession(null);
          setError("The backend was restarted, so your last image is gone. Load it again to continue.");
          return;
        }
        start(saved.imageId, img, "Restored your last session.", saved.fileName, saved.sampleId);
        if (saved.ops?.length) {
          setCommitted(await processImage(saved.imageId, saved.ops));
          setOps(saved.ops);
        }
        setRedo(saved.redo || []);
        setCategory(saved.category ?? null);
        setToolsOpen(saved.toolsOpen ?? true);
      } catch (e) {
        setError(e.message);
      } finally {
        setBusy(false);
      }
    })();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [backend]);

  // Scrolling down over the tools panel moves the page first, until the
  // panel reaches its sticky position and is fully on screen; only then
  // does the panel scroll its own content. Scrolling up empties the
  // panel first, then the page follows (normal scroll chaining).
  useEffect(() => {
    const panel = toolsRef.current;
    if (!panel) return;
    function onWheel(e) {
      if (e.deltaY <= 0 || getComputedStyle(panel).position !== "sticky") return;
      const stickyTop = parseFloat(getComputedStyle(panel).top) || 0;
      const pageEnd = document.documentElement.scrollHeight - window.innerHeight;
      if (panel.getBoundingClientRect().top <= stickyTop + 1 || window.scrollY >= pageEnd - 1) return;
      e.preventDefault();
      const unit = e.deltaMode === 1 ? 16 : e.deltaMode === 2 ? window.innerHeight : 1;
      window.scrollBy(0, e.deltaY * unit);
    }
    panel.addEventListener("wheel", onWheel, { passive: false });
    return () => panel.removeEventListener("wheel", onWheel);
  }, [original]);

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

  const neutral = isNeutral(tool, params, committed);

  // live preview of the tool being adjusted
  useEffect(() => {
    if (!committed) return;
    clearTimeout(timer.current);
    if (!tool || tool === "crop" || tool === "grayscale" || neutral || !dirty) {
      setPreview(null); setComputing(false); return;
    }
    // backend-only tools are not in imageOps.js: their preview is computed
    // by the backend with the applied steps plus the step being edited.
    // The slider fires many requests, so debounce; replies can arrive out
    // of order, so only the latest request may update the state.
    if (BACKEND_ONLY.has(tool)) {
      if (backend !== "server") { setPreview(null); setComputing(false); return; }
      const id = ++previewReq.current;
      setComputing(true);
      timer.current = setTimeout(async () => {
        try {
          const out = await processImage(imageId, [...ops, { type: tool, params: opParams(tool, params) }]);
          if (id === previewReq.current) setPreview(out);
        } catch (e) {
          if (id === previewReq.current) { setPreview(null); setError(e.message); }
        } finally {
          if (id === previewReq.current) setComputing(false);
        }
      }, 300);
      return () => clearTimeout(timer.current);
    }
    // local preview with applyOp (same result as the backend for these ops)
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
  }, [committed, tool, params, neutral, dirty, backend, imageId, ops]);

  function start(id, img, note = "", name = "image", sampleId = "") {
    setImageId(id);
    setFileName(name.replace(/\.[^.]+$/, "") || "image");
    setExported("");
    setOriginal(img);
    setCommitted(img);
    setOps([]);
    setRedo([]);
    setError("");
    setNotice(note);
    setSelectedSampleId(sampleId);
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

  async function loadBuiltin(sample) {
    setSampleLoading(true);
    setError("");
    try {
      const { imageId: id, image, scaled } = await uploadFromUrl(sample.url, sample.file);
      start(id, image, scaled ? "Large image: reduced to 1600 px on the longest side." : "", sample.file, sample.id);
    } catch (e) {
      setError(e.message);
    } finally {
      setSampleLoading(false);
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

  // Closing the tool discards its unapplied change, like Cancel.
  function closeTool() {
    setTool(null);
    setParams({});
    setDirty(false);
  }

  function openCategory(id) {
    setNavDir("forward");
    setCategory(id);
    closeTool();
  }

  function backToCategories() {
    setNavDir("back");
    setCategory(null);
    closeTool();
  }

  function selectTool(t) {
    setTool(t);
    setParams(defaultParams(t, committed));
    // these tools have no "do nothing" value, so show their default effect right away
    setDirty(PREVIEW_ON_OPEN.has(t));
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

  const t = tool ? TOOLS[tool] : null;
  const cat = CATEGORIES.find((c) => c.id === category);
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
              <button onClick={loadSample}>Use generated test image</button>
            </div>
            {error && <p className="error" role="alert">{error}</p>}
          </div>
        </main>
      ) : (
        <main className={"work" + (toolsOpen ? "" : " tools-closed")}>
          <aside ref={toolsRef} className={"tools panel" + (toolsOpen ? "" : " closed")} aria-label="Tools">
            <div className="tools-head">
              <button type="button" className="icon-btn" onClick={() => setToolsOpen((o) => !o)}
                aria-expanded={toolsOpen} aria-controls="tools-body"
                aria-label={toolsOpen ? "Hide tools" : "Show tools"} title={toolsOpen ? "Hide tools" : "Show tools"}>
                {ICONS.menu}
              </button>
              <h3 className="tools-title">{cat ? cat.label : "Tools"}</h3>
              {cat && (
                <button type="button" className="icon-btn back" onClick={backToCategories}
                  aria-label="Back to categories" title="Back to categories">
                  {ICONS.back}
                </button>
              )}
            </div>
            <span className="tools-rail" aria-hidden="true">Tools</span>

            <div className="tools-body" id="tools-body" inert={toolsOpen ? undefined : ""}>
              <div key={cat ? cat.id : "root"} className={"tools-view " + navDir}>
                {!cat ? (
                  <div className="category-list">
                    {CATEGORIES.map((c) => (
                      <button key={c.id} type="button" className="category" onClick={() => openCategory(c.id)}>
                        <span className="category-icon">{ICONS[c.id]}</span>
                        <span className="category-label">{c.label}</span>
                        <span className="category-chevron">{ICONS.chevron}</span>
                      </button>
                    ))}
                  </div>
                ) : (
                  <>
                    <div className="tool-list" role="tablist" aria-label={cat.label}>
                      {cat.tools.map((k) => {
                        const v = TOOLS[k];
                        const offline = v.backendOnly && backend !== "server";
                        return (
                          <button key={k} role="tab" aria-selected={tool === k}
                            disabled={offline}
                            title={offline ? "Needs backend" : undefined}
                            className={"tool" + (tool === k ? " on" : "")} onClick={() => selectTool(k)}>
                            {v.label}
                            {offline && <span className="note"> (needs backend)</span>}
                          </button>
                        );
                      })}
                    </div>

                    {!t ? (
                      <p className="hint">Choose a tool to see its formula and controls.</p>
                    ) : (
                      <>
                        <div className="formula">
                          <Formula tool={tool} params={params} img={committed} />
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
                      </>
                    )}
                  </>
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
            </div>
          </aside>

          <div className="stage">
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
          </div>
        </main>
      )}

      <input ref={fileRef} type="file" accept="image/jpeg,image/png,image/tiff,.tif,.tiff" hidden
        onChange={(e) => { handleFile(e.target.files[0]); e.target.value = ""; }} />
    </div>
  );
}
