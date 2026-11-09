# Image Lab — frontend prototype

React (Vite) prototype of the Interactive Web-Based Image Processing Platform.

## Run

```bash
npm install
npm run dev
```

Then open http://localhost:5173 in the browser.

To use the app from a phone on the same Wi-Fi, open the **Network** URL that
Vite prints (for example `http://192.168.1.20:5173`). Vite listens on all
interfaces (`host: true`); start the backend with `python run.py` so it does
too. If the phone cannot connect, allow ports 5173 and 8000 for private
networks in Windows Firewall.

## Structure

| File | Role |
|------|------|
| `src/App.jsx` | UI: upload, tools, original/processed view, history. An Edit / Compare / Camera switch sits above the samples: Compare shows the base image plus up to five tools (A–E), each applied to that image on its own |
| `src/imageOps.js` | Operations for instant previews: crop (4), resize (5), grayscale (6), brightness (8), contrast stretching, Gaussian blur (10), median (33); histogram statistics (12–14). Same results as the backend |
| `src/api.js` | API layer. Talks to the FastAPI backend; simulates it in the browser if the backend is not running |
| `src/App.jsx` (backend-only tools) | Salt & pepper (18), quantization (22), Rayleigh (24), uniform (25), exponential (26), gamma / speckle (20)–(21) and Poisson (19) noise; a Denoise category for salt & pepper, Poisson, quantization and uniform; sharpen (unsharp mask); arithmetic (27), geometric (28) and harmonic (29) mean filters; min (31), max (32), midpoint (34), alpha-trimmed mean (35) and adaptive median (5.4.3) filters; band-reject (40), band-pass (41), Wiener (44) and a 1-D Kalman smoother; an Advanced tools category with line detection (six classic methods chosen from a menu: G&W masks, SHT with a Hough-space view, PHT, LSD, EDLines, Hessian ridges; no ML) and edge detection with five methods chosen from buttons (Sobel, Prewitt, Roberts, LoG, Canny) and Harris–Stephens corner detection. Line, edge and corner detection update the image only when Apply is pressed; Apply again replaces that step. Preview and apply always run on the FastAPI backend — these tools are disabled ("needs backend") when the backend is not connected. Hovering the processed image reads the pixel coordinate and its value. |
| `src/styles.css` | Styles (light/dark theme) |

## Camera

The Camera tab (Edit / Compare / Camera) processes a live camera picture. Choose the camera
in the Device menu (the browser lists the cameras by name once camera permission is given)
and press Start camera. One window shows the processed picture, with the frame rate (the
processed frames shown per second) in its top-left corner. `src/Camera.jsx`.

Only tools that keep up with a live picture are offered: Basic tools (grayscale,
brightness, contrast, Gaussian blur, sharpen), Noise (all seven models; random noise draws
a new pattern on every frame), Filters (mean, geometric, harmonic, median, min, max,
midpoint, window up to 7 x 7), Frequency (band-reject, band-pass, Wiener and Kalman),
Denoise (all eight noise-removal tools, with windows up to 7 x 7) and Advanced tools
(line, edge and corner detection with all their methods and views, including the Hough
space). The open tool runs live; Add to chain
keeps it while another tool is tried, and the chain runs on every frame in order. A
detection tool replaces detection steps at the end of the chain, so it never runs on another
detector's drawing.

Frames go to the backend's `/process-frame` with `fast=true` at up to 640 px on the longest
side, one at a time: a new frame is sent only when the previous result has arrived, so a
slow filter lowers the frame rate instead of building up a delay. With `fast` the window
filters use OpenCV versions of the same formulas (`backend/app/imaging/ops/fast.py`). The
camera needs the backend, and a secure page: `http://localhost` works, a LAN address such
as `http://192.168.1.20:5173` is blocked by the browser unless it is served over https.

## Backend

Start the backend first (see `../backend/README.md`). The status badge in
the top right shows "Backend connected" when the frontend reaches it.
If the backend is not running, the app still works: requests are simulated
in the browser and the badge says so.

The backend address defaults to port 8000 on the host the page was opened
with (`http://localhost:8000`, or `http://192.168.1.20:8000` from a phone).
To force a fixed address, copy `.env.example` to `.env.local` and set
`VITE_API_URL`.
