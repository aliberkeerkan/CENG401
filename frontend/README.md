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
| `src/App.jsx` | UI: upload, tools, original/processed view, history, histogram, request log |
| `src/imageOps.js` | Operations for instant previews: crop (4), resize (5), grayscale (6), brightness (8), contrast stretching, Gaussian blur (10), median (33); histogram statistics (12–14). Same results as the backend |
| `src/api.js` | API layer. Talks to the FastAPI backend; simulates it in the browser if the backend is not running |
| `src/App.jsx` (backend-only tools) | Salt & pepper (18), quantization (22), Rayleigh (24) and uniform (25) noise; arithmetic (27), geometric (28) and harmonic (29) mean filters. Preview and apply always run on the FastAPI backend — these tools are disabled ("needs backend") when the backend is not connected |
| `src/styles.css` | Styles (light/dark theme) |

## Backend

Start the backend first (see `../backend/README.md`). The status badge in
the top right shows "Backend connected" when the frontend reaches it.
If the backend is not running, the app still works: requests are simulated
in the browser and the badge says so.

The backend address defaults to port 8000 on the host the page was opened
with (`http://localhost:8000`, or `http://192.168.1.20:8000` from a phone).
To force a fixed address, copy `.env.example` to `.env.local` and set
`VITE_API_URL`.
