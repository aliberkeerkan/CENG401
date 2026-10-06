# Image Lab — frontend prototype

React (Vite) prototype of the Interactive Web-Based Image Processing Platform.

## Run

```bash
npm install
npm run dev
```

Then open http://localhost:5173 in the browser.

## Structure

| File | Role |
|------|------|
| `src/App.jsx` | UI: upload, tools, original/processed view, history, histogram, request log |
| `src/imageOps.js` | Operations for instant previews: crop (4), resize (5), grayscale (6), brightness (8), contrast (9), Gaussian blur (10), median (33); histogram statistics (12–14). Same results as the backend |
| `src/api.js` | API layer. Talks to the FastAPI backend; simulates it in the browser if the backend is not running |
| `src/styles.css` | Styles (light/dark theme) |

## Backend

Start the backend first (see `../backend/README.md`). The status badge in
the top right shows "Backend connected" when the frontend reaches it.
If the backend is not running, the app still works: requests are simulated
in the browser and the badge says so.

The backend address defaults to `http://localhost:8000`. To change it, copy
`.env.example` to `.env.local` and edit `VITE_API_URL`.
