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
| `src/imageOps.js` | Pixel operations (Eqs. 6, 8, 9) and histogram statistics (Eqs. 12–14) |
| `src/api.js` | API layer. Currently simulates the backend in the browser |
| `src/styles.css` | Styles (light/dark theme) |

## Backend contract (planned)

- `POST /upload` — multipart file → `{ image_id, width, height }`
- `POST /process` — `{ image_id, operations: [{ type, params }] }` → `image/png`

When the FastAPI backend is ready, only `src/api.js` needs to change:
replace the simulated `upload` and `processImage` with `fetch` calls to
these endpoints. Slider previews can keep using `imageOps.js` locally.
