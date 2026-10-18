# Image Lab — backend

FastAPI service that stores uploaded images and applies image processing
operations with NumPy and Pillow. Formulas follow Section 5 of the project plan.

## Setup (once)

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux
pip install -r requirements.txt
```

## Run

```bash
.venv\Scripts\activate
python run.py
```

`run.py` is the same as `uvicorn app.main:app --reload --host 0.0.0.0 --port 8000`.
`--host 0.0.0.0` lets phones and other PCs on the same network reach the API.

- API: http://localhost:8000 (on another device: `http://<this-pc-ip>:8000`)
- Interactive docs: http://localhost:8000/docs

## Test

```bash
pytest
```

## Endpoints

| Method | Path | Body | Returns |
|--------|------|------|---------|
| GET | `/health` | – | `{ status, operations }` |
| POST | `/upload` | multipart `file` (JPEG, PNG, TIFF; max 20 MB) | `{ image_id, width, height, scaled, source_format }` |
| GET | `/images/{image_id}/original` | – | `image/png` |
| POST | `/process` | `{ image_id, operations: [{ type, params }] }` | `image/png` |
| POST | `/export` | `{ image_id, operations, format: png \| jpeg \| tiff, quality }` | file download |

| Group | Operation types and parameters |
|-------|--------------------------------|
| Basic | `crop` {x, y, width, height}, `resize` {width, height, method}, `grayscale`, `brightness` {beta}, `contrast` {r_min, r_max: 0–255, r_min < r_max} |
| Filters | `gaussian_blur` {sigma}, `median`, `arithmetic_mean`, `geometric_mean`, `harmonic_mean` {size: odd, 3–15} |
| Frequency | `band_reject` / `band_pass` {c0, width, kind, n} (G&W Table 4.7, Eq. 4-148), `wiener` {k} (Eq. 5-85), `kalman` {q, r} |
| Noise | `salt_pepper_noise` {p_pepper, p_salt, seed}, `quantization_noise` {bits: 1–8}, `rayleigh_noise` {a, b, zero_mean, seed}, `uniform_noise` {a, b, seed} |

Limits are listed in `app/schemas.py` and on the `/docs` page.

Random noise uses `seed` (default 0). `/process` re-applies the whole list
on every change, so keep the seed of a noise step fixed to keep the same
noise pattern across previews, undo/redo and `/export`; send a new seed to
draw a new pattern. With `rayleigh_noise`, `zero_mean: true` removes the
mean, so `a` then has no effect.

`/process` always applies the full operation list to the stored original,
so undo/redo only changes the list on the client.

## Structure

```
app/
  main.py            endpoints, CORS
  schemas.py         request/response models (validation)
  storage.py         in-memory image store (prototype)
  imaging/
    io.py            decode uploads (incl. 16-bit TIFF), encode PNG
    ops/
      __init__.py    operation registry and pipeline
      basic.py       grayscale (6), brightness (8), contrast stretching (r_min, r_max)
      geometry.py    crop (4), resize: nearest / bilinear (5)
      filters.py     Gaussian blur (10), mean filters (27)-(29), median (33)
      frequency.py   band-reject / band-pass (G&W 4.10), Wiener (5.85), Kalman
      noise.py       salt-and-pepper (18), quantization (22), Rayleigh (24), uniform (25)
tests/               unit and API tests
```

## Adding an operation

1. Write the function in `app/imaging/ops/` (input and output: float array of shape (H, W, C), 0–255).
2. Add its parameter model and an `...Op` class in `app/schemas.py`, and add it to `Operation`.
3. Register it in `OPERATIONS` in `app/imaging/ops/__init__.py`. Use `geometric=True` if it changes the image size, so the alpha channel is transformed too.
4. Add a test in `tests/`.

## Configuration

- `CORS_ORIGINS`: comma-separated list of allowed frontend origins.
  Default: `http://localhost:5173,http://127.0.0.1:5173`.
  Private network origins (`192.168.x.x`, `10.x.x.x`, `172.16–31.x.x`, any
  port) are always allowed, so a new DHCP address needs no change here.
