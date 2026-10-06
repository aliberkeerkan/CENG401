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
uvicorn app.main:app --reload
```

- API: http://localhost:8000
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

Operation types: `crop`, `resize`, `grayscale`, `brightness`, `contrast`, `gaussian_blur`, `median`.
Parameters and limits are listed in `app/schemas.py` and on the `/docs` page.

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
      basic.py       grayscale (6), brightness (8), contrast (9)
      geometry.py    crop (4), resize: nearest / bilinear (5)
      filters.py     Gaussian blur (10), median filter (33)
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
