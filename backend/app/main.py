"""Image Lab backend (FastAPI).

Run:  uvicorn app.main:app --reload
Docs: http://localhost:8000/docs
"""

import os

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

from .imaging import io as imgio
from .imaging.ops import OPERATIONS, OperationError, run_pipeline
from .schemas import ExportRequest, ProcessRequest, UploadResponse
from .storage import store

MAX_UPLOAD_BYTES = 20 * 1024 * 1024

app = FastAPI(title="Image Lab API", version="0.1.0")

# The React dev server runs on another port, so the browser needs CORS.
origins = os.getenv("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in origins if o.strip()],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok", "operations": sorted(OPERATIONS)}


@app.post("/upload", response_model=UploadResponse)
async def upload(file: UploadFile = File(...)):
    data = await file.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "The file is larger than 20 MB.")
    try:
        rgb, alpha, scaled, fmt = imgio.decode(data)
    except imgio.ImageDecodeError as e:
        raise HTTPException(415, str(e)) from e
    image_id = store.add(rgb, alpha)
    h, w = rgb.shape[:2]
    return UploadResponse(image_id=image_id, width=w, height=h, scaled=scaled, source_format=fmt)


def _run(item, operations):
    try:
        return run_pipeline(item.rgb, item.alpha, operations)
    except OperationError as e:
        raise HTTPException(422, str(e)) from e


def _get(image_id: str):
    item = store.get(image_id)
    if item is None:
        raise HTTPException(404, "Image not found. It may have expired; upload it again.")
    return item


@app.get("/images/{image_id}/original", response_class=Response)
def original(image_id: str):
    """The uploaded image as PNG (TIFF and other formats become viewable)."""
    item = _get(image_id)
    return Response(imgio.encode_png(item.rgb, item.alpha), media_type="image/png")


@app.post("/process", response_class=Response)
def process(req: ProcessRequest):
    """Apply the operation list to the original image and return a PNG.

    The client always sends the full list, so undo/redo only changes the
    list and the server stays stateless apart from the stored original.
    """
    item = _get(req.image_id)
    rgb, alpha = _run(item, req.operations)
    return Response(imgio.encode_png(rgb, alpha), media_type="image/png")


@app.post("/export", response_class=Response)
def export(req: ExportRequest):
    """Apply the operation list and return the result as a downloadable file
    in the chosen format, at the stored resolution."""
    item = _get(req.image_id)
    rgb, alpha = _run(item, req.operations)
    data = imgio.encode(rgb, alpha, req.format, req.quality)
    _, media_type = imgio.EXPORT_FORMATS[req.format]
    ext = "jpg" if req.format == "jpeg" else req.format
    return Response(
        data,
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{req.image_id}_processed.{ext}"'},
    )
