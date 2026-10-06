import io

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app

client = TestClient(app)


def png_bytes(arr, fmt="PNG"):
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format=fmt)
    return buf.getvalue()


def upload(data, name="test.png"):
    return client.post("/upload", files={"file": (name, data, "application/octet-stream")})


def decode(resp):
    return np.asarray(Image.open(io.BytesIO(resp.content)))


def test_health():
    r = client.get("/health")
    assert r.status_code == 200 and "brightness" in r.json()["operations"]


def test_upload_and_process_pipeline():
    arr = np.full((4, 6, 3), 100, dtype=np.uint8)
    r = upload(png_bytes(arr))
    assert r.status_code == 200
    body = r.json()
    assert (body["width"], body["height"], body["scaled"]) == (6, 4, False)

    ops = [{"type": "brightness", "params": {"beta": 30}}, {"type": "grayscale", "params": {}}]
    r = client.post("/process", json={"image_id": body["image_id"], "operations": ops})
    assert r.status_code == 200 and r.headers["content-type"] == "image/png"
    assert np.all(decode(r) == 130)


def test_empty_pipeline_returns_original():
    arr = np.random.default_rng(1).integers(0, 256, (5, 5, 3), dtype=np.uint8)
    image_id = upload(png_bytes(arr)).json()["image_id"]
    r = client.post("/process", json={"image_id": image_id, "operations": []})
    assert np.array_equal(decode(r), arr)


def test_tiff_16bit_is_converted_and_viewable():
    arr = np.linspace(0, 65535, 64, dtype=np.uint16).reshape(8, 8)
    r = upload(png_bytes(arr, fmt="TIFF"), name="scan.tif")
    assert r.status_code == 200 and r.json()["source_format"] == "TIFF"
    png = client.get(f"/images/{r.json()['image_id']}/original")
    out = decode(png)
    assert out.shape == (8, 8, 3) and out.min() == 0 and out.max() == 255


def test_large_image_is_scaled():
    arr = np.zeros((2000, 1000, 3), dtype=np.uint8)
    body = upload(png_bytes(arr)).json()
    assert body["scaled"] and max(body["width"], body["height"]) == 1600


def test_alpha_channel_is_preserved():
    arr = np.zeros((3, 3, 4), dtype=np.uint8)
    arr[..., 3] = 128
    image_id = upload(png_bytes(arr)).json()["image_id"]
    r = client.post("/process", json={"image_id": image_id, "operations": [{"type": "brightness", "params": {"beta": 50}}]})
    out = decode(r)
    assert out.shape == (3, 3, 4) and np.all(out[..., 3] == 128) and np.all(out[..., 0] == 50)


def test_errors():
    assert upload(b"not an image", name="x.txt").status_code == 415
    assert client.post("/process", json={"image_id": "missing", "operations": []}).status_code == 404
    image_id = upload(png_bytes(np.zeros((2, 2, 3), dtype=np.uint8))).json()["image_id"]
    bad = {"image_id": image_id, "operations": [{"type": "contrast", "params": {"alpha": -1}}]}
    assert client.post("/process", json=bad).status_code == 422
    unknown = {"image_id": image_id, "operations": [{"type": "sepia", "params": {}}]}
    assert client.post("/process", json=unknown).status_code == 422


def test_export_formats_apply_operations():
    arr = np.full((4, 4, 3), 100, dtype=np.uint8)
    image_id = upload(png_bytes(arr)).json()["image_id"]
    ops = [{"type": "brightness", "params": {"beta": 50}}]
    for fmt, mime, pil in [("png", "image/png", "PNG"), ("jpeg", "image/jpeg", "JPEG"), ("tiff", "image/tiff", "TIFF")]:
        r = client.post("/export", json={"image_id": image_id, "operations": ops, "format": fmt})
        assert r.status_code == 200 and r.headers["content-type"] == mime
        assert "attachment" in r.headers["content-disposition"]
        img = Image.open(io.BytesIO(r.content))
        assert img.format == pil
        assert abs(int(np.asarray(img)[0, 0, 0]) - 150) <= 1  # JPEG may differ by 1


def test_export_jpeg_flattens_alpha_on_white():
    arr = np.zeros((2, 2, 4), dtype=np.uint8)  # fully transparent black
    image_id = upload(png_bytes(arr)).json()["image_id"]
    r = client.post("/export", json={"image_id": image_id, "operations": [], "format": "jpeg"})
    assert np.all(np.asarray(Image.open(io.BytesIO(r.content))) >= 250)


def test_export_rejects_unknown_format():
    image_id = upload(png_bytes(np.zeros((2, 2, 3), dtype=np.uint8))).json()["image_id"]
    r = client.post("/export", json={"image_id": image_id, "operations": [], "format": "bmp"})
    assert r.status_code == 422


def test_new_operations_through_api():
    arr = np.random.default_rng(5).integers(0, 256, (40, 60, 3), dtype=np.uint8)
    image_id = upload(png_bytes(arr)).json()["image_id"]
    ops = [
        {"type": "crop", "params": {"x": 10, "y": 5, "width": 30, "height": 20}},
        {"type": "resize", "params": {"width": 60, "height": 40, "method": "bilinear"}},
        {"type": "gaussian_blur", "params": {"sigma": 1.2}},
        {"type": "median", "params": {"size": 5}},
    ]
    r = client.post("/process", json={"image_id": image_id, "operations": ops})
    assert r.status_code == 200 and decode(r).shape == (40, 60, 3)
    r = client.post("/export", json={"image_id": image_id, "operations": ops, "format": "tiff"})
    assert r.status_code == 200 and Image.open(io.BytesIO(r.content)).size == (60, 40)


def test_crop_outside_returns_422_and_even_median_rejected():
    image_id = upload(png_bytes(np.zeros((10, 10, 3), dtype=np.uint8))).json()["image_id"]
    bad_crop = [{"type": "crop", "params": {"x": 5, "y": 5, "width": 8, "height": 8}}]
    r = client.post("/process", json={"image_id": image_id, "operations": bad_crop})
    assert r.status_code == 422 and "outside" in r.json()["detail"]
    even = [{"type": "median", "params": {"size": 4}}]
    assert client.post("/process", json={"image_id": image_id, "operations": even}).status_code == 422


def test_crop_and_resize_transform_alpha_too():
    arr = np.zeros((10, 10, 4), dtype=np.uint8)
    arr[:, :5, 3] = 255  # left half opaque
    image_id = upload(png_bytes(arr)).json()["image_id"]
    ops = [{"type": "crop", "params": {"x": 0, "y": 0, "width": 5, "height": 10}},
           {"type": "resize", "params": {"width": 10, "height": 20, "method": "nearest"}}]
    out = decode(client.post("/process", json={"image_id": image_id, "operations": ops}))
    assert out.shape == (20, 10, 4) and np.all(out[..., 3] == 255)
