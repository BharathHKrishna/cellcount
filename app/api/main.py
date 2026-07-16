"""v2: FastAPI version of the app, once it needs to look more like a product."""

import io
import os
import sys

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from PIL import Image
from pydantic import BaseModel

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from segment.cellpose_runner import run_cellpose  # noqa: E402
from segment.overlay import draw_outlines  # noqa: E402
from segment.postprocess import measure_instances  # noqa: E402

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models")
UNET_WEIGHTS = os.path.join(MODELS_DIR, "unet_v1.pt")

app = FastAPI(title="CellCount API")


class Measurement(BaseModel):
    label: int
    area_px: int
    centroid: tuple[float, float]


class SegmentResponse(BaseModel):
    count: int
    measurements: list[Measurement]


@app.get("/health")
def health():
    return {"status": "ok"}


def _load_image(upload_bytes):
    return np.array(Image.open(io.BytesIO(upload_bytes)).convert("RGB"))


def _run_model(image, model):
    if model == "unet":
        from segment.unet_model import run_unet
        return run_unet(image, weights_path=UNET_WEIGHTS)
    return run_cellpose(image)


@app.post("/segment", response_model=SegmentResponse)
async def segment(file: UploadFile = File(...), model: str = "cellpose"):
    image = _load_image(await file.read())
    masks = _run_model(image, model)
    measurements = measure_instances(masks)
    return {"count": len(measurements), "measurements": measurements}


@app.post("/segment/overlay")
async def segment_overlay(file: UploadFile = File(...), model: str = "cellpose"):
    image = _load_image(await file.read())
    masks = _run_model(image, model)
    overlay = draw_outlines(image, masks)
    buf = io.BytesIO()
    Image.fromarray(overlay).save(buf, format="PNG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")


@app.post("/segment/csv")
async def segment_csv(file: UploadFile = File(...), model: str = "cellpose"):
    image = _load_image(await file.read())
    masks = _run_model(image, model)
    measurements = measure_instances(masks)
    df = pd.DataFrame(measurements)
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    return StreamingResponse(iter([buf.getvalue()]), media_type="text/csv",
                              headers={"Content-Disposition": "attachment; filename=measurements.csv"})
