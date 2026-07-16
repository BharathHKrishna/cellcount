"""v2: FastAPI version of the app, once it needs to look more like a product."""

from fastapi import FastAPI

app = FastAPI(title="CellCount API")


@app.get("/health")
def health():
    return {"status": "ok"}
